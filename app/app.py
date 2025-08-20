import streamlit as st
import pandas as pd
import os
import re
from collections import Counter
import itertools
import io
import zipfile
import json
import requests
import firebase_admin
from firebase_admin import credentials, firestore

# =============================
# App meta (MUST BE FIRST)
# =============================
st.set_page_config(page_title="🪶 Fly Tying Recommender", page_icon="🪶", layout="wide")

# =============================
# Firebase Admin Setup
# =============================
DB = None
try:
    if not firebase_admin._apps:
        # st.secrets is natively supported in Streamlit Community Cloud
        service_account_info = st.secrets.get("gcp_service_account")
        if service_account_info:
            cred = credentials.Certificate(dict(service_account_info))
            firebase_admin.initialize_app(cred)
            DB = firestore.client()
        else:
            st.error("Firebase service account credentials not found in secrets.")
except Exception as e:
    st.error(f"Firebase initialization failed: {e}")

# =============================
# GitHub & Local Data Loaders
# =============================
@st.cache_data
def gh_url(path: str) -> str | None:
    """Build a Raw GitHub URL from .streamlit/secrets.toml [github].raw_base."""
    try:
        base = st.secrets.get("github", {}).get("raw_base")
        if not base:
            base = os.getenv("GITHUB_RAW_BASE")
        return f"{str(base).rstrip('/')}/{str(path).lstrip('/')}" if base else None
    except Exception:
        return None

@st.cache_data
def read_csv_from_github(path: str, *, sep=",") -> pd.DataFrame:
    url = gh_url(path)
    if url:
        try:
            return pd.read_csv(url, encoding="utf-8", sep=sep)
        except Exception:
            pass # Fall through to local
    return pd.read_csv(path, encoding="utf-8", sep=sep)

@st.cache_data
def fetch_bytes_from_github(path: str) -> bytes | None:
    url = gh_url(path)
    if url:
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            return r.content
        except Exception:
            pass # Fall through to local
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None

# =============================
# Authentication Logic (NEW & SIMPLIFIED)
# =============================
# If the user is not logged in, show the login button and stop the app execution.
if not st.user:
    st.title("🪶 Welcome to the Fly Tying Recommender")
    st.info("Please log in with Google to continue and save your inventory.")
    
    # The st.login() method will automatically handle the Google OAuth flow
    if st.button("Log in with Google"):
        st.login(provider="google")
        
    st.stop()

# --- If logged in, the app continues to run from here ---

# =============================
# UI — Hero and Logout Button
# =============================
st.markdown(
    f"""
    <div style="padding: 1.2rem; border-radius: 16px; background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); color: #e2e8f0; border: 1px solid #334155;">
      <h1 style="margin: 0 0 .4rem 0; font-size: 1.8rem;">🪶 Fly Tying Recommender</h1>
      <p style="margin: 0;">Welcome, {st.user.name}! See what you can tie, what you're missing, and what to buy next.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.button("Log out"):
    st.logout()

# =============================
# Cloud Sync Functions (Updated for st.user)
# =============================
def get_user_doc_id() -> str | None:
    """Returns the user's email to use as a unique document ID in Firestore."""
    if st.user:
        return st.user.email
    return None

def save_user_inventory(inv_df: pd.DataFrame) -> bool:
    doc_id = get_user_doc_id()
    if DB is None or not doc_id:
        st.error("You must be logged in to save.")
        return False
    try:
        # Using set with merge=True will create the document if it doesn't exist, or update it if it does.
        DB.collection("users").document(doc_id).set(
            {"inventory": inv_df.to_dict(orient="records")}, merge=True
        )
        return True
    except Exception as e:
        st.error(f"Failed to save inventory: {e}")
        return False

def load_user_inventory() -> pd.DataFrame | None:
    doc_id = get_user_doc_id()
    if DB is None or not doc_id:
        return None
    try:
        doc_ref = DB.collection("users").document(doc_id)
        doc = doc_ref.get()
        if doc.exists:
            inventory_data = doc.to_dict().get("inventory", [])
            return pd.DataFrame(inventory_data)
        return None # Return None if no document exists for the user
    except Exception as e:
        st.error(f"Failed to load inventory: {e}")
        return None

# =============================
# Helpers — normalization, parsing, matching (Your existing functions)
# =============================
def normalize(token: str) -> str:
    if not isinstance(token, str):
        return ""
    return " ".join(token.strip().lower().split())

def strip_label(token: str) -> str:
    if not isinstance(token, str):
        return ""
    t = token.strip().lower()
    return t.split(":", 1)[1].strip() if ":" in t else t

def split_materials(materials_cell: str) -> list[str]:
    if not isinstance(materials_cell, str) or not materials_cell.strip():
        return []
    return [p for p in [normalize(p) for p in materials_cell.split(";")] if p]

def coerce_flies_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize(c) for c in df.columns]
    synonyms = {
        "fly": "fly_name", "flyname": "fly_name", "name": "fly_name",
        "pattern": "fly_name", "tutorial": "tutorials", "links": "tutorials", "link": "tutorials",
    }
    for old, new in synonyms.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)
    missing = [c for c in ["fly_name", "type", "species", "materials"] if c not in df.columns]
    if missing:
        raise ValueError(f"flies.csv must have headers: fly_name,type,species,materials. Missing: {missing}.")
    if "tutorials" not in df.columns:
        df["tutorials"] = ""
    return df

def build_flies_df(raw: pd.DataFrame) -> pd.DataFrame:
    df = coerce_flies_columns(raw)
    df["species"] = df["species"].fillna("").apply(
        lambda s: [normalize(x) for x in str(s).split(";") if str(s).strip()]
    )
    df["materials_list"] = df["materials"].apply(split_materials)
    return df

@st.cache_data
def load_flies_local(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8", engine="python")
    return build_flies_df(raw)

@st.cache_data
def load_subs_local() -> dict[str, set[str]]:
    d = {}
    try:
        sub_df = read_csv_from_github("data/substitutions.csv")
        for _, row in sub_df.iterrows():
            base = normalize(row.get("material", ""))
            eq_raw = row.get("equivalents", "")
            eqs = set(normalize(e) for e in str(eq_raw).split(";") if e.strip())
            if base:
                d[base] = eqs
    except Exception:
        pass
    return d

@st.cache_data
def load_aliases_local() -> dict[str, str]:
    d = {}
    try:
        df = read_csv_from_github("data/aliases.csv")
        for _, row in df.iterrows():
            a = normalize(row.get("alias", ""))
            c = normalize(row.get("canonical", ""))
            if a and c:
                d[a] = c
    except Exception:
        pass
    return d

def parse_color(token: str, color_map: dict[str, str]) -> tuple[str | None, str | None]:
    t = normalize(token)
    words = t.split()
    if not words: return None, None
    last = words[-1]
    if last in color_map: return last, color_map[last]
    for w in words:
        if w in color_map: return w, color_map[w]
    return None, None

def size_to_metric(size_str: str) -> float | None:
    if not size_str: return None
    s = size_str.strip().lower().lstrip("#")
    try:
        return 100 + int(s.replace("/0", "")) if "/0" in s else 100 - int(s)
    except Exception:
        return None

def parse_hook(token: str, hook_map: dict[str, str]) -> tuple[str | None, float | None, str | None]:
    t = normalize(token)
    if not t.startswith("hook:"): return None, None, None
    core = t.split(":", 1)[1].strip()
    size_metric = size_to_metric(re.search(r"#\s*([0-9]+(?:/0)?)", core).group(1)) if re.search(r"#\s*([0-9]+(?:/0)?)", core) else None
    length_tag = re.search(r"\b([0-9]+xl)\b", core).group(1) if re.search(r"\b([0-9]+xl)\b", core) else None
    fam = next((f for kw, f in hook_map.items() if kw in core), None)
    if not fam:
        if "streamer" in core: fam = "streamer"
        elif any(k in core for k in ["nymph", "midge", "scud"]): fam = "nymph"
        elif "dry" in core: fam = "dry"
        elif "jig" in core: fam = "jig"
        elif "wet" in core: fam = "wet"
    return fam, size_metric, length_tag

def apply_alias(token: str, aliases: dict[str, str]) -> str:
    return aliases.get(normalize(token), normalize(token))

def map_aliases_list(items: list[str], aliases: dict[str, str]) -> list[str]:
    return [apply_alias(it, aliases) for it in items] if aliases else items

def tokens_equal_loose(req: str, have: str, color_map: dict[str, str], ignore_labels: bool, ignore_color: bool) -> bool:
    req_token = strip_label(req) if ignore_labels else normalize(req)
    have_token = strip_label(have) if ignore_labels else normalize(have)
    if req_token == have_token: return True
    if ignore_color:
        rc, rf = parse_color(req_token, color_map)
        hc, hf = parse_color(have_token, color_map)
        if rf and rf == hf:
            req_base = req_token.replace(f" {rc}", "") if rc else req_token
            have_base = have_token.replace(f" {hc}", "") if hc else have_token
            if req_base == have_base: return True
    return False

def hook_compatible(req: str, have: str, hook_map: dict[str, str], size_tolerance: int, require_length_match: bool) -> bool:
    if not all(isinstance(s, str) for s in [req, have]): return False
    rfam, rsize, rlen = parse_hook(req, hook_map)
    hfam, hsize, hlen = parse_hook(have, hook_map)
    if not rfam or rfam != hfam: return False
    if require_length_match and (rlen or hlen) and rlen != hlen: return False
    return abs(rsize - hsize) <= size_tolerance if rsize is not None and hsize is not None else True

def expand_with_substitutions(inv_set: set[str], subs: dict[str, set[str]]) -> set[str]:
    expanded = set(inv_set)
    for base, eqs in subs.items():
        if base in inv_set: expanded.update(eqs)
        if any(eq in inv_set for eq in eqs): expanded.add(base)
    return expanded

def compute_matches(flies_df, inv_tokens, aliases_map, subs_map, use_subs, ignore_labels, ignore_color, color_map, hook_map, size_tolerance, require_length_match):
    inv = expand_with_substitutions({apply_alias(tok, aliases_map) for tok in inv_tokens}, subs_map) if use_subs and subs_map else {apply_alias(tok, aliases_map) for tok in inv_tokens}
    results = []
    for _, row in flies_df.iterrows():
        req_list = map_aliases_list(row["materials_list"], aliases_map)
        missing = [mreq for mreq in req_list if not any(
            hook_compatible(mreq, have, hook_map, size_tolerance, require_length_match) if normalize(mreq).startswith("hook:")
            else tokens_equal_loose(mreq, have, color_map, ignore_labels, ignore_color)
            for have in inv
        )]
        results.append({
            "fly_name": row["fly_name"], "type": row["type"], "species": "; ".join(row["species"]),
            "required_count": len(req_list), "missing_count": len(missing), "missing": "; ".join(missing),
        })
    return pd.DataFrame(results)

def best_single_buys(matches_df: pd.DataFrame) -> pd.DataFrame:
    tokens = [t.strip() for cell in matches_df[matches_df["missing_count"] == 1]["missing"] if isinstance(cell, str) for t in cell.split(";") if t.strip()]
    return pd.DataFrame([{"material": m, "unlocks": n} for m, n in Counter(tokens).most_common()]) if tokens else pd.DataFrame(columns=["material", "unlocks"])

def best_pair_buys(matches_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    pairs = [tuple(sorted(toks)) for _, row in matches_df[matches_df["missing_count"] == 2].iterrows() if isinstance(row["missing"], str) and len(toks := [t.strip() for t in row["missing"].split(";") if t.strip()]) == 2]
    return pd.DataFrame([{"materials_pair": " + ".join(p), "unlocks": n} for p, n in Counter(pairs).most_common(top_n)]) if pairs else pd.DataFrame()

def make_shopping_list(matches_df: pd.DataFrame, max_missing: int = 2) -> pd.DataFrame:
    items = Counter([t.strip() for _, row in matches_df[(matches_df["missing_count"] > 0) & (matches_df["missing_count"] <= max_missing)].iterrows() if isinstance(row["missing"], str) for t in row["missing"].split(";") if t.strip()])
    return pd.DataFrame([{"material": m, "appears_in_near_misses": n} for m, n in items.most_common()])

@st.cache_data
def load_hooks_catalog(path: str) -> pd.DataFrame:
    try: return read_csv_from_github(path)
    except: return pd.DataFrame(columns=["brand", "model", "family", "length_tag", "wire", "eye", "barbless", "notes"])

@st.cache_data
def load_brand_prefs(path: str) -> dict[str, list[str]]:
    d = {}
    try:
        df = read_csv_from_github(path)
        for _, row in df.iterrows():
            if cat := normalize(row.get("category", "")):
                d[cat] = [p.strip() for p in str(row.get("preferred_brands", "")).split(";") if p.strip()]
    except: pass
    return d

def suggest_hook_brand_model(req_token, hooks_catalog, brand_prefs, hook_map):
    rfam, _, rlen = parse_hook(req_token, hook_map)
    if not rfam: return None, None
    df = hooks_catalog[hooks_catalog["family"].fillna("").str.lower() == rfam]
    if rlen and not (df2 := df[df["length_tag"].fillna("").str.lower() == rlen.lower()]).empty: df = df2
    if df.empty: return None, None
    if prefs := brand_prefs.get("hook", []):
        for pb in prefs:
            if not (cand := df[df["brand"].str.lower() == pb.lower()]).empty:
                row = cand.iloc[0]
                return row["brand"], row["model"]
    row = df.iloc[0]
    return row["brand"], row["model"]

def enrich_buy_suggestions(df, prefer_brands, hooks_catalog, brand_prefs, hook_map):
    if df is None or df.empty or "material" not in df.columns: return df
    suggestions = [
        suggest_hook_brand_model(m, hooks_catalog, brand_prefs, hook_map) if normalize(m).startswith("hook:") and prefer_brands else (None, None)
        for m in df["material"]
    ]
    df = df.copy()
    df["suggested_brand"] = [s[0] or "" for s in suggestions]
    df["suggested_model"] = [s[1] or "" for s in suggestions]
    return df

def safe_read_csv(file_or_path, required_cols: list[str] | None = None) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1"):
        for sep in (",", ";"):
            try:
                # Reset stream position if it's a file-like object
                if hasattr(file_or_path, 'seek'):
                    file_or_path.seek(0)
                df = pd.read_csv(file_or_path, encoding=enc, sep=sep, engine="python")
                if required_cols and any(c not in df.columns for c in required_cols):
                    continue
                return df
            except Exception:
                continue
    raise ValueError("Could not parse CSV. Please check headers and delimiter.")

def first_http_link(text: str) -> str:
    if not isinstance(text, str): return ""
    return next((u for part in text.split(";") if (u := part.strip()).startswith("http")), "")

# =============================
# Main App Logic Starts Here
# =============================

# --- Auto-load user's inventory from Firestore on first run after login
if "inventory_df" not in st.session_state:
    cloud_inv = load_user_inventory()
    if cloud_inv is not None and not cloud_inv.empty:
        # Ensure all required columns exist
        for col in ["material", "status", "brand", "model"]:
            if col not in cloud_inv.columns:
                cloud_inv[col] = ""
        st.session_state.inventory_df = cloud_inv[["material", "status", "brand", "model"]].fillna("")
        st.toast("✅ Loaded your inventory from the cloud!", icon="☁️")
    else:
        st.session_state.inventory_df = pd.DataFrame(columns=["material", "status", "brand", "model"])

# Initialize other session state variables
st.session_state.setdefault("matches_df", None)
st.session_state.setdefault("matches_sim", None)

# --- Sidebar
with st.sidebar:
    st.info("Download templates, upload your recipe CSV, and set matching options.")
    with st.expander("📄 Examples & Templates"):
        st.download_button("⬇️ Download All Templates (ZIP)", fetch_bytes_from_github("data/templates.zip"), "templates.zip", "application/zip")

    st.header("1) Load Recipes")
    flies_file = st.file_uploader("Upload your flies.csv", type=["csv"], key="flies_csv")
    if flies_file:
        try:
            raw_flies = safe_read_csv(io.BytesIO(flies_file.getvalue()), required_cols=["fly_name", "materials"])
            flies_df = build_flies_df(raw_flies)
        except Exception as e:
            st.error(f"flies.csv parse issue: {e}")
            st.stop()
    else:
        st.markdown("Using bundled **data/flies.csv**")
        flies_df = load_flies_local("data/flies.csv")

    st.header("2) Substitutions & Aliases")
    subs_map = load_subs_local()
    aliases_map = load_aliases_local()
    st.write(f"Loaded {len(subs_map)} subs & {len(aliases_map)} aliases.")

    st.header("3) Matching Options")
    use_subs = st.checkbox("Use substitutions", value=True)
    ignore_labels = st.checkbox("Ignore part labels", value=True)
    ignore_color = st.checkbox("Ignore color variations", value=True)
    size_tol = st.slider("Hook size tolerance", 0, 8, 2)
    require_len = st.checkbox("Require hook length match", value=False)
    prefer_brands = st.checkbox("Suggest preferred brands", value=True)

# Load static data
color_map = {normalize(r.get("color", "")): normalize(r.get("family", "")) for _, r in read_csv_from_github("data/color_families.csv").iterrows()}
hook_map = {normalize(r.get("keyword", "")): normalize(r.get("family", "")) for _, r in read_csv_from_github("data/hooks_map.csv").iterrows()}
hooks_catalog = load_hooks_catalog("data/hooks_catalog.csv")
brand_prefs = load_brand_prefs("data/brand_prefs.csv")

# --- Main App Content
st.header("📝 Manage Your Inventory")
st.info("This inventory is saved to your account. Add, edit, or remove items below, then click 'Save Inventory'.")

edited_df = st.data_editor(
    st.session_state.inventory_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "material": st.column_config.TextColumn("Material Name", required=True),
        "status": st.column_config.SelectboxColumn("Status", options=["NEW", "HALF", "LOW", "OUT", ""], required=True),
        "brand": st.column_config.TextColumn("Brand"),
        "model": st.column_config.TextColumn("Model / Size"),
    },
    key="inventory_editor_main",
)
st.session_state.inventory_df = edited_df

c1, c2 = st.columns(2)
with c1:
    if st.button("☁️ Save Inventory to Cloud", use_container_width=True, type="primary"):
        if save_user_inventory(st.session_state.inventory_df):
            st.success("Saved your inventory to the cloud.")
with c2:
    inv_uploader = st.file_uploader("📤 Upload CSV to Editor", type=['csv'], key="inv_uploader")
    if inv_uploader:
        try:
            uploaded_df = safe_read_csv(io.BytesIO(inv_uploader.getvalue()), required_cols=["material"])
            for col in ["material", "status", "brand", "model"]:
                if col not in uploaded_df.columns: uploaded_df[col] = ""
            st.session_state.inventory_df = uploaded_df[["material", "status", "brand", "model"]].fillna("")
            st.success("CSV loaded. Click 'Save Inventory' to persist changes.")
            st.rerun()
        except Exception as e:
            st.error(f"Could not parse uploaded inventory CSV: {e}")

st.markdown("---")

if st.button("▶️ Run Matching", type="primary", use_container_width=True):
    if 'flies_df' not in locals() or flies_df.empty:
        st.error("Recipes not loaded. Please upload a flies.csv file.")
        st.stop()

    inv_df = st.session_state.get("inventory_df", pd.DataFrame())
    inv_tokens = set(inv_df.loc[inv_df.get("status", "").astype(str).str.upper() != "OUT", "material"].dropna().map(normalize)) if not inv_df.empty else set()
    if not inv_tokens: st.warning("Your inventory is empty.")

    matches_df = compute_matches(flies_df, inv_tokens, aliases_map, subs_map, use_subs, ignore_labels, ignore_color, color_map, hook_map, size_tol, require_len)
    matches_df = matches_df.merge(flies_df[["fly_name", "tutorials"]], on="fly_name", how="left")
    matches_df["tutorial"] = matches_df["tutorials"].apply(first_http_link)
    matches_df.drop(columns=["tutorials"], inplace=True)
    st.session_state.matches_df = matches_df
    st.session_state.matches_sim = None

if st.session_state.matches_df is not None:
    matches_df = st.session_state.matches_df
    st.subheader("Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("✅ Can tie now", int((matches_df["missing_count"] == 0).sum()))
    c2.metric("🟡 Missing 1", int((matches_df["missing_count"] == 1).sum()))
    c3.metric("🟠 Missing 2", int((matches_df["missing_count"] == 2).sum()))
    
    # --- (Your results display logic: filters, tabs, etc. - can be pasted here without changes)
    types = sorted(flies_df["type"].dropna().unique().tolist())
    species_all = sorted(set(itertools.chain.from_iterable(flies_df["species"].tolist())))
    
    with st.expander("Filters and Display Options"):
        col1, col2, col3 = st.columns(3)
        with col1: type_filter = st.multiselect("Filter by type", types, default=types)
        with col2: species_filter = st.multiselect("Filter by species", species_all, default=species_all)
        with col3: show_cols = st.multiselect("Columns to show", matches_df.columns.tolist(), default=["fly_name", "type", "missing_count", "missing", "tutorial"])
        fly_query = st.text_input("🔎 Search by fly name", "")
        near_miss_cap = st.slider("Near-miss threshold (max missing items)", 2, 6, 3)

    def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
        mask_type = df["type"].isin(type_filter)
        mask_species = df["species"].apply(lambda s: any(sp in s for sp in species_filter))
        mask_query = df["fly_name"].str.contains(fly_query.strip(), case=False, na=False) if fly_query.strip() else True
        return df[mask_type & mask_species & mask_query][show_cols]

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["✅ Tie now", "🟡 Near Misses", "🛒 Buy Suggestions", "🔮 What-if", "📦 Full Report"])

    with tab1:
        st.dataframe(apply_filters(matches_df[matches_df["missing_count"] == 0]), use_container_width=True)
    with tab2:
        st.dataframe(apply_filters(matches_df[(matches_df["missing_count"] > 0) & (matches_df["missing_count"] <= near_miss_cap)]).sort_values("missing_count"), use_container_width=True)
    with tab4:
        st.markdown("Try adding prospective buys to your inventory and preview what unlocks.")
        base_shop_df = make_shopping_list(matches_df, max_missing=near_miss_cap)
        pick = st.multiselect("Select items to hypothetically add", base_shop_df["material"].tolist() if not base_shop_df.empty else [])
        if st.button("Run what-if simulation"):
            inv_df = st.session_state.get("inventory_df", pd.DataFrame())
            current_inv_tokens = set(inv_df.loc[inv_df.get("status", "").astype(str).str.upper() != "OUT", "material"].dropna().map(normalize)) if not inv_df.empty else set()
            hypothetical_inv = current_inv_tokens.union(set(pick))
            matches_sim = compute_matches(flies_df, hypothetical_inv, aliases_map, subs_map, use_subs, ignore_labels, ignore_color, color_map, hook_map, size_tol, require_len)
            st.session_state.matches_sim = matches_sim
        if st.session_state.matches_sim is not None:
            matches_sim = st.session_state.matches_sim
            unlocked = matches_sim[(matches_sim["missing_count"] == 0) & (matches_df["missing_count"] > 0)]
            st.metric("Newly Unlocked Patterns", len(unlocked))
            st.dataframe(unlocked[["fly_name", "type", "species"]], use_container_width=True)
    with tab3:
        c4, c5 = st.columns(2)
        with c4:
            st.markdown("**Top single items to buy:**")
            singles = best_single_buys(matches_df)
            st.dataframe(enrich_buy_suggestions(singles, prefer_brands, hooks_catalog, brand_prefs, hook_map), use_container_width=True)
        with c5:
            st.markdown("**Top two-item combos:**")
            st.dataframe(best_pair_buys(matches_df), use_container_width=True)
        shopping_df = make_shopping_list(matches_df, max_missing=near_miss_cap)
        st.download_button("⬇️ Download full shopping list", enrich_buy_suggestions(shopping_df, prefer_brands, hooks_catalog, brand_prefs, hook_map).to_csv(index=False).encode("utf-8"), "shopping_list.csv", "text/csv")
    with tab5:
        st.dataframe(apply_filters(matches_df), use_container_width=True)
