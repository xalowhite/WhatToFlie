import streamlit as st
import pandas as pd
import os
import re
from collections import Counter
import itertools
import io
import zipfile
import hashlib, hmac
import json
from urllib.error import URLError
import requests
import streamlit.components.v1 as components

# =============================
# App meta (MUST BE FIRST)
# =============================
st.set_page_config(page_title="🪶 Fly Tying Recommender", page_icon="🪶", layout="wide")

APP_BUILD = "debug-v3"
st.caption(f"Build: {APP_BUILD}")

if "page_loads" not in st.session_state:
    st.session_state["page_loads"] = 0
st.session_state["page_loads"] += 1

st.write(f"DEBUG: Page load count: {st.session_state['page_loads']}")
st.write(f"DEBUG: Has token in URL: {'token' in st.query_params}")
st.write(f"DEBUG: Already authenticated: {bool(st.session_state.get('firebase_uid'))}")

@st.cache_data
def gh_url(path: str) -> str | None:
    """Build a Raw GitHub URL from .streamlit/secrets.toml [github].raw_base or env GITHUB_RAW_BASE."""
    import os
    try:
        base = None
        gh = st.secrets.get("github", {})
        if isinstance(gh, dict):
            base = gh.get("raw_base")
        if not base:
            base = os.getenv("GITHUB_RAW_BASE")
        if not base:
            return None
        return f"{str(base).rstrip('/')}/{str(path).lstrip('/')}"
    except Exception:
        return None


# =============================
# Firebase Admin Setup (Clean)
# =============================
DB = None
admin_auth = None

try:
    import firebase_admin
    from firebase_admin import credentials, firestore, auth as admin_auth

    if not firebase_admin._apps:
        service_account_info = st.secrets.get("gcp_service_account")
        if service_account_info:
            cred = credentials.Certificate(dict(service_account_info))
            firebase_admin.initialize_app(cred)
            st.success("✅ Firebase Admin initialized successfully")
        else:
            st.error("❌ No Firebase service account found in secrets")
    DB = firestore.client()
    st.success("✅ Firestore client ready")
except ImportError:
    st.error("❌ firebase-admin not installed")
except Exception as e:
    st.error(f"❌ Firebase initialization failed: {e}")

# =============================
# Firebase Web SDK Config
# =============================
FIREBASE_WEB_CONFIG = dict(st.secrets.get("firebase_web", {})) or {
    "apiKey": "AIzaSyA_dlivqpvqiYQVp0AC-yF1ZTkDSjIxVuE",
    "authDomain": "whattoflie.firebaseapp.com",
    "projectId": "whattoflie",
    # Optional keys (appId, storageBucket, etc.) may be present in secrets
}

# =============================
# Improved Google Sign-In
# =============================
def render_google_login_popup():
    """Open login in popup to avoid redirects"""
    
    login_url = "https://whattoflie.web.app/login.html"
    
    components.html(
        f"""
        <script>
        function openLoginPopup() {{
            const popup = window.open(
                '{login_url}?popup=true',
                'GoogleLogin',
                'width=500,height=600'
            );
            
            // Listen for messages from popup
            window.addEventListener('message', function(e) {{
                if (e.origin !== 'https://whattoflie.web.app') return;
                
                if (e.data.type === 'auth_success') {{
                    popup.close();
                    // Reload with token in URL
                    window.location.href = window.location.origin + 
                        window.location.pathname + 
                        '?token=' + e.data.token + 
                        '&uid=' + e.data.uid + 
                        '&email=' + e.data.email;
                }}
            }});
        }}
        </script>
        
        <button onclick="openLoginPopup()" style="
            padding: 12px 24px;
            background: #4285f4;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
        ">
            🔑 Sign in with Google (Popup)
        </button>
        """,
        height=60
    )

# =============================
# Authentication Processing (FIXED - No Redirect Loop)
# =============================
def get_firebase_user():
    """Process Firebase authentication token from URL"""
    if admin_auth is None:
        return None
    
    # Check if we already have a user in session
    if st.session_state.get("firebase_uid"):
        # Already authenticated, don't process token again
        return None
        
    token = st.query_params.get("token", "")
    if isinstance(token, list):
        token = token[0] if token else ""
    if not token:
        return None
        
    try:
        decoded_token = admin_auth.verify_id_token(token)
        st.session_state["firebase_uid"] = decoded_token.get("uid", "")
        st.session_state["firebase_email"] = decoded_token.get("email", "")
        
        # DON'T clean URL or rerun - just let it be
        # The token in URL doesn't hurt anything
        return decoded_token
    except Exception as e:
        st.error(f"Authentication verification failed: {e}")
        return None

def handle_logout():
    """Handle user logout"""
    if st.button("Sign out", key="signout_btn", type="secondary"):
        st.session_state.pop("firebase_uid", None)
        st.session_state.pop("firebase_email", None)
        st.query_params.clear()
        st.rerun()

# Process authentication
_ = get_firebase_user()

# Display authentication status
st.markdown("### 🔐 Authentication Status")
if st.session_state.get("firebase_uid"):
    user_email = st.session_state.get("firebase_email", "Unknown")
    st.success(f"✅ Signed in as: {user_email}")
    with st.expander("Debug: User Info"):
        st.json(
            {
                "uid": st.session_state.get("firebase_uid"),
                "email": st.session_state.get("firebase_email"),
                "db_available": DB is not None,
            }
        )
    handle_logout()  # This now handles the button internally
else:
    st.info("🔑 Please sign in to sync your data across devices")
    render_google_login_popup()
    
# Debug current URL params
if st.query_params:
    with st.expander("Debug: URL Parameters"):
        st.json(dict(st.query_params))

# =============================
# Legacy hashed ID helper + UID preference
# =============================
def doc_id_for(user_id: str) -> str:
    secret = st.secrets.get("pepper", "local-dev")
    return hmac.new(secret.encode(), str(user_id).strip().lower().encode(), hashlib.sha256).hexdigest()[:24]

def _effective_user_doc_id(user_id_fallback: str) -> str:
    uid = st.session_state.get("firebase_uid", "").strip()
    return uid or doc_id_for(user_id_fallback)

# =============================
# GitHub & local loaders (NO gh_url dependency)
# =============================
@st.cache_data
def read_csv_from_github(path: str, *, sep=",") -> pd.DataFrame:
    url = gh_url(path)
    if url:
        try:
            return pd.read_csv(url, encoding="utf-8", sep=sep)
        except Exception:
            pass  # fall through to local
    # local dev fallback
    return pd.read_csv(path, encoding="utf-8", sep=sep)

@st.cache_data
def fetch_bytes_from_github(path: str) -> bytes | None:
    url = gh_url(path)
    if url:
        try:
            r = requests.get(url, timeout=10)
            if r.ok:
                return r.content
        except Exception:
            pass
    # local dev fallback
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None

# =============================
# Helpers — normalization, parsing, matching
# =============================
def normalize(token: str) -> str:
    if not isinstance(token, str):
        return ""
    t = token.strip().lower()
    return " ".join(t.split())

def strip_label(token: str) -> str:
    if not isinstance(token, str):
        return ""
    t = token.strip().lower()
    if ":" in t:
        return t.split(":", 1)[1].strip()
    return t

def split_materials(materials_cell: str) -> list[str]:
    if not isinstance(materials_cell, str) or not materials_cell.strip():
        return []
    parts = [normalize(p) for p in materials_cell.split(";")]
    return [p for p in parts if p]

def coerce_flies_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize(c) for c in df.columns]
    synonyms = {
        "fly": "fly_name",
        "flyname": "fly_name",
        "name": "fly_name",
        "pattern": "fly_name",
        "tutorial": "tutorials",
        "links": "tutorials",
        "link": "tutorials",
    }
    for old, new in synonyms.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)
    missing = [c for c in ["fly_name", "type", "species", "materials"] if c not in df.columns]
    if missing:
        raise ValueError(
            "flies.csv must have headers: fly_name,type,species,materials[,tutorials]. "
            f"Missing: {missing}. Your columns: {list(df.columns)}"
        )
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
        sub_df = pd.read_csv("data/substitutions.csv", encoding="utf-8", engine="python")
        for _, row in sub_df.iterrows():
            base = normalize(row.get("material", ""))
            eq_raw = row.get("equivalents", "")
            eqs = set()
            if isinstance(eq_raw, str):
                for e in eq_raw.split(";"):
                    e = normalize(e)
                    if e:
                        eqs.add(e)
            if base:
                d[base] = eqs
        st.success(f"✅ Loaded {len(d)} substitution rules")
    except FileNotFoundError:
        st.warning("⚠️ data/substitutions.csv not found")
    except Exception as e:
        st.warning(f"⚠️ Error loading substitutions: {e}")
    return d

@st.cache_data
def load_aliases_local() -> dict[str, str]:
    d = {}
    try:
        df = pd.read_csv("data/aliases.csv", encoding="utf-8", engine="python")
        for _, row in df.iterrows():
            a = normalize(row.get("alias", ""))
            c = normalize(row.get("canonical", ""))
            if a and c:
                d[a] = c
        st.success(f"✅ Loaded {len(d)} aliases")
    except FileNotFoundError:
        st.warning("⚠️ data/aliases.csv not found")
    except Exception as e:
        st.warning(f"⚠️ Error loading aliases: {e}")
    return d

@st.cache_data
def load_subs(path: str) -> dict[str, set[str]]:
    d = {}
    try:
        sub_df = pd.read_csv(path, encoding="utf-8", engine="python")
        for _, row in sub_df.iterrows():
            base = normalize(row.get("material", ""))
            eq_raw = row.get("equivalents", "")
            eqs = set()
            if isinstance(eq_raw, str):
                for e in eq_raw.split(";"):
                    e = normalize(e)
                    if e:
                        eqs.add(e)
            if base:
                d[base] = eqs
    except Exception:
        pass
    return d

@st.cache_data
def load_aliases(path: str) -> dict[str, str]:
    d = {}
    try:
        df = pd.read_csv(path, encoding="utf-8", engine="python")
        for _, row in df.iterrows():
            a = normalize(row.get("alias", ""))
            c = normalize(row.get("canonical", ""))
            if a and c:
                d[a] = c
    except Exception:
        pass
    return d

@st.cache_data
def load_color_families(path: str) -> dict[str, str]:
    d = {}
    try:
        df = pd.read_csv(path, encoding="utf-8", engine="python")
        for _, row in df.iterrows():
            c = normalize(row.get("color", ""))
            fam = normalize(row.get("family", ""))
            if c and fam:
                d[c] = fam
    except Exception:
        pass
    return d

@st.cache_data
def load_hook_families(path: str) -> dict[str, str]:
    d = {}
    try:
        df = pd.read_csv(path, encoding="utf-8", engine="python")
        for _, row in df.iterrows():
            kw = normalize(row.get("keyword", ""))
            fam = normalize(row.get("family", ""))
            if kw and fam:
                d[kw] = fam
    except Exception:
        pass
    return d

def parse_color(token: str, color_map: dict[str, str]) -> tuple[str | None, str | None]:
    t = normalize(token)
    words = t.split()
    if not words:
        return None, None
    last = words[-1]
    fam = color_map.get(last)
    if fam:
        return last, fam
    for w in words:
        if w in color_map:
            return w, color_map[w]
    return None, None

def size_to_metric(size_str: str) -> float | None:
    if not size_str:
        return None
    s = size_str.strip().lower().lstrip("#")
    if "/0" in s:
        try:
            n = int(s.replace("/0", ""))
            return 100 + n
        except Exception:
            return None
    try:
        n = int(s)
        return 100 - n
    except Exception:
        return None

def parse_hook(token: str, hook_map: dict[str, str]) -> tuple[str | None, float | None, str | None]:
    t = normalize(token)
    if not t.startswith("hook:"):
        return None, None, None
    core = t.split(":", 1)[1].strip()
    m = re.search(r"#\s*([0-9]+(?:/0)?)", core)
    size_metric = size_to_metric(m.group(1)) if m else None
    m2 = re.search(r"\b([0-9]+xl)\b", core)
    length_tag = m2.group(1) if m2 else None
    fam = None
    for kw, f in hook_map.items():
        if kw in core:
            fam = f
            break
    if fam is None:
        if "streamer" in core:
            fam = "streamer"
        elif "nymph" in core or "midge" in core or "scud" in core:
            fam = "nymph"
        elif "dry" in core:
            fam = "dry"
        elif "jig" in core:
            fam = "jig"
        elif "wet" in core:
            fam = "wet"
    return fam, size_metric, length_tag

def apply_alias(token: str, aliases: dict[str, str]) -> str:
    t = normalize(token)
    return aliases.get(t, t)

def map_aliases_list(items: list[str], aliases: dict[str, str]) -> list[str]:
    if not aliases:
        return items
    return [apply_alias(it, aliases) for it in items]

def tokens_equal_loose(
    req: str,
    have: str,
    color_map: dict[str, str],
    ignore_labels: bool,
    ignore_color: bool,
) -> bool:
    req_token = strip_label(req) if ignore_labels else normalize(req)
    have_token = strip_label(have) if ignore_labels else normalize(have)
    if req_token == have_token:
        return True
    if ignore_color:
        rc, rf = parse_color(req_token, color_map)
        hc, hf = parse_color(have_token, color_map)
        if rf and hf and rf == hf:
            req_base = req_token.replace(f" {rc}", "") if rc else req_token
            have_base = have_token.replace(f" {hc}", "") if hc else have_token
            if req_base == have_base:
                return True
    return False

def hook_compatible(
    req: str,
    have: str,
    hook_map: dict[str, str],
    size_tolerance: int,
    require_length_match: bool,
) -> bool:
    if not isinstance(req, str) or not isinstance(have, str):
        return False
    rfam, rsize, rlen = parse_hook(req, hook_map)
    hfam, hsize, hlen = parse_hook(have, hook_map)
    if not rfam or not hfam:
        return False
    if rfam != hfam:
        return False
    if require_length_match and (rlen or hlen) and (rlen != hlen):
        return False
    if rsize is None or hsize is None:
        return True
    return abs(rsize - hsize) <= size_tolerance

def expand_with_substitutions(inv_set: set[str], subs: dict[str, set[str]]) -> set[str]:
    expanded = set(inv_set)
    for base, eqs in subs.items():
        if base in inv_set:
            expanded.update(eqs)
        if any(eq in inv_set for eq in eqs):
            expanded.add(base)
    return expanded

def compute_matches(
    flies_df: pd.DataFrame,
    inv_tokens: set[str],
    aliases_map: dict[str, str],
    subs_map: dict[str, set[str]] | None,
    use_subs: bool,
    ignore_labels: bool,
    ignore_color: bool,
    color_map: dict[str, str],
    hook_map: dict[str, str],
    size_tolerance: int,
    require_length_match: bool,
):
    inv_tokens_alias = {apply_alias(tok, aliases_map) for tok in inv_tokens}
    if use_subs and subs_map:
        inv = expand_with_substitutions(inv_tokens_alias, subs_map)
    else:
        inv = set(inv_tokens_alias)

    results = []
    for _, row in flies_df.iterrows():
        req_list = map_aliases_list(row["materials_list"], aliases_map)
        missing = []
        for mreq in req_list:
            if normalize(mreq).startswith("hook:"):
                ok = any(
                    have
                    for have in inv
                    if hook_compatible(mreq, have, hook_map, size_tolerance, require_length_match)
                )
                if not ok:
                    missing.append(mreq)
                continue
            ok = any(tokens_equal_loose(mreq, have, color_map, ignore_labels, ignore_color) for have in inv)
            if not ok:
                missing.append(mreq)
        results.append(
            {
                "fly_name": row["fly_name"],
                "type": row["type"],
                "species": "; ".join(row["species"]),
                "required_count": len(req_list),
                "missing_count": len(missing),
                "missing": "; ".join(missing),
            }
        )
    return pd.DataFrame(results)

def best_single_buys(matches_df: pd.DataFrame) -> pd.DataFrame:
    singles = matches_df[matches_df["missing_count"] == 1]["missing"]
    tokens = []
    for cell in singles:
        if isinstance(cell, str) and cell.strip():
            tokens.extend([t.strip() for t in cell.split(";") if t.strip()])
    counter = Counter(tokens)
    if not counter:
        return pd.DataFrame(columns=["material", "unlocks"])
    rows = [{"material": m, "unlocks": n} for m, n in counter.most_common()]
    return pd.DataFrame(rows)

def best_pair_buys(matches_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    pairs_counter = Counter()
    df2 = matches_df[matches_df["missing_count"] == 2]
    for _, row in df2.iterrows():
        cell = row["missing"]
        if isinstance(cell, str) and cell.strip():
            toks = [t.strip() for t in cell.split(";") if t.strip()]
            if len(toks) == 2:
                pair = tuple(sorted(toks))
                pairs_counter[pair] += 1
    rows = [{"materials_pair": " + ".join(pair), "unlocks": n} for pair, n in pairs_counter.most_common(top_n)]
    return pd.DataFrame(rows)

def make_shopping_list(matches_df: pd.DataFrame, max_missing: int = 2) -> pd.DataFrame:
    subset = matches_df[(matches_df["missing_count"] >= 1) & (matches_df["missing_count"] <= max_missing)]
    items = Counter()
    for _, row in subset.iterrows():
        cell = row["missing"]
        if isinstance(cell, str) and cell.strip():
            for t in [x.strip() for x in cell.split(";") if x.strip()]:
                items[t] += 1
    out = pd.DataFrame([{"material": m, "appears_in_near_misses": n} for m, n in items.most_common()])
    return out

@st.cache_data
def load_brand_aliases(path: str) -> dict[str, str]:
    m = {}
    try:
        df = pd.read_csv(path, encoding="utf-8", engine="python")
        for _, row in df.iterrows():
            a = normalize(row.get("alias", ""))
            b = row.get("brand", "")
            if a and b:
                m[a] = b
    except Exception:
        pass
    return m

@st.cache_data
def load_hooks_catalog(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8", engine="python")
    except Exception:
        return pd.DataFrame(columns=["brand", "model", "family", "length_tag", "wire", "eye", "barbless", "notes"])

@st.cache_data
def load_brand_prefs(path: str) -> dict[str, list[str]]:
    d = {}
    try:
        df = pd.read_csv(path, encoding="utf-8", engine="python")
        for _, row in df.iterrows():
            cat = normalize(row.get("category", ""))
            prefs = row.get("preferred_brands", "")
            arr = [p.strip() for p in str(prefs).split(";") if str(p).strip()]
            if cat:
                d[cat] = arr
    except Exception:
        pass
    return d

def canonical_brand(name: str, brand_aliases: dict[str, str]) -> str:
    n = normalize(name)
    return brand_aliases.get(n, name)

def suggest_hook_brand_model(
    req_token: str, hooks_catalog: pd.DataFrame, brand_prefs: dict[str, list[str]], hook_map: dict[str, str]
):
    rfam, rsize, rlen = parse_hook(req_token, hook_map)
    if rfam is None:
        return None, None
    df = hooks_catalog.copy()
    df = df[df["family"].fillna("").str.lower() == rfam]
    if rlen:
        df2 = df[df["length_tag"].fillna("").str.lower() == rlen.lower()]
        if not df2.empty:
            df = df2
    if df.empty:
        return None, None
    prefs = brand_prefs.get("hook", [])
    if prefs:
        for pb in prefs:
            cand = df[df["brand"].str.lower() == pb.lower()]
            if not cand.empty:
                row = cand.iloc[0]
                return row["brand"], row["model"]
    row = df.iloc[0]
    return row["brand"], row["model"]

def enrich_buy_suggestions(
    df: pd.DataFrame,
    prefer_brands: bool,
    hooks_catalog: pd.DataFrame,
    brand_prefs: dict[str, list[str]],
    hook_map: dict[str, str],
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    brands = []
    models = []
    for m in df["material"] if "material" in df.columns else []:
        mtok = normalize(m)
        if mtok.startswith("hook:") and prefer_brands:
            b, mod = suggest_hook_brand_model(mtok, hooks_catalog, brand_prefs, hook_map)
            brands.append(b or "")
            models.append(mod or "")
        else:
            brands.append("")
            models.append("")
    if "material" in df.columns:
        df = df.copy()
        df["suggested_brand"] = brands
        df["suggested_model"] = models
    return df

def validate_csv_schema(df: pd.DataFrame, required_cols: list[str], name: str) -> list[str]:
    errors = []
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        errors.append(f"{name}: missing required columns: {missing}")
    for col in required_cols:
        if col in df.columns and df[col].isnull().all():
            errors.append(f"{name}: column '{col}' is entirely empty")
    return errors

def safe_read_csv(file_or_path, required_cols: list[str] | None = None) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1"):
        for sep in (",", ";"):
            try:
                df = pd.read_csv(file_or_path, encoding=enc, sep=sep, engine="python")
                if required_cols:
                    missing = [c for c in required_cols if c not in df.columns]
                    if missing:
                        continue
                return df
            except Exception:
                continue
    raise ValueError("Could not parse CSV. Please check the headers and delimiter.")

def first_http_link(text: str) -> str:
    if not isinstance(text, str):
        return ""
    for part in text.split(";"):
        u = part.strip()
        if u.startswith("http://") or u.startswith("https://"):
            return u
    return ""

# =============================
# Brand auto-detect support (inventory editor)
# =============================
@st.cache_data
def build_known_brands_aliases() -> dict[str, str]:
    def _load_default_brand_aliases():
        m = load_brand_aliases("data/brands_aliases.csv")
        if not m:
            m = load_brand_aliases("data/brand_aliases.csv")
        return m

    alias_map = _load_default_brand_aliases()
    hooks = load_hooks_catalog("data/hooks_catalog.csv")

    known = {normalize(k): v for k, v in alias_map.items()}
    passthrough = set(
        [normalize(b) for b in (hooks["brand"].dropna().unique().tolist() if not hooks.empty else [])]
    )
    passthrough.update(["wapsi", "hareline", "uni", "veevus", "utc", "semperfli"])
    for b in passthrough:
        if b and b not in known:
            known[b] = b.title()
    return known

KNOWN_BRANDS_ALIASES = build_known_brands_aliases()

def normalize_inventory_entry(material_name: str, brand: str = "", model: str = ""):
    if not isinstance(material_name, str):
        material_name = ""
    tokens = normalize(material_name).split()
    found_brand = ""
    material_tokens = []
    for tok in tokens:
        if not found_brand and tok in KNOWN_BRANDS_ALIASES:
            found_brand = KNOWN_BRANDS_ALIASES[tok]
        else:
            material_tokens.append(tok)
    final_material = " ".join(material_tokens)
    final_brand = brand.strip().title() if isinstance(brand, str) and brand.strip() else found_brand
    final_model = model.strip() if isinstance(model, str) else ""
    if not final_material and final_brand:
        final_material = final_brand.lower()
        final_brand = ""
    return final_material, final_brand, final_model

# =============================
# UI — Hero
# =============================
st.markdown(
    """
    <div style="padding: 1.2rem; border-radius: 16px; background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); color: #e2e8f0; border: 1px solid #334155;">
      <h1 style="margin: 0 0 .4rem 0; font-size: 1.8rem;">🪶 Fly Tying Recommender</h1>
      <p style="margin: 0;">See what you can tie now, what you're 1–2 items away from, and what to buy next — with brand/model preferences for hooks.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================
# Sidebar — onboarding & data sources
# =============================
with st.sidebar:
    st.info(
        "This tool accepts CSVs with specific headers. Download templates below and follow the examples to avoid formatting issues."
    )

    with st.expander("🧭 Onboarding Checklist", expanded=True):
        if st.button(
            "🚀 Quick Start (use bundled data & run)",
            use_container_width=True,
            help="Loads the bundled CSVs and jumps straight to results",
        ):
            st.session_state["quickstart_run"] = True
            st.rerun()
        st.markdown(
            "- Download **CSV templates**\n- Fill them in (keep headers)\n- Upload here\n- Toggle matching options as desired\n- Review results + Download shopping list"
        )

    with st.expander("📄 Examples & Templates", expanded=False):
        st.code(
            """flies.csv headers:
fly_name,type,species,materials,tutorials
Elk Hair Caddis (olive),Dry,trout;panfish,hook: dry #12; thread: 8/0 brown; body: dry dubbing olive; rib: gold wire; wing: elk hair; hackle: brown rooster saddle,https://www.youtube.com/watch?v=EXAMPLE
""",
            language="text",
        )
        st.code(
            """inventory.csv headers:
material,status,brand,model
thread: 8/0 brown,HALF,UNI,
elk hair,LOW,Nature's Spirit,
""",
            language="text",
        )

        st.markdown("### 📥 Download CSV templates")
        FALLBACKS = {
            "data/templates/flies_template.csv": b"fly_name,type,species,materials,tutorials\nElk Hair Caddis (olive),Dry,trout;panfish,hook: dry #12; thread: 8/0 brown; body: dry dubbing olive; rib: gold wire; wing: elk hair; hackle: brown rooster saddle,https://www.youtube.com/watch?v=EXAMPLE\n",
            "data/templates/inventory_template.csv": b"material,status,brand,model\nthread: 8/0 brown,HALF,UNI,\nelk hair,LOW,Nature's Spirit,\n",
            "data/templates/substitutions_template.csv": b"material,equivalents\nelk hair,deer hair\ngold wire,copper wire\n",
            "data/templates/aliases_template.csv": b"alias,canonical\nolive green,olive\npt,pheasant tail\n",
        }
        files = [
            ("flies_template.csv", "data/templates/flies_template.csv", "Recipe template with headers + examples"),
            ("inventory_template.csv", "data/templates/inventory_template.csv", "Inventory template (material,status,brand,model)"),
            ("substitutions_template.csv", "data/templates/substitutions_template.csv", "Substitution rules template"),
            ("aliases_template.csv", "data/templates/aliases_template.csv", "Aliases template for normalizing inputs"),
        ]
        c1, c2, c3, c4 = st.columns(4)
        for col, (label, path, helptext) in zip([c1, c2, c3, c4], files):
            content = fetch_bytes_from_github(path) or FALLBACKS[path]
            with col:
                st.download_button(label, content, file_name=label, mime="text/csv", help=helptext)

    # ---- 1) Load Recipes
    st.header("1) Load Recipes")
    flies_file = st.file_uploader("Upload your flies.csv", type=["csv"], key="flies_csv")
    if flies_file is not None:
        try:
            raw_flies = safe_read_csv(io.BytesIO(flies_file.getvalue()), required_cols=["fly_name", "type", "species", "materials"])
            errs = validate_csv_schema(coerce_flies_columns(raw_flies.copy()), ["fly_name", "type", "species", "materials"], "flies.csv")
            for e in errs:
                st.warning(e)
            flies_df = build_flies_df(raw_flies)
        except Exception as e:
            st.error(f"flies.csv parse issue: {e}")
            st.stop()
    else:
        st.markdown("Using bundled **data/flies.csv**")
        flies_df = load_flies_local("data/flies.csv")

# =============================
# 2) Substitutions & Aliases (Local bundled)
# =============================
st.header("2) Substitutions")
subs_map = load_subs_local()

st.header("2a) Aliases")
aliases_map = load_aliases_local()

# =============================
# 3) Inventory input
# =============================
st.header("3) Inventory")
method = st.radio("How will you provide inventory?", ["Upload CSV", "Paste text", "Use bundled sample"])
inv_tokens_from_step3 = set()
inv_full_df_from_step3 = None
if method == "Upload CSV":
    inv_file = st.file_uploader(
        "Upload inventory.csv (columns: material[, status][, brand][, model])", type=["csv"], key="inv_csv"
    )
    if inv_file is not None:
        try:
            inv_df = safe_read_csv(io.BytesIO(inv_file.getvalue()), required_cols=["material"])
            errs = validate_csv_schema(inv_df, ["material"], "inventory.csv")
            for e in errs:
                st.warning(e)
        except Exception as e:
            st.error(f"inventory.csv could not be parsed: {e}")
            inv_df = None
        if inv_df is not None:
            inv_full_df_from_step3 = inv_df.copy()
            mat_col = "material" if "material" in inv_df.columns else inv_df.columns[0]
            status_col = "status" if "status" in inv_df.columns else None
            if status_col:
                inv_df[status_col] = inv_df[status_col].fillna("").str.upper()
                present_mask = inv_df[status_col] != "OUT"
                inv_present = inv_df[present_mask]
            else:
                inv_present = inv_df
            inv_tokens_from_step3 = set(inv_present[mat_col].dropna().map(normalize).tolist())
elif method == "Paste text":
    inv_text = st.text_area(
        "One material per line. Optional: append ',STATUS,BRAND,MODEL' to a line.",
        height=200,
        placeholder="elk hair\nthread: 8/0 brown,HALF,UNI,\nhook: nymph #16,NEW,Tiemco,TMC 2488",
    )
    inv_lines = [line for line in inv_text.splitlines() if line.strip()]
    if st.button("Use pasted inventory"):
        for line in inv_lines:
            material = line.split(",")[0].strip()
            if material:
                inv_tokens_from_step3.add(normalize(material))
        st.success("Inventory parsed from pasted text.")
else:
    try:
        sample_inv = read_csv_from_github("data/inventory.csv")
    except Exception:
        sample_inv = pd.read_csv("data/inventory.csv", encoding="utf-8", engine="python")
    inv_full_df_from_step3 = sample_inv.copy()
    if "status" in sample_inv.columns:
        present_mask = sample_inv["status"].astype(str).str.upper().ne("OUT")
    else:
        present_mask = pd.Series(True, index=sample_inv.index)

    inv_tokens_from_step3 = set(sample_inv[present_mask]["material"].dropna().map(normalize).tolist())
    st.markdown("Using bundled **data/inventory.csv**.")

# =============================
# 4) Matching Options
# =============================
st.header("4) Matching Options")
use_subs = st.checkbox("Use substitutions", value=st.session_state.get("pref_use_subs", True), key="pref_use_subs")
ignore_labels = st.checkbox(
    "Ignore part labels (looser matching)", value=st.session_state.get("pref_ignore_labels", True), key="pref_ignore_labels"
)
ignore_color = st.checkbox(
    "Ignore color variations (match by color family)", value=st.session_state.get("pref_ignore_color", True), key="pref_ignore_color"
)

st.markdown("---")
st.header("Hooks Matching")
size_tol = st.slider("Hook size tolerance (higher = looser)", 0, 8, st.session_state.get("pref_size_tol", 2), key="pref_size_tol")
require_len = st.checkbox("Require length tag match (e.g., 3xl)", value=st.session_state.get("pref_require_len", False), key="pref_require_len")

st.header("Brand & Model Preferences")
brand_aliases_file = st.file_uploader("Upload brands_aliases.csv (alias,brand)", type=["csv"], key="brand_aliases_csv")

def _load_brand_aliases_default():
    try:
        df = read_csv_from_github("data/brands_aliases.csv")
    except Exception:
        try:
            df = read_csv_from_github("data/brand_aliases.csv")
        except Exception:
            return {}
    m = {}
    for _, row in df.iterrows():
        a = normalize(row.get("alias", ""))
        b = row.get("brand", "")
        if a and b:
            m[a] = b
    return m

brand_aliases = load_brand_aliases(brand_aliases_file) if brand_aliases_file is not None else _load_brand_aliases_default()

hooks_catalog_file = st.file_uploader(
    "Upload hooks_catalog.csv (brand,model,family,length_tag,wire,eye,barbless,notes)", type=["csv"], key="hooks_catalog_csv"
)
hooks_catalog = load_hooks_catalog(hooks_catalog_file) if hooks_catalog_file is not None else load_hooks_catalog("data/hooks_catalog.csv")

brand_prefs_file = st.file_uploader(
    "Upload brand_prefs.csv (category,preferred_brands)", type=["csv"], key="brand_prefs_csv"
)
brand_prefs = load_brand_prefs(brand_prefs_file) if brand_prefs_file is not None else load_brand_prefs("data/brand_prefs.csv")

prefer_brands = st.checkbox(
    "Prefer my brands/models in recommendations", value=st.session_state.get("pref_prefer_brands", True), key="pref_prefer_brands"
)

# If user uploaded brand aliases, refresh the in-app auto-detection map immediately
if brand_aliases_file is not None:
    KNOWN_BRANDS_ALIASES.clear()
    KNOWN_BRANDS_ALIASES.update({normalize(k): v for k, v in brand_aliases.items()})
    passthrough = set([normalize(b) for b in (hooks_catalog["brand"].dropna().unique().tolist() if not hooks_catalog.empty else [])])
    passthrough.update(["wapsi", "hareline", "uni", "veevus", "utc", "semperfli"])
    for b in passthrough:
        if b and b not in KNOWN_BRANDS_ALIASES:
            KNOWN_BRANDS_ALIASES[b] = b.title()

# =============================
# Status panel
# =============================
st.markdown("---")
st.subheader("Status")

try:
    _flies_rows = int(flies_df.shape[0]) if "flies_df" in locals() else 0
    st.write(f"**flies_df**: {_flies_rows} rows loaded")
except Exception as e:
    st.error(f"flies_df not ready: {e}")

_subs_len = len(subs_map) if "subs_map" in locals() and subs_map else 0
_aliases_len = len(aliases_map) if "aliases_map" in locals() and aliases_map else 0
st.write(f"**substitutions**: {_subs_len} rules | **aliases**: {_aliases_len} entries")
_inv_count = len(inv_tokens_from_step3) if "inv_tokens_from_step3" in locals() else 0
st.write(f"**inventory materials present (Step 3)**: {_inv_count}")

# Load color/hook maps (bundled, prefer GitHub if configured)
try:
    df_c = read_csv_from_github("data/color_families.csv")
except Exception:
    df_c = pd.read_csv("data/color_families.csv", encoding="utf-8", engine="python")
color_map = {
    normalize(r.get("color", "")): normalize(r.get("family", ""))
    for _, r in df_c.iterrows()
    if normalize(r.get("color", "")) and normalize(r.get("family", ""))
}

try:
    df_h = read_csv_from_github("data/hooks_map.csv")
except Exception:
    df_h = pd.read_csv("data/hooks_map.csv", encoding="utf-8", engine="python")
hook_map = {
    normalize(r.get("keyword", "")): normalize(r.get("family", ""))
    for _, r in df_h.iterrows()
    if normalize(r.get("keyword", "")) and normalize(r.get("family", ""))
}

# =============================
# In-app Inventory Editor (session state)
# =============================
if "inventory_df" not in st.session_state:
    st.session_state.inventory_df = pd.DataFrame(columns=["material", "status", "brand", "model"])
if "matches_df" not in st.session_state:
    st.session_state.matches_df = None

if "matches_sim" not in st.session_state:
    st.session_state.matches_sim = None

st.header("📝 Manage Inventory (optional)")
st.info(
    """
    Use this in-app editor if you want to manage inventory interactively without juggling CSVs.
    - **Add rows** at the bottom, **edit** any cell, or **remove** rows via the editor UI.
    - The tool will auto-detect brand tokens in the **Material Name** (e.g., “Wapsi pheasant tail” → Brand=Wapsi).
    - Use the download button to save an updated `inventory.csv` for your `data/` folder.
    """
)

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

if st.button("↩️ Reset editor to bundled sample"):
    try:
        sample_inv = pd.read_csv("data/inventory.csv", encoding="utf-8", engine="python")
        expected = ["material", "status", "brand", "model"]
        for col in expected:
            if col not in sample_inv.columns:
                sample_inv[col] = ""
        st.session_state.inventory_df = (
            sample_inv[expected].fillna("").drop_duplicates().reset_index(drop=True)
        )
        st.toast("Inventory editor reset to bundled sample.", icon="♻️")
        st.rerun()
    except Exception as e:
        st.warning(f"Could not load bundled inventory: {e}")

st.markdown("---")
st.subheader("Save Your Edited Inventory")
csv_data = st.session_state.inventory_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download updated inventory.csv", csv_data, file_name="inventory.csv", mime="text/csv")

# =============================
# Tutorials & Recipe Builder
# =============================
st.header("Tutorials & Recipe Builder")
st.write("Associate tutorials with recipes, or add new recipes from a tutorial.")

with st.expander("➕ Add a tutorial link to an existing recipe"):
    sel = st.selectbox(
        "Choose a recipe",
        locals().get("flies_df", pd.DataFrame({"fly_name": []}))["fly_name"].tolist() if "flies_df" in locals() else [],
        key="add_tut_sel",
    )
    new_url = st.text_input("Tutorial URL (YouTube, blog, etc.)", placeholder="https://...", key="add_tut_url")
    add_btn = st.button("Add tutorial to recipe")
    if add_btn and new_url.strip() and "flies_df" in locals() and not locals()["flies_df"].empty:
        flies_df = locals()["flies_df"]
        idx = flies_df.index[flies_df["fly_name"] == sel][0]
        current = str(flies_df.at[idx, "tutorials"]) if "tutorials" in flies_df.columns else ""
        urls = [u.strip() for u in current.split(";") if u.strip()] if current else []
        if new_url.strip() not in urls:
            urls.append(new_url.strip())
        flies_df.at[idx, "tutorials"] = "; ".join(urls)
        st.success(f"Added tutorial to {sel}. Use the download button below to save updated flies.csv.")

with st.expander("🆕 Create a new recipe from a tutorial"):
    with st.form("new_recipe_form"):
        nr_name = st.text_input("Fly name *")
        nr_type = st.text_input("Type (e.g., Dry, Nymph, Streamer) *")
        nr_species = st.text_input("Species (semicolon-separated, e.g., trout;bass) *")
        nr_url = st.text_input("Tutorial URL (optional)")
        nr_materials = st.text_area("Materials (one per line OR semicolon-separated) *", height=150)
        nr_use_alias = st.checkbox("Apply aliases to materials on save", value=True)
        submitted = st.form_submit_button("Add recipe")
    if submitted and "flies_df" in locals():
        mats_raw = nr_materials.replace("\n", ";").replace("\r", ";")
        parts = [normalize(p) for p in mats_raw.split(";") if p.strip()]
        if nr_use_alias:
            parts = map_aliases_list(parts, locals().get("aliases_map", {}))
        materials_joined = "; ".join(parts)
        row = {
            "fly_name": nr_name.strip(),
            "type": nr_type.strip(),
            "species": ";".join([normalize(s) for s in nr_species.split(";") if s.strip()]),
            "materials": materials_joined,
            "tutorials": nr_url.strip() if nr_url.strip() else "",
        }
        flies_df = locals()["flies_df"]
        flies_df.loc[len(flies_df)] = row
        st.success(
            f"Added new recipe '{nr_name}'. It will be included immediately in matching. Use the download button below to save updated flies.csv."
        )

# =============================
# Cloud sync: Account + prefs (only if DB configured)
# =============================
def _current_doc_id():
    uid = st.session_state.get("firebase_uid", "").strip()
    if uid:
        return uid
    return st.session_state.get("account_id", "").strip().lower()

qp = st.query_params
if "acct" in qp and not st.session_state.get("account_id"):
    acct_from_url = qp.get("acct")
    if isinstance(acct_from_url, list):
        acct_from_url = acct_from_url[0] if acct_from_url else ""
    st.session_state["account_id"] = acct_from_url or ""
    if st.session_state["account_id"]:
        st.toast(f"Loaded account from URL: {st.session_state['account_id']}", icon="✅")

def set_account_id():
    acct = st.session_state.get("account_id", "").strip()
    if acct:
        st.query_params["acct"] = acct
    else:
        if "acct" in st.query_params:
            del st.query_params["acct"]


st.markdown("### ☁️ Cloud preferences")
acct_id_prefs = st.text_input(
    "Account ID for preferences (use the same as Cloud sync above)",
    key="account_id_prefs",
    value=st.session_state.get("account_id", ""),
    placeholder="you@example.com",
)

def save_user_inventory(inv_df: pd.DataFrame) -> bool:
    if DB is None or not st.session_state.get("firebase_uid"):
        st.error("You must be logged in to save.")
        return False
    try:
        doc_id = st.session_state["firebase_uid"]
        DB.collection("users").document(doc_id).collection("app").document("inventory").set(
            {"rows": inv_df.to_dict(orient="records")}, merge=True
        )
        return True
    except Exception as e:
        st.error(f"Failed to save inventory: {e}")
        return False

def load_user_inventory() -> pd.DataFrame | None:
    if DB is None or not st.session_state.get("firebase_uid"):
        return None
    try:
        doc_id = st.session_state["firebase_uid"]
        doc = DB.collection("users").document(doc_id).collection("app").document("inventory").get()
        return pd.DataFrame(doc.to_dict().get("rows", [])) if doc.exists else None
    except Exception as e:
        st.error(f"Failed to load inventory: {e}")
        return None


def save_user_prefs(user_id: str, prefs: dict) -> bool:
    if DB is None:
        return False
    try:
        doc_id = _effective_user_doc_id(user_id)
        DB.collection("users").document(doc_id).collection("app").document("preferences").set(prefs, merge=True)
        return True
    except Exception as e:
        st.error(f"Failed to save preferences: {e}")
        return False

def load_user_prefs(user_id: str) -> dict | None:
    if DB is None:
        return None
    try:
        doc_id = _effective_user_doc_id(user_id)
        doc = DB.collection("users").document(doc_id).collection("app").document("preferences").get()
        return doc.to_dict() if doc.exists else None
    except Exception as e:
        st.error(f"Failed to load preferences: {e}")
        return None

cP1, cP2 = st.columns(2)
with cP1:
    if st.button("Save Preferences to Cloud"):
        prefs = {
            "pref_use_subs": bool(st.session_state.get("pref_use_subs", True)),
            "pref_ignore_labels": bool(st.session_state.get("pref_ignore_labels", True)),
            "pref_ignore_color": bool(st.session_state.get("pref_ignore_color", True)),
            "pref_size_tol": int(st.session_state.get("pref_size_tol", 2)),
            "pref_require_len": bool(st.session_state.get("pref_require_len", False)),
            "pref_prefer_brands": bool(st.session_state.get("pref_prefer_brands", True)),
            "aliases_map": locals().get("aliases_map", {}),
            "subs_map": {k: sorted(list(v)) for k, v in (locals().get("subs_map", {}) or {}).items()},
            "brand_prefs": locals().get("brand_prefs", {}),
        }
        if save_user_prefs(acct_id_prefs, prefs):
            st.success("Preferences saved to cloud.")
with cP2:
    if st.button("Load Preferences from Cloud"):
        prefs = load_user_prefs(acct_id_prefs)
        if not prefs:
            st.warning("No preferences found for this account.")
        else:
            for k in [
                "pref_use_subs",
                "pref_ignore_labels",
                "pref_ignore_color",
                "pref_size_tol",
                "pref_require_len",
                "pref_prefer_brands",
            ]:
                if k in prefs:
                    st.session_state[k] = prefs[k]
            if "aliases_map" in prefs and isinstance(prefs["aliases_map"], dict):
                aliases_map.update({str(k): str(v) for k, v in prefs["aliases_map"].items()})
            if "subs_map" in prefs and isinstance(prefs["subs_map"], dict):
                subs_map.clear()
                for b, eqs in prefs["subs_map"].items():
                    subs_map[str(b)] = set(str(e) for e in (eqs or []))
            if "brand_prefs" in prefs and isinstance(prefs["brand_prefs"], dict):
                brand_prefs.update(prefs["brand_prefs"])
            st.success("Preferences loaded from cloud.")
            st.rerun()

with st.expander("📦 Export / Import preferences (JSON)"):
    cur_prefs = {
        "pref_use_subs": bool(st.session_state.get("pref_use_subs", True)),
        "pref_ignore_labels": bool(st.session_state.get("pref_ignore_labels", True)),
        "pref_ignore_color": bool(st.session_state.get("pref_ignore_color", True)),
        "pref_size_tol": int(st.session_state.get("pref_size_tol", 2)),
        "pref_require_len": bool(st.session_state.get("pref_require_len", False)),
        "pref_prefer_brands": bool(st.session_state.get("pref_prefer_brands", True)),
        "aliases_map": locals().get("aliases_map", {}),
        "subs_map": {k: sorted(list(v)) for k, v in (locals().get("subs_map", {}) or {}).items()},
        "brand_prefs": locals().get("brand_prefs", {}),
    }
    st.download_button(
        "⬇️ Export preferences.json",
        data=json.dumps(cur_prefs, indent=2).encode("utf-8"),
        file_name="preferences.json",
        mime="application/json",
        help="Download your current preferences",
    )

    prefs_json_upload = st.file_uploader("Import preferences.json", type=["json"], key="prefs_json_upload")
    if prefs_json_upload is not None:
        try:
            imported = json.loads(prefs_json_upload.getvalue().decode("utf-8"))
            for k in [
                "pref_use_subs",
                "pref_ignore_labels",
                "pref_ignore_color",
                "pref_size_tol",
                "pref_require_len",
                "pref_prefer_brands",
            ]:
                if k in imported:
                    st.session_state[k] = imported[k]
            if "aliases_map" in imported and isinstance(imported["aliases_map"], dict):
                aliases_map.update({str(k): str(v) for k, v in imported["aliases_map"].items()})
            if "subs_map" in imported and isinstance(imported["subs_map"], dict):
                subs_map.clear()
                for b, eqs in imported["subs_map"].items():
                    subs_map[str(b)] = set(str(e) for e in (eqs or []))
            if "brand_prefs" in imported and isinstance(imported["brand_prefs"], dict):
                brand_prefs.update(imported["brand_prefs"])
            st.success("Preferences imported from JSON.")
            st.rerun()
        except Exception as e:
            st.error(f"Could not import preferences JSON: {e}")

cA, cB = st.columns(2)
with cA:
    if st.button("Save to Cloud (inventory editor)"):
        if st.session_state.inventory_df.empty:
            st.warning("Inventory editor is empty. Add rows first.")
        elif save_user_inventory(st.session_state.inventory_df):
            st.success("Saved your inventory to the cloud.")
with cB:
    if st.button("Load from Cloud (replace editor)"):
        df_cloud = load_user_inventory()
        if df_cloud is None:
            st.warning("No cloud inventory found for this account.")
        else:
            expected = ["material", "status", "brand", "model"]
            for col in expected:
                if col not in df_cloud.columns:
                    df_cloud[col] = ""
            df_cloud = df_cloud[expected].fillna("")
            st.session_state.inventory_df = df_cloud.drop_duplicates().reset_index(drop=True)
            st.success("Loaded inventory from the cloud.")
            st.rerun()

# --- Auto-load inventory & prefs once per session (after inputs are visible)
if DB is not None and st.session_state.get("firebase_uid") and not st.session_state.get("did_auto_load"):
    try:
        df_cloud = load_user_inventory()
        if isinstance(df_cloud, pd.DataFrame) and not df_cloud.empty:
            expected = ["material", "status", "brand", "model"]
            for col in expected:
                if col not in df_cloud.columns:
                    df_cloud[col] = ""
            st.session_state.inventory_df = (
                df_cloud[expected].fillna("").drop_duplicates().reset_index(drop=True)
            )

        prefs = load_user_prefs(st.session_state["firebase_uid"])
        if prefs:
            for k in [
                "pref_use_subs",
                "pref_ignore_labels",
                "pref_ignore_color",
                "pref_size_tol",
                "pref_require_len",
                "pref_prefer_brands",
            ]:
                if k in prefs:
                    st.session_state[k] = prefs[k]
            if "aliases_map" in prefs and isinstance(prefs["aliases_map"], dict):
                aliases_map.update({str(k): str(v) for k, v in prefs["aliases_map"].items()})
            if "subs_map" in prefs and isinstance(prefs["subs_map"], dict):
                subs_map.clear()
                for b, eqs in prefs["subs_map"].items():
                    subs_map[str(b)] = set(str(e) for e in (eqs or []))
            if "brand_prefs" in prefs and isinstance(prefs["brand_prefs"], dict):
                brand_prefs.update(prefs["brand_prefs"])

        st.session_state["did_auto_load"] = True
        st.rerun()
    except Exception as e:
        st.warning(f"Auto-load failed: {e}")

# =============================
# Inventory Source for Matching (RESTORED)
# =============================
st.header("Inventory Source for Matching")
inv_source = st.radio(
    "Choose which inventory to use when matching:",
    ["Step 3 upload/paste/sample", "Inventory Editor (session)", "Merge both"],
    index=0,
    help="The editor lets you manage inventory inline. Merge = union of both sources (excluding items with status OUT).",
)

# =============================
# Run matching
# =============================
run_now = st.button("▶️ Run matching")
if st.session_state.get("quickstart_run"):
    run_now = True
    st.session_state["quickstart_run"] = False

if run_now:
    if "flies_df" not in locals() or locals()["flies_df"].empty:
        st.error("No recipes loaded. Please upload a valid flies.csv or use the bundled one from data/flies.csv.")
        st.stop()

    flies_df = locals()["flies_df"]
    aliases_map = locals().get("aliases_map", {})
    subs_map = locals().get("subs_map", {})
    prefer_brands = locals().get("prefer_brands", True)
    size_tol = locals().get("size_tol", 2)
    require_len = locals().get("require_len", False)

    # Build inventory tokens
    inv_tokens_step3 = locals().get("inv_tokens_from_step3", set())
    inv_editor_df = st.session_state.get("inventory_df", pd.DataFrame(columns=["material", "status", "brand", "model"]))
    if not inv_editor_df.empty:
        present_mask_editor = inv_editor_df.get("status", "").astype(str).str.upper().ne("OUT")
        inv_tokens_editor = set(inv_editor_df.loc[present_mask_editor, "material"].dropna().map(normalize).tolist())
    else:
        inv_tokens_editor = set()

    if inv_source == "Step 3 upload/paste/sample":
        inv_tokens = inv_tokens_step3
    elif inv_source == "Inventory Editor (session)":
        inv_tokens = inv_tokens_editor
    else:
        inv_tokens = inv_tokens_step3.union(inv_tokens_editor)

    # Compute matches
    matches_df = compute_matches(
        flies_df=flies_df,
        inv_tokens=inv_tokens,
        aliases_map=aliases_map,
        subs_map=subs_map,
        use_subs=locals().get("use_subs", True),
        ignore_labels=locals().get("ignore_labels", True),
        ignore_color=locals().get("ignore_color", True),
        color_map=color_map,
        hook_map=hook_map,
        size_tolerance=size_tol,
        require_length_match=require_len,
    )

    # Attach first tutorial link
    matches_df = matches_df.merge(flies_df[["fly_name", "tutorials"]], on="fly_name", how="left")
    matches_df["tutorial"] = matches_df["tutorials"].apply(first_http_link)
    matches_df.drop(columns=["tutorials"], inplace=True)

    # Store results in session state
    st.session_state.matches_df = matches_df
    # Clear any old simulation results
    st.session_state.matches_sim = None

def add_items_to_editor(items: list[str]) -> int:
    """Add normalized 'NEW' items to the in-app inventory editor, skipping duplicates."""
    if not items:
        return 0
    base = st.session_state.get(
        "inventory_df", pd.DataFrame(columns=["material", "status", "brand", "model"])
    ).copy()

    new_rows = []
    for raw in items:
        tok = normalize(raw)
        if not tok:
            continue
        # Skip if already present
        if not base.empty and (base["material"].astype(str).str.lower() == tok).any():
            continue
        mat, brand, model = normalize_inventory_entry(tok, "", "")
        new_rows.append({"material": mat, "status": "NEW", "brand": brand, "model": model})

    if new_rows:
        st.session_state.inventory_df = (
            pd.concat([base, pd.DataFrame(new_rows)], ignore_index=True)
            .drop_duplicates()
            .reset_index(drop=True)
        )
        return len(new_rows)
    return 0

if st.button("🧹 Clear results / start fresh"):
    st.session_state.matches_df = None
    st.session_state.matches_sim = None
    st.rerun()
    
# =============================
# Results & tools (Display Logic)
# =============================
if st.session_state.matches_df is not None:
    matches_df = st.session_state.matches_df  # Use the stored results

    st.subheader("Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("✅ Can tie now", int((matches_df["missing_count"] == 0).sum()))
    c2.metric("🟡 Missing 1", int((matches_df["missing_count"] == 1).sum()))
    c3.metric("🟠 Missing 2", int((matches_df["missing_count"] == 2).sum()))

    types = sorted(flies_df["type"].dropna().unique().tolist())
    species_all = sorted(set(itertools.chain.from_iterable(flies_df["species"].tolist())))
    col1, col2, col3 = st.columns(3)
    with col1:
        type_filter = st.multiselect("Filter by type", types, default=types)
    with col2:
        species_filter = st.multiselect("Filter by species", species_all, default=species_all)
    with col3:
        show_cols = st.multiselect(
            "Columns to show",
            ["fly_name", "type", "species", "required_count", "missing_count", "missing", "tutorial"],
            default=["fly_name", "type", "species", "required_count", "missing_count", "missing", "tutorial"],
        )

    fly_query = st.text_input("🔎 Search by fly name", "", placeholder="e.g. elk hair, pheasant tail, zonker...")
    near_miss_cap = st.slider(
        "Near-miss threshold (≤ this many missing)", 2, 6, 3,
        help="Show and aggregate flies that are within this many missing materials."
    )

    def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
        mask_type = df["type"].isin(type_filter)
        mask_species = df["species"].apply(lambda s: any(sp in s for sp in species_filter))
        mask_query = df["fly_name"].str.contains(fly_query.strip(), case=False, na=False) if fly_query.strip() else True
        out = df[mask_type & mask_species & mask_query].copy()
        if show_cols:
            keep = [c for c in show_cols if c in out.columns]
            out = out[keep]
        return out

    # Tabs
    tab1, tab2, tab3, tabN, tab4, tab5, tabW = st.tabs(
        ["✅ Tie now", "🟡 Missing 1", "🟠 Missing 2", f"≤{near_miss_cap} Missing", "🛒 Buy suggestions", "📦 Inventory", "🔮 What-if"]
    )

    # --- Tab: Can tie now
    with tab1:
        df = apply_filters(matches_df[matches_df["missing_count"] == 0])
        st.dataframe(df, use_container_width=True)
        if df.empty:
            st.info("No results here with the current filters. Try widening Type/Species or raising the near-miss threshold.")

    # --- Tab: Missing 1
    with tab2:
        df = apply_filters(matches_df[matches_df["missing_count"] == 1])
        st.dataframe(df, use_container_width=True)
        if df.empty:
            st.info("No results here with the current filters. Try widening Type/Species or raising the near-miss threshold.")

        miss_pool_1 = []
        for cell in df["missing"].dropna():
            miss_pool_1.extend([x.strip() for x in str(cell).split(";") if x.strip()])
        miss_opts_1 = sorted(set(miss_pool_1))
        sel_add_1 = st.multiselect("➕ Add these missing items to the Inventory Editor", miss_opts_1, key="add_miss1")

        if st.button("Add selected to Editor", key="btn_add_miss1"):
            added = add_items_to_editor(sel_add_1)
            st.success(f"Added {added} item(s) to the editor.")

    # --- Tab: Missing 2
    with tab3:
        df = apply_filters(matches_df[matches_df["missing_count"] == 2])
        st.dataframe(df, use_container_width=True)
        if df.empty:
            st.info("No results here with the current filters. Try widening Type/Species or raising the near-miss threshold.")

        miss_pool_2 = []
        for cell in df["missing"].dropna():
            miss_pool_2.extend([x.strip() for x in str(cell).split(";") if x.strip()])
        miss_opts_2 = sorted(set(miss_pool_2))
        sel_add_2 = st.multiselect("➕ Add these missing items to the Inventory Editor", miss_opts_2, key="add_miss2")

        if st.button("Add selected to Editor", key="btn_add_miss2"):
            added = add_items_to_editor(sel_add_2)
            st.success(f"Added {added} item(s) to the editor.")

    # --- Tab: ≤ N missing (Near misses)
    with tabN:
        df = apply_filters(matches_df[matches_df["missing_count"] <= near_miss_cap])
        st.dataframe(df.sort_values(["missing_count", "required_count", "fly_name"]), use_container_width=True)
        if df.empty:
            st.info("No results here with the current filters. Try widening Type/Species or raising the near-miss threshold.")

    # --- Tab: Buy suggestions
    with tab4:
        singles = best_single_buys(matches_df)
        try:
            hooks_catalog = read_csv_from_github("data/hooks_catalog.csv")
        except Exception:
            hooks_catalog = pd.read_csv("data/hooks_catalog.csv", encoding="utf-8", engine="python")
        try:
            df_bp = read_csv_from_github("data/brand_prefs.csv")
            brand_prefs = {}
            for _, row in df_bp.iterrows():
                cat = normalize(row.get("category", ""))
                prefs = row.get("preferred_brands", "")
                arr = [p.strip() for p in str(prefs).split(";") if str(p).strip()]
                if cat:
                    brand_prefs[cat] = arr
        except Exception:
            brand_prefs = {}

        singles = enrich_buy_suggestions(singles, prefer_brands, hooks_catalog, brand_prefs, hook_map)
        pairs = best_pair_buys(matches_df)

        c4, c5 = st.columns(2)
        with c4:
            st.markdown("**Top single items to buy (unlocks immediately):**")
            st.dataframe(singles, use_container_width=True)
            if singles.empty:
                st.info("No single-item unlocks with current filters.")
            st.download_button(
                "⬇️ Download top singles (enriched)",
                singles.to_csv(index=False).encode("utf-8"),
                file_name="best_single_buys_enriched.csv",
                mime="text/csv",
            )
        with c5:
            st.markdown("**Top two-item combos (buy both to unlock):**")
            st.dataframe(pairs, use_container_width=True)
            if pairs.empty:
                st.info("No two-item combos with current filters.")
            st.download_button(
                "⬇️ Download top pairs",
                pairs.to_csv(index=False).encode("utf-8"),
                file_name="best_pair_buys.csv",
                mime="text/csv",
            )

        shopping_df = make_shopping_list(matches_df, max_missing=near_miss_cap)
        shopping_df = enrich_buy_suggestions(shopping_df, prefer_brands, hooks_catalog, brand_prefs, hook_map)
        st.download_button(
            "⬇️ Download shopping list (enriched CSV)",
            shopping_df.to_csv(index=False).encode("utf-8"),
            file_name="shopping_list_enriched.csv",
            mime="text/csv",
        )

        if not shopping_df.empty:
            lines = []
            for _, r in shopping_df.iterrows():
                line = f"- {r.get('material','')}"
                if prefer_brands and str(r.get("suggested_brand", "")).strip():
                    bm = f"{r.get('suggested_brand','')} {r.get('suggested_model','')}".strip()
                    if bm:
                        line += f" — {bm}"
                lines.append(line)
            shopping_text = "\n".join(lines)
            st.markdown("**Shopping list (plain text):**")
            st.text_area("Copyable shopping list", shopping_text, height=150, label_visibility="collapsed")
            escaped = json.dumps(shopping_text)
            st.markdown(
                f"""
                <button onClick='navigator.clipboard.writeText({escaped});' style="margin-top:6px;padding:6px 10px;border-radius:8px;border:1px solid #ccc;cursor:pointer">
                    📋 Copy shopping list
                </button>
                """,
                unsafe_allow_html=True,
            )

    # --- Tab: Inventory view & QA
    with tab5:
        st.markdown("**Inventory from Step 3 (upload/paste/sample):**")
        inv_full_df_from_step3 = locals().get("inv_full_df_from_step3", None)

        def status_badge(s: str) -> str:
            m = str(s).upper()
            return {"NEW": "🆕 New", "HALF": "🌓 Half", "LOW": "🪫 Low", "OUT": "⛔️ Out", "": ""}.get(m, m)

        if inv_full_df_from_step3 is not None:
            disp = inv_full_df_from_step3.copy()
            if "status" in disp.columns:
                disp["status_badge"] = disp["status"].apply(status_badge)
                cols = ["status_badge"] + [c for c in ["material", "brand", "model", "status"] if c in disp.columns]
                st.dataframe(disp[cols], use_container_width=True)
                counts = disp["status"].fillna("").str.upper().value_counts().reset_index()
                counts.columns = ["status", "count"]
                st.markdown("**Counts by status:**")
                st.dataframe(counts, use_container_width=True)
            else:
                st.dataframe(disp, use_container_width=True)
        else:
            st.info("Upload an inventory CSV to see statuses and brands here.")

        with st.expander("⚠️ Inventory QA (possible typos / unknown materials)"):
            inv_tokens_step3 = locals().get("inv_tokens_from_step3", set())
            inv_tokens_editor = (
                set(st.session_state.inventory_df["material"].dropna().map(normalize).tolist())
                if not st.session_state.inventory_df.empty
                else set()
            )
            inv_all_tokens = inv_tokens_step3.union(inv_tokens_editor)

            def derive_known_material_tokens(
                flies_df: pd.DataFrame, aliases_map: dict[str, str], subs_map: dict[str, set[str]] | None
            ) -> set[str]:
                known = set()
                for lst in flies_df["materials_list"]:
                    known.update(map_aliases_list(lst, aliases_map))
                known.update([normalize(v) for v in aliases_map.values()])
                if subs_map:
                    for b, eqs in subs_map.items():
                        known.add(normalize(b))
                        known.update([normalize(e) for e in eqs])
                return {normalize(strip_label(k)) for k in known if k}

            known_tokens = derive_known_material_tokens(flies_df, aliases_map, subs_map)

            suspicious = []
            for t in sorted(inv_all_tokens):
                if t.startswith("hook:"):
                    continue
                cand = normalize(strip_label(apply_alias(t, aliases_map)))
                if cand not in known_tokens:
                    suspicious.append({"material_in_inventory": t, "normalized": cand})

            if suspicious:
                st.dataframe(pd.DataFrame(suspicious), use_container_width=True)
            else:
                st.success("No obvious anomalies found in your inventory 🎉")

    # --- Tab: What-if simulator
    with tabW:
        st.markdown("Try adding prospective buys to your inventory and preview what unlocks.")
        base_shop_df = make_shopping_list(matches_df, max_missing=near_miss_cap)
        base_items = base_shop_df["material"].tolist() if not base_shop_df.empty else []
        pick = st.multiselect(
            "Select items to hypothetically add",
            base_items,
            help="Start with the shopping list; you can also type free-form entries below.",
        )
        extra_freeform = st.text_area(
            "Optional free-form additions (one per line)", height=100,
            placeholder="hook: nymph #16\ndry dubbing olive\nkrystal flash pearl"
        )
        extra_tokens = [normalize(x) for x in extra_freeform.splitlines() if x.strip()]

        # Rebuild current inv_tokens if needed (e.g., page reloaded with stored matches)
        if "inv_tokens" in locals():
            current_inv_tokens = inv_tokens
        else:
            inv_tokens_step3 = locals().get("inv_tokens_from_step3", set())
            inv_editor_df = st.session_state.get(
                "inventory_df", pd.DataFrame(columns=["material", "status", "brand", "model"])
            )
            if not inv_editor_df.empty:
                present_mask_editor = inv_editor_df.get("status", "").astype(str).str.upper().ne("OUT")
                inv_tokens_editor = set(inv_editor_df.loc[present_mask_editor, "material"].dropna().map(normalize).tolist())
            else:
                inv_tokens_editor = set()
            if inv_source == "Step 3 upload/paste/sample":
                current_inv_tokens = inv_tokens_step3
            elif inv_source == "Inventory Editor (session)":
                current_inv_tokens = inv_tokens_editor
            else:
                current_inv_tokens = inv_tokens_step3.union(inv_tokens_editor)

        simulate_btn = st.button("Run what-if simulation")

        if simulate_btn:
            hypothetical_inv = set(current_inv_tokens).union(set(pick)).union(set(extra_tokens))
            matches_sim = compute_matches(
                flies_df=flies_df,
                inv_tokens=hypothetical_inv,
                aliases_map=aliases_map,
                subs_map=subs_map,
                use_subs=locals().get("use_subs", True),
                ignore_labels=locals().get("ignore_labels", True),
                ignore_color=locals().get("ignore_color", True),
                color_map=color_map,
                hook_map=hook_map,
                size_tolerance=size_tol,
                require_length_match=require_len,
            )
            st.session_state.matches_sim = matches_sim  # Store simulation results

        if st.session_state.matches_sim is not None:
            matches_sim = st.session_state.matches_sim  # Use stored simulation results
            cA, cB, cC = st.columns(3)
            cA.metric("✅ Can tie now (what-if)", int((matches_sim["missing_count"] == 0).sum()))
            cB.metric("🟡 Missing 1 (what-if)", int((matches_sim["missing_count"] == 1).sum()))
            cC.metric("🟠 Missing 2 (what-if)", int((matches_sim["missing_count"] == 2).sum()))
            unlocked = matches_sim[(matches_sim["missing_count"] == 0) & (matches_df["missing_count"] > 0)]
            st.markdown("**Newly unlocked patterns (vs. current):**")
            st.dataframe(
                unlocked[["fly_name", "type", "species"]].sort_values(["type", "fly_name"]),
                use_container_width=True
            )
            st.download_button(
                "⬇️ Download what-if unlocked list (CSV)",
                unlocked[["fly_name", "type", "species"]].to_csv(index=False).encode("utf-8"),
                file_name="what_if_unlocked.csv",
                mime="text/csv",
            )

    # =============================
    # Export bundle
    # =============================
    st.markdown("---")
    st.subheader("⬇️ Download everything (ZIP bundle)")

    if st.button("Build ZIP bundle"):
        hooks_catalog = load_hooks_catalog("data/hooks_catalog.csv")
        brand_prefs = load_brand_prefs("data/brand_prefs.csv")
        singles_all = enrich_buy_suggestions(best_single_buys(matches_df), prefer_brands, hooks_catalog, brand_prefs, hook_map)
        pairs_all = best_pair_buys(matches_df)
        shopping_all = enrich_buy_suggestions(
            make_shopping_list(matches_df, max_missing=near_miss_cap), prefer_brands, hooks_catalog, brand_prefs, hook_map
        )

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("matches.csv", matches_df.to_csv(index=False))
            z.writestr("best_single_buys_enriched.csv", singles_all.to_csv(index=False))
            z.writestr("best_pair_buys.csv", pairs_all.to_csv(index=False))
            z.writestr("shopping_list_enriched.csv", shopping_all.to_csv(index=False))
            inv_full_df_from_step3 = locals().get("inv_full_df_from_step3", None)
            if inv_full_df_from_step3 is not None:
                z.writestr("inventory_step3.csv", inv_full_df_from_step3.to_csv(index=False))
            if not st.session_state.inventory_df.empty:
                z.writestr("inventory_editor.csv", st.session_state.inventory_df.to_csv(index=False))
            z.writestr("flies.csv", flies_df.to_csv(index=False))
            opts = {
                "use_subs": bool(locals().get("use_subs", True)),
                "ignore_labels": bool(locals().get("ignore_labels", True)),
                "ignore_color": bool(locals().get("ignore_color", True)),
                "size_tol": int(locals().get("size_tol", 2)),
                "require_len": bool(locals().get("require_len", False)),
                "near_miss_cap": int(near_miss_cap),
                "type_filter": types,
                "species_filter": species_all,
            }
            z.writestr("options.json", pd.Series(opts, dtype=object).to_json())

        st.download_button(
            "⬇️ Download ZIP bundle", data=buf.getvalue(),
            file_name="fly_tying_recommender_bundle.zip", mime="application/zip"
        )

    def log_event(name: str, payload: dict):
        if DB is None:
            return
        try:
            ev = {"name": name, "ts": pd.Timestamp.utcnow().isoformat(), **payload}
            DB.collection("events").add(ev)
        except Exception as e:
            st.warning(f"Analytics not saved: {e}")

    # Note: this runs only in the same execution path as matching; keep it here.
    if "inv_tokens" in locals():
        log_event(
            "run_matching",
            {
                "acct": st.session_state.get("account_id", ""),
                "inventory_source": inv_source,
                "inv_count": len(inv_tokens),
                "can_tie": int((matches_df["missing_count"] == 0).sum()),
                "miss1": int((matches_df["missing_count"] == 1).sum()),
                "miss2": int((matches_df["missing_count"] == 2).sum()),
                "size_tol": size_tol,
                "ignore_labels": ignore_labels,
                "ignore_color": ignore_color,
                "prefer_brands": prefer_brands,
                "src": st.query_params.get("src", ""),
            },
        )
else:
    st.info("Click **Run matching** to see your results.")

