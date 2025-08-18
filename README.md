# Fly Tying Recommender — Guide Box Edition

**What it does:** Upload your inventory and see:
- Which flies you can tie now
- Which patterns are 1–2 items away
- The best single/pair purchases to unlock the most flies
- Optional brand/model prefs for hooks
- Add your own recipes and tutorial links in-app

## Run
```bash
pip install -r requirements.txt
streamlit run app/app.py
```
Use the **bundled data** first. Click **▶️ Run matching**. Then upload your own.

## CSVs & Schemas
- `data/flies.csv`: `fly_name,type,species,materials,tutorials`
- `data/inventory.csv`: `material,status,brand,model` (status: NEW/HALF/LOW/OUT)
- `data/substitutions.csv`: `material,equivalents` (semicolon-separated equivalents)
- `data/aliases.csv`: `alias,canonical`
- `data/color_families.csv`: `color,family`
- `data/hooks_map.csv`: `keyword,family`
- `data/brand_prefs.csv`: `category,preferred_brands`
- `data/brands_aliases.csv`: `alias,brand`
- `data/hooks_catalog.csv`: `brand,model,family,length_tag,wire,eye,barbless,notes`

Templates are in `data/templates/` and downloadable from the app sidebar.
