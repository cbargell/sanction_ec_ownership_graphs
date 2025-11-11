import os
import pandas as pd

gcap_data = '/oak/stanford/groups/maggiori/GCAP/data'

root = f"{gcap_data}/raw/orbis/latest/firm_description/dta/appended_countries"

# countries = list of first-level dirs
countries = sorted(
    e.name for e in os.scandir(root)
    if e.is_dir() and not e.name.startswith(".")
)


# Keep a single primary industry code for each entity
for country in countries:
    print({country})
    in_path = f"{root}/{country}/{country}_industry_classifications_long.dta"
    if not os.path.isfile(in_path):
        print(f"[skip] {country}: file not found")
        continue

    try:
        df = pd.read_stata(in_path, convert_categoricals=False)
        before = len(df)

        df = df.drop_duplicates(subset=["bvdid"], keep="first")
        after = len(df)

        out_path = f"{gcap_data}/scratch/chiara/cs230/orbis/temp/industry_classifications/{country}_industry_classifications_long_first.dta"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        df.to_stata(out_path, write_index=False, version=118)
        print(f"[ok] {country}: {before} -> {after} saved → {out_path}")
    except Exception as e:
        print(f"[error] {country}: {e}")