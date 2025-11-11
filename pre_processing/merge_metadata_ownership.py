import os
import pandas as pd
from pathlib import Path
import pyreadstat

gcap_data = '/oak/stanford/groups/maggiori/GCAP/data'


#### FUNCTIONS

def merge_code_vars(df, df_code, var_list, using_key):
    for var in var_list:
        prefix = var.split("_")[0] + "_"  # sub_ or share_

        # Prepare df_code copy
        tmp = df_code.rename(columns={using_key: var}).copy()

        # Convert both sides to string
        df[var] = df[var].astype("string")
        tmp[var] = tmp[var].astype("string")
        
        # Replace NA with empty string "" (string missing)
        df[var] = df[var].fillna("").replace(".", "")

        # Strip whitespace
        df[var] = df[var].str.strip()
        tmp[var] = tmp[var].str.strip()

        # Prefix all other columns
        cols_to_prefix = {c: prefix + c for c in tmp.columns if c != var}
        tmp = tmp.rename(columns=cols_to_prefix)

        # Merge
        df = df.merge(tmp, on=var, how="left")

    return df


######## SHAREHOLDER DATA MERGE

# Inputs
in_dir  = Path(f"{gcap_data}/scratch/chiara/cs230/orbis/temp/ownership/ownership_chunks")
out_dir = Path(f"{gcap_data}/scratch/chiara/cs230/orbis/temp/ownership/ownership_industry_code")
out_dir.mkdir(parents=True, exist_ok=True)

# parameters
country = "CN"
year = 2022
total_parts = 20
parts = ["01","02","03", "04","05", "06","07","08", "09","10", "11","12","13","14","15","16","17","18","19", "20"]

for part in parts:
    fname = f"SHARE_{country}_Links_{year}_part{part}of{total_parts}.dta"
    in_path = in_dir / fname
    if not in_path.exists():
        print(f"Missing {fname} — skipping")
        continue
    df, meta = pyreadstat.read_dta(in_path)  # load whole file (no chunking)

    # ---- YOUR PROCESSING HERE ----
    # e.g., inspect shape/variables
    print(f"  Rows: {len(df):,} | Vars: {len(df.columns)}")
    
    # Drop instances where the company owns itself, ie the same bvdid is owned by the same shareholderbvdid
    df_non_self = df[df["bvdid"] != df["shareholderbvdid"]]

    industry_dir = Path(f"{gcap_data}/scratch/chiara/cs230/orbis/temp/industry_classifications")

    out = df_non_self.copy()

    ##### BVDID

    key = "bvdid"

    isos = df_non_self["ISO_final_subsidiary"].dropna().unique().tolist()

    # Build the file list by globbing per ISO prefix
    files = sorted(
        fp
        for iso in isos
        for fp in industry_dir.glob(f"{iso}_*industry_classifications_long_first.dta")
    )

    for fp in files:
        sub = pd.read_stata(fp)

        if key not in sub.columns:
            print(f"[skip] {fp.name}: missing '{key}'")
            break

        # rename all non-key columns with 'sub_' prefix (keep the key as-is)
        sub = sub.rename(columns={c: (c if c == key else f"sub_{c}") for c in sub.columns})

        # keep 1 row per key to avoid 1-to-many explosions
        sub = sub.drop_duplicates(subset=key, keep="first")

        # Identify using-file columns (excluding the key)
        sub_nonkey = [c for c in sub.columns if c != key]
        if not sub_nonkey:
            print(f"[skip] {fp.name}: no non-key columns")
            continue

        # Merge once; overlapping columns get a *_new suffix from the right (using) df
        merged = out.merge(sub, on=key, how="left", suffixes=("", "_new"))

        # For every column that exists in BOTH out and sub, fill missing values from the new data
        overlap = [c for c in sub_nonkey if c in out.columns]
        for c in overlap:
            new_col = f"{c}_new"
            if new_col in merged.columns:
                merged[c] = merged[c].fillna(merged[new_col])

        # Drop all temporary *_new columns introduced by this file
        new_suffix_cols = [c for c in merged.columns if c.endswith("_new")]
        merged = merged.drop(columns=new_suffix_cols)

        out = merged
        print(f"[merge] {fp.name}: out shape -> {out.shape}")

    ##### BVDID SHAREHOLDER

    key = "shareholderbvdid"

    isos = df_non_self["ISO_final_shareholder"].dropna().unique().tolist()

    # Build the file list by globbing per ISO prefix
    files = sorted(
        fp
        for iso in isos
        for fp in industry_dir.glob(f"{iso}_*industry_classifications_long_first.dta")
    )

    for fp in files:
        sub = pd.read_stata(fp)
        
        sub = sub.rename(columns={"bvdid": "shareholderbvdid"})

        if key not in sub.columns:
            print(f"[skip] {fp.name}: missing '{key}'")
            break

        # rename all non-key columns with 'share_' prefix (keep the key as-is)
        sub = sub.rename(columns={c: (c if c == key else f"share_{c}") for c in sub.columns})

        # keep 1 row per key to avoid 1-to-many explosions
        sub = sub.drop_duplicates(subset=key, keep="first")

        # Identify using-file columns (excluding the key)
        sub_nonkey = [c for c in sub.columns if c != key]
        if not sub_nonkey:
            print(f"[skip] {fp.name}: no non-key columns")
            continue

        # Merge once; overlapping columns get a *_new suffix from the right (using) df
        merged = out.merge(sub, on=key, how="left", suffixes=("", "_new"))

        # For every column that exists in BOTH out and sub, fill missing values from the new data
        overlap = [c for c in sub_nonkey if c in out.columns]
        for c in overlap:
            new_col = f"{c}_new"
            if new_col in merged.columns:
                merged[c] = merged[c].fillna(merged[new_col])

        # Drop all temporary *_new columns introduced by this file
        new_suffix_cols = [c for c in merged.columns if c.endswith("_new")]
        merged = merged.drop(columns=new_suffix_cols)

        out = merged
        print(f"[merge] {fp.name}: out shape -> {out.shape}")
        
    ### MERGE IN INDUSTRY DESCRIPTIONS
    
    nace_vars = [
        "sub_nacerev2primarycodes",
        "share_nacerev2primarycodes"
    ]

    naics_vars = [
        "sub_naicsprimarycodes",
        "share_naicsprimarycodes"
    ]

    sic_vars = [
        "sub_ussicprimarycodes",
        "share_ussicprimarycodes"
    ]
    
    path_nace  = f"{gcap_data}/scratch/yicheng/230/nace_primary_unique_dedup.tsv"
    path_naics = f"{gcap_data}/scratch/yicheng/230/naics_primary_unique_dedup.tsv"
    path_sic   = f"{gcap_data}/scratch/yicheng/230/sic_primary_unique.tsv"

    df_nace  = pd.read_csv(path_nace, sep="\t")
    df_naics = pd.read_csv(path_naics, sep="\t")
    df_sic   = pd.read_csv(path_sic, sep="\t")
    
    out = merge_code_vars(out, df_nace,  nace_vars,  "nace_primary_codes")
    out = merge_code_vars(out, df_naics, naics_vars, "naics_primary_codes")
    out = merge_code_vars(out, df_sic,   sic_vars,   "sic_primary_codes")
    
    # write as .dta (keeps Stata types nicely)
    out_path = Path(f"{out_dir}/{in_path.name}_with_industry.dta")
    pyreadstat.write_dta(out, out_path)
    print(f"    -> saved {out_path}")
    
    
######## SUBSIDIARY DATA MERGE

# Inputs
in_dir  = Path(f"{gcap_data}/scratch/chiara/cs230/orbis/temp/ownership/ownership_chunks")
out_dir = Path(f"{gcap_data}/scratch/chiara/cs230/orbis/temp/ownership/ownership_industry_code")
out_dir.mkdir(parents=True, exist_ok=True)

# parameters
country = "CN"
year = 2022
total_parts = 20
parts = ["01","02","03", "04","05", "06","07","08", "09","10", "11","12","13","14","15","16","17","18","19", "20"]

for part in parts:
    fname = f"SUB_{country}_Links_{year}_part{part}of{total_parts}.dta"
    in_path = in_dir / fname
    if not in_path.exists():
        print(f"Missing {fname} — skipping")
        continue
    df, meta = pyreadstat.read_dta(in_path)  # load whole file (no chunking)

    # ---- YOUR PROCESSING HERE ----
    # e.g., inspect shape/variables
    print(f"  Rows: {len(df):,} | Vars: {len(df.columns)}")
    
    # Drop instances where the company owns itself, ie the same bvdid is owned by the same shareholderbvdid
    df_non_self = df[df["bvdid"] != df["shareholderbvdid"]]

    industry_dir = Path(f"{gcap_data}/scratch/chiara/cs230/orbis/temp/industry_classifications")

    out = df_non_self.copy()

    ##### BVDID

    key = "bvdid"

    isos = df_non_self["ISO_final_subsidiary"].dropna().unique().tolist()

    # Build the file list by globbing per ISO prefix
    files = sorted(
        fp
        for iso in isos
        for fp in industry_dir.glob(f"{iso}_*industry_classifications_long_first.dta")
    )

    for fp in files:
        sub = pd.read_stata(fp)

        if key not in sub.columns:
            print(f"[skip] {fp.name}: missing '{key}'")
            break

        # rename all non-key columns with 'sub_' prefix (keep the key as-is)
        sub = sub.rename(columns={c: (c if c == key else f"sub_{c}") for c in sub.columns})

        # keep 1 row per key to avoid 1-to-many explosions
        sub = sub.drop_duplicates(subset=key, keep="first")

        # Identify using-file columns (excluding the key)
        sub_nonkey = [c for c in sub.columns if c != key]
        if not sub_nonkey:
            print(f"[skip] {fp.name}: no non-key columns")
            continue

        # Merge once; overlapping columns get a *_new suffix from the right (using) df
        merged = out.merge(sub, on=key, how="left", suffixes=("", "_new"))

        # For every column that exists in BOTH out and sub, fill missing values from the new data
        overlap = [c for c in sub_nonkey if c in out.columns]
        for c in overlap:
            new_col = f"{c}_new"
            if new_col in merged.columns:
                merged[c] = merged[c].fillna(merged[new_col])

        # Drop all temporary *_new columns introduced by this file
        new_suffix_cols = [c for c in merged.columns if c.endswith("_new")]
        merged = merged.drop(columns=new_suffix_cols)

        out = merged
        print(f"[merge] {fp.name}: out shape -> {out.shape}")

    ##### BVDID SHAREHOLDER

    key = "shareholderbvdid"

    isos = df_non_self["ISO_final_shareholder"].dropna().unique().tolist()

    # Build the file list by globbing per ISO prefix
    files = sorted(
        fp
        for iso in isos
        for fp in industry_dir.glob(f"{iso}_*industry_classifications_long_first.dta")
    )

    for fp in files:
        sub = pd.read_stata(fp)
        
        sub = sub.rename(columns={"bvdid": "shareholderbvdid"})

        if key not in sub.columns:
            print(f"[skip] {fp.name}: missing '{key}'")
            break

        # rename all non-key columns with 'share_' prefix (keep the key as-is)
        sub = sub.rename(columns={c: (c if c == key else f"share_{c}") for c in sub.columns})

        # keep 1 row per key to avoid 1-to-many explosions
        sub = sub.drop_duplicates(subset=key, keep="first")

        # Identify using-file columns (excluding the key)
        sub_nonkey = [c for c in sub.columns if c != key]
        if not sub_nonkey:
            print(f"[skip] {fp.name}: no non-key columns")
            continue

        # Merge once; overlapping columns get a *_new suffix from the right (using) df
        merged = out.merge(sub, on=key, how="left", suffixes=("", "_new"))

        # For every column that exists in BOTH out and sub, fill missing values from the new data
        overlap = [c for c in sub_nonkey if c in out.columns]
        for c in overlap:
            new_col = f"{c}_new"
            if new_col in merged.columns:
                merged[c] = merged[c].fillna(merged[new_col])

        # Drop all temporary *_new columns introduced by this file
        new_suffix_cols = [c for c in merged.columns if c.endswith("_new")]
        merged = merged.drop(columns=new_suffix_cols)

        out = merged
        print(f"[merge] {fp.name}: out shape -> {out.shape}")
        
    ### MERGE IN INDUSTRY DESCRIPTIONS
    
    nace_vars = [
        "sub_nacerev2primarycodes",
        "share_nacerev2primarycodes"
    ]

    naics_vars = [
        "sub_naicsprimarycodes",
        "share_naicsprimarycodes"
    ]

    sic_vars = [
        "sub_ussicprimarycodes",
        "share_ussicprimarycodes"
    ]
    
    path_nace  = f"{gcap_data}/scratch/yicheng/230/nace_primary_unique_dedup.tsv"
    path_naics = f"{gcap_data}/scratch/yicheng/230/naics_primary_unique_dedup.tsv"
    path_sic   = f"{gcap_data}/scratch/yicheng/230/sic_primary_unique.tsv"

    df_nace  = pd.read_csv(path_nace, sep="\t")
    df_naics = pd.read_csv(path_naics, sep="\t")
    df_sic   = pd.read_csv(path_sic, sep="\t")
    
    out = merge_code_vars(out, df_nace,  nace_vars,  "nace_primary_codes")
    out = merge_code_vars(out, df_naics, naics_vars, "naics_primary_codes")
    out = merge_code_vars(out, df_sic,   sic_vars,   "sic_primary_codes")
    
    # write as .dta (keeps Stata types nicely)
    out_path = Path(f"{out_dir}/{in_path.name}_with_industry.dta")
    pyreadstat.write_dta(out, out_path)
    print(f"    -> saved {out_path}")