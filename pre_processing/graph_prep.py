import os, re, csv
import numpy as np
import pandas as pd

# format data
in_template = "/oak/stanford/groups/maggiori/GCAP/data/scratch/chiara/cs230/orbis/temp/ownership/ownership_industry_code/SHARE_CN_Links_2022_part13of20.dta_with_industry.dta"
out_template = "/oak/stanford/groups/maggiori/GCAP/data/scratch/yicheng/230/SHARE_CN_Links_2022_part13of20.dta_with_industry.csv"

cols = [
    "bvdid", "shareholderbvdid",
    "ISO_final_subsidiary", "ISO_final_shareholder",
    "sub_nacerev2primarycodes", "sub_naicsprimarycodes", "sub_ussicprimarycodes",
    "share_nacerev2primarycodes", "share_naicsprimarycodes", "share_ussicprimarycodes",
    "sub_nace_primary_desc", "share_nace_primary_desc",
    "sub_naics_primary_desc", "share_naics_primary_desc",
    "sub_sic_primary_desc", "share_sic_primary_desc",
]

# helpers
def first_nonmissing_across(df, cols):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series(pd.NA, index=df.index)
    arr = df[cols].to_numpy(dtype=object)
    mask = ~pd.isna(arr) & (arr != "")
    any_nonmissing = mask.any(axis=1)
    idx = np.where(any_nonmissing, mask.argmax(axis=1), 0)
    out = arr[np.arange(len(df)), idx].astype(object)
    out[~any_nonmissing] = pd.NA
    return pd.Series(out, index=df.index)

def longest_string_across(df, cols):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series(pd.NA, index=df.index)
    arr = df[cols].fillna("").to_numpy(dtype=str)
    lens = np.char.str_len(arr)
    any_nonempty = (lens > 0).any(axis=1)
    idx = lens.argmax(axis=1)
    out = arr[np.arange(len(df)), idx].astype(object)
    out[~any_nonempty] = pd.NA
    return pd.Series(out, index=df.index)

sub_code_cols   = ['sub_ussicprimarycodes','sub_naicsprimarycodes','sub_nacerev2primarycodes']
share_code_cols = ['share_ussicprimarycodes','share_naicsprimarycodes','share_nacerev2primarycodes']

sub_desc_cols   = ['sub_sic_primary_desc','sub_naics_primary_desc','sub_nace_primary_desc']
share_desc_cols = ['share_sic_primary_desc','share_naics_primary_desc','share_nace_primary_desc']

# process parts 1..20
for i in range(1, 10):
    in_path  = re.sub(r"part\d{1,2}of20", f"part0{i}of20", in_template)
    out_path = re.sub(r"part\d{1,2}of20", f"part0{i}of20", out_template)

    if not os.path.exists(in_path):
        print(f"[skip] missing: {in_path}")
        continue

    try:
        df = pd.read_stata(in_path)

        # keep listed columns that exist
        df = df.filter(items=cols)

        # first non-missing codes
        df['sub_sic']         = first_nonmissing_across(df, sub_code_cols)
        df['shareholder_sic'] = first_nonmissing_across(df, share_code_cols)

        # longest description
        df['sub_sic_descr']         = longest_string_across(df, sub_desc_cols)
        df['shareholder_sic_descr'] = longest_string_across(df, share_desc_cols)

        # drop originals
        drop_cols = [c for c in (sub_code_cols + share_code_cols + sub_desc_cols + share_desc_cols) if c in df.columns]
        df.drop(columns=drop_cols, inplace=True, errors='ignore')
        
        df = df.rename(columns={
            "ISO_final_subsidiary": "sub_iso",
            "ISO_final_shareholder": "shareholder_iso",
        })

        # write CSV (ensure dir exists)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)

        print(f"[ok] wrote: {out_path}")
    except Exception as e:
        print(f"[error] {in_path}: {e}")




template = "/oak/stanford/groups/maggiori/GCAP/data/scratch/yicheng/230/SHARE_CN_Links_2022_part13of20.dta_with_industry.csv"

for i in range(1, 10):
    path = re.sub(r"part\d{1,2}of20", f"part0{i}of20", template)
    if not os.path.exists(path):
        print(f"[skip] missing: {path}")
        continue

    tmp = path + ".tmp"
    try:
        with open(path, "r", newline="", encoding="utf-8") as f_in:
            sample = f_in.read(4096)
            f_in.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample)
            except Exception:
                dialect = csv.excel

            reader = csv.reader(f_in, dialect)
            header = next(reader)

            # rename only these two columns
            new_header = [
                ("sub_iso" if c == "ISO_final_subsidiary"
                 else "shareholder_iso" if c == "ISO_final_shareholder"
                 else c)
                for c in header
            ]

            with open(tmp, "w", newline="", encoding="utf-8") as f_out:
                writer = csv.writer(f_out, dialect)
                writer.writerow(new_header)
                writer.writerows(reader)

        #replace
        os.replace(tmp, path)
        print(f"[ok] updated header: {path}")
    except Exception as e:
        # clean up temp on error
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        print(f"[error] {path}: {e}")


template = "/oak/stanford/groups/maggiori/GCAP/data/scratch/yicheng/230/SHARE_CN_Links_2022_part13of20.dta_with_industry.csv"

def is_blank(s):
    # treat NaN or whitespace-only as blank
    return s.isna() | s.astype(str).str.strip().eq("")

for i in range(1, 10):
    path = re.sub(r"part\d{1,2}of20", f"part0{i}of20", template)
    if not os.path.exists(path):
        print(f"[skip] missing: {path}")
        continue

    try:
        df = pd.read_csv(path, low_memory=False)

        # drop rows if either description is blank (NaN or empty/whitespace)
        sub_col   = "sub_sic_descr"
        share_col = "shareholder_sic_descr"
        if sub_col in df.columns and share_col in df.columns:
            mask_drop = is_blank(df[sub_col]) | is_blank(df[share_col])
            before = len(df)
            df = df.loc[~mask_drop].copy()
            after = len(df)
            print(f"[ok] {path}: dropped {before - after} rows, kept {after}")
        else:
            print(f"[warn] {path}: missing '{sub_col}' or '{share_col}', no row drop performed")

        df.to_csv(path, index=False)

    except Exception as e:
        print(f"[error] {path}: {e}")

