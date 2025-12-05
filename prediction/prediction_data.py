import pandas as pd
import os
import glob
import math

gcap_data = "/oak/stanford/groups/maggiori/GCAP/data"

# Our firm-lelve embeddings are created on the basis of the BvD ID number
# Our sanctions and export controls data are created on the basis of factset_entity_id
# We need to create a crosswalk from factset_entity_id to BvD ID number
# We can do this by using the identifiers.txt file, which maps BvD ID number to Ticker symbol and ISIN number
# We can then use the factset_entity_id to ISIN mapping to create a crosswalk from factset_entity_id to BvD ID number
# We use this crosswalk to create a prediction dataset that contains the firm-level embeddings, the sanctions and export controls data, and firm-level features.

# -------------------------------------------------
# CREATE MAPPING FROM ISIN TO BVDID
# -------------------------------------------------

input_path = f"{gcap_data}/raw/orbis/latest/firm_description/txt/Identifiers.txt"
output_dir = f"{gcap_data}/scratch/chiara/cs230/orbis/temp/identifiers"

os.makedirs(output_dir, exist_ok=True)

cols_to_keep = ["BvD ID number", "Ticker symbol", "ISIN number"]

# 1) Count total data rows (excluding header)
with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
    total_lines = sum(1 for _ in f)  # includes header
total_rows = total_lines - 1

n_chunks = 10
chunk_size = math.ceil(total_rows / n_chunks)

print(f"Total rows: {total_rows}, chunk size: {chunk_size}")

# 2) Read in chunks and save each chunk
reader = pd.read_csv(
    input_path,
    sep="\t",
    usecols=cols_to_keep,
    dtype=str,
    low_memory=False,
    chunksize=chunk_size
)

for i, chunk in enumerate(reader, start=1):
    out_path = os.path.join(output_dir, f"identifiers_chunk_{i:02d}.csv")
    chunk.to_csv(out_path, index=False)
    print(f"Saved chunk {i} to {out_path}")

# 3) Process each chunk and drop rows where BOTH ticker and isin are missing (if they are missing, we can't use them to create the crosswalk from bvdid to factset_entity_id)

# Directory where your chunk files live
in_dir = f"{gcap_data}/scratch/chiara/cs230/orbis/temp/ticker"

# Pattern for the chunk files you already created
pattern = os.path.join(in_dir, "tickers_chunk_*.csv")
files = sorted(glob.glob(pattern))

print(f"Found {len(files)} identifier chunks.")

# Output file that will contain all cleaned + appended data
out_path = os.path.join(in_dir, "identifiers_all_clean.csv")

first_file = True  # to control header writing once

for fpath in files:
    fname = os.path.basename(fpath)
    print(f"\nProcessing {fname} ...")

    # Read one chunk
    df = pd.read_csv(fpath, dtype=str, low_memory=False)

    # Rename columns
    df = df.rename(
        columns={
            "BvD ID number": "bvdid",
            "Ticker symbol": "ticker",
            "ISIN number": "isin",
        }
    )

    # Keep only the three columns (helps with memory / size)
    df = df[["bvdid", "ticker", "isin"]]

    # Drop rows where BOTH ticker and isin are missing
    before = len(df)
    df = df[~(df["ticker"].isna() & df["isin"].isna())]
    after = len(df)
    print(f"  Dropped {before - after} rows with missing ticker & isin "
          f"({before} -> {after}).")

    # Append this cleaned chunk to the big output file
    df.to_csv(
        out_path,
        mode="w" if first_file else "a",  # write for first, append after
        index=False,
        header=first_file                 # header only on the first chunk
    )

    first_file = False

print(f"\nDone. All cleaned chunks appended to: {out_path}")

# keep only the bvdid and isin columns, drop duplicates
out_path = "/Volumes/data/scratch/chiara/cs230/orbis/temp/ticker/identifiers_all_clean.csv"
isin_bvdid = pd.read_csv(out_path)
isin_bvdid = isin_bvdid[["isin", "bvdid"]].drop_duplicates()
isin_bvdid = isin_bvdid.dropna(subset=["isin"])
isin_bvdid = isin_bvdid.dropna(subset=["bvdid"])
isin_bvdid.to_csv(f"{gcap_data}/scratch/chiara/cs230/orbis/temp/ticker/isin_bvdid.csv", index=False)

# -------------------------------------------------
# IMPORT MAPPING FROM ISIN TO FACTSETID
# -------------------------------------------------
isin_factset = pd.read_stata(f"{gcap_data}/shared/WRDS_Transcripts/temp/companyid_to_factsetid/isin_to_entity.dta")

# select only the factset_entity_id and isin columns
isin_factset_fewcols = isin_factset[["factset_entity_id", "isin"]]

# -------------------------------------------------
# CREATE MAPPING FROM FACTSETID TO BVDID
# -------------------------------------------------

# import mapping from isin to bvdid
isin_bvdid = pd.read_csv(f"{gcap_data}/scratch/chiara/cs230/orbis/temp/ticker/isin_bvdid.csv")

# merge the two dataframes on the isin column
# merge
factset_to_bvdid = isin_factset_fewcols.merge(
    isin_bvdid,
    left_on="isin",   
    right_on="isin", 
    how="left",
    indicator=True
)

# drop rows where the factset_entity_id is missing
factset_to_bvdid = factset_to_bvdid.dropna(
    subset=["bvdid"]
)

# keep only the factset_entity_id , isin, and bvdid columns
factset_to_bvdid = factset_to_bvdid[["factset_entity_id","isin", "bvdid"]]

# generate a dataset with one unique row for each factset_entity_id, with a list of all bvdid's
factset_to_bvdid_list_df = (
    factset_to_bvdid
    .groupby("factset_entity_id", as_index=False)
    .agg({"bvdid": list})
    .rename(columns={"bvdid": "bvdid_list"})
)
assert factset_to_bvdid_list_df["factset_entity_id"].is_unique

# -----------------------------------------------------------------------
# MERGE CLAYTON ET AL POLICY SANCTION DATA WITH FACTSETID TO BVDID MAPPING
# -----------------------------------------------------------------------

# IMPORT CLAYTON ET AL POLICY SANCTION DATA
broad_long_wide = pd.read_stata(f"{gcap_data}/ai_geo1/temp/build_test_sep2025/temp/broad_long_wide.dta")
broad_long_wide = broad_long_wide[["factset_entity_id", "quarter","year","sanctions_any","export_controls_any","country_iso","siccode","primary_sic_code","factset_short_name"]]

# merge with the mapping from factset entity id to bvdid list
merged_sanctioned = broad_long_wide.merge(
    factset_to_bvdid_list_df,
    left_on="factset_entity_id",   
    right_on="factset_entity_id", 
    how="left",
    indicator=True
)

# convert the bvdid_list column to a string
merged_sanctioned["bvdid_list"] = merged_sanctioned["bvdid_list"].astype(str)

# save the merged dataset to a stata file
out_path = f"{gcap_data}/scratch/chiara/cs230/orbis/temp/broad_long_wide_bvdid.dta"
merged_sanctioned.to_stata(
    out_path,
    write_index=False,
    version=118,
    convert_dates={"quarter": "tq"}   # <-- key line: Stata quarterly date
)

# -----------------------------------------------------------------------
# BUILD DATASET WITH ONE ROW PER BVDID
# -----------------------------------------------------------------------

in_path = f"{gcap_data}/scratch/chiara/cs230/orbis/temp/broad_long_wide_bvdid.dta"
broad_long_wide_bvdid = pd.read_stata(in_path)

# drop when bvdid is missing
broad_long_wide_bvdid_mapped = broad_long_wide_bvdid[broad_long_wide_bvdid["bvdid_list"] != "nan"]

# drop column _merge
broad_long_wide_bvdid_mapped = broad_long_wide_bvdid_mapped.drop(columns=["_merge"])

# generate a unique identifier for each factset_entity_id and quarter
broad_long_wide_bvdid_mapped["feq_id"] = (
    broad_long_wide_bvdid_mapped
    .groupby(["factset_entity_id", "quarter"])
    .ngroup()
    + 1  # make it start at 1 instead of 0 (Stata-like)
)
broad_long_wide_bvdid_mapped = broad_long_wide_bvdid_mapped.sort_values(
    by="feq_id"
).reset_index(drop=True)

# turn "DE123, US456" → ["DE123", "US456"]
broad_long_wide_bvdid_mapped["bvdid_list"] = (
    broad_long_wide_bvdid_mapped["bvdid_list"]
    .astype(str)
    .str.split(",")
    .apply(lambda lst: [x.strip(" []'\" ") for x in lst] if isinstance(lst, list) else lst)
)

# explode to multiple rows
broad_long_wide_bvdid_mapped_exploded = broad_long_wide_bvdid_mapped.explode("bvdid_list", ignore_index=True)

# -----------------------------------------------------------------------
# MERGE EMBEDDING DATA WITH CLAYTON DATA
# -----------------------------------------------------------------------

#load embedding data
embedding = pd.read_csv(f"{gcap_data}/scratch/yicheng/230/emb_vgae_directed_10_cpu.csv")

# merge clayton data with embedding data
blw_bvdid_exploded_embedding = broad_long_wide_bvdid_mapped_exploded.merge(
    embedding,
    left_on="bvdid_list",   
    right_on="firm_id", 
    how="left",
    indicator=True
)

# only keep observations that we manage to map to the embedding
blw_bvdid_exploded_embedding = blw_bvdid_exploded_embedding[
    blw_bvdid_exploded_embedding["_merge"] == "both"
].drop(columns=["_merge"])


# EXPORT to CSV
out_dir = f"{gcap_data}/scratch/chiara/cs230/orbis/output"
os.makedirs(out_dir, exist_ok=True)

# CSV
out_csv = os.path.join(out_dir, "blw_bvdid_exploded_embedding_10_cpu.csv")
blw_bvdid_exploded_embedding.to_csv(out_csv, index=False)