import math
from pathlib import Path
import pyreadstat

#### SHAREHOLDERS DATA

# inputs 
in_path = Path(f"{gcap_data}/raw/orbis/latest/ownership/dta/ownership_structure/appended_countries/CN/SHARE_CN_Links_2022.dta")
out_dir = Path(f"{gcap_data}/scratch/chiara/cs230/orbis/temp/ownership/ownership_chunks")
base = "SHARE_CN_Links_2022"
out_dir.mkdir(parents=True, exist_ok=True)

# params
n_chunks = 20

# get total rows without loading data 
_, meta = pyreadstat.read_dta(in_path, metadataonly=True)
total_rows = int(meta.number_rows)
print(f"Total rows: {total_rows:,}")

# compute per-chunk sizes (distribute remainder to first chunks) 
base_size = total_rows // n_chunks
remainder = total_rows % n_chunks

sizes = [(base_size + 1 if i < remainder else base_size) for i in range(n_chunks)]
assert sum(sizes) == total_rows

# iterate and save chunks 
offset = 0
saved_rows = 0
for i, size in enumerate(sizes, start=1):
    start = offset
    stop = offset + size - 1
    print(f"Reading chunk {i}/{n_chunks}: rows {start:,}..{stop:,} (size={size:,})")
    df, _ = pyreadstat.read_dta(in_path, row_offset=offset, row_limit=size)

    out_path = out_dir / f"{base}_part{i:02d}of{n_chunks:02d}.dta"
    pyreadstat.write_dta(df, out_path)
    print(f" -> saved {out_path} ({len(df):,} rows)")
    saved_rows += len(df)

    # advance offset and free memory
    offset += size
    del df

# (optional) sanity check
assert saved_rows == total_rows, f"Row count mismatch! saved {saved_rows:,} vs total {total_rows:,}"
print(f"Done. {n_chunks} chunks cover all rows exactly once.")

#### SUBSIDIARIES DATA

import math
from pathlib import Path
import pyreadstat

#  inputs
in_path = Path(f"{gcap_data}/raw/orbis/latest/ownership/dta/ownership_structure/appended_countries/CN/SUB_CN_Links_2022.dta")
out_dir = Path(f"{gcap_data}/scratch/chiara/cs230/orbis/temp/ownership/ownership_chunks")
base = "SUB_CN_Links_2022"
out_dir.mkdir(parents=True, exist_ok=True)

# params
n_chunks = 20

# get total rows without loading data
_, meta = pyreadstat.read_dta(in_path, metadataonly=True)
total_rows = int(meta.number_rows)
print(f"Total rows: {total_rows:,}")

# compute per-chunk sizes (distribute remainder to first chunks)
base_size = total_rows // n_chunks
remainder = total_rows % n_chunks

sizes = [(base_size + 1 if i < remainder else base_size) for i in range(n_chunks)]
assert sum(sizes) == total_rows

# iterate and save chunks
offset = 0
saved_rows = 0
for i, size in enumerate(sizes, start=1):
    start = offset
    stop = offset + size - 1
    print(f"Reading chunk {i}/{n_chunks}: rows {start:,}..{stop:,} (size={size:,})")
    df, _ = pyreadstat.read_dta(in_path, row_offset=offset, row_limit=size)
    
    # drop rows where ISO_final_shareholder == "CN"
    # These rows will already be present in the SHARE datasets built above
    if "ISO_final_shareholder" in df.columns:
        before = len(df)
        df = df[df["ISO_final_shareholder"] != "CN"].copy()
        after = len(df)
        print(f"    Dropped {before - after:,} rows with ISO_final_shareholder == 'CN'")
    else:
        print("    WARNING: Column 'ISO_final_shareholder' not found in chunk")

    out_path = out_dir / f"{base}_part{i:02d}of{n_chunks:02d}.dta"
    pyreadstat.write_dta(df, out_path)
    print(f" -> saved {out_path} ({len(df):,} rows)")
    saved_rows += len(df)

    # advance offset and free memory
    offset += size
    del df