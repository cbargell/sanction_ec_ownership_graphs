gcap_data = "/oak/stanford/groups/maggiori/GCAP/data"

def lookup_bvd_name(bvdid, path):
    """
    Return the NAME corresponding to a given BvD ID number,
    or None if not found.
    """
    with open(path, encoding="utf-8") as f:
        # Skip header
        next(f)
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            bvd_col, name = line.split("\t", 1)  # split into ID and NAME
            if bvd_col == bvdid:
                return name
    return None


path = f"{gcap_data}/raw/orbis/latest/firm_description/txt/BvD_ID_and_Name.txt"

print(lookup_bvd_name("US710415188", path))  # -> Walmart Inc.
print(lookup_bvd_name("US911646860", path))  # -> Amazon.Com, Inc.
