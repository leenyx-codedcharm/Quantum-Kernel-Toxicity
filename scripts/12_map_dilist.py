from pathlib import Path
from urllib.parse import quote
import re, time
import pandas as pd
import requests
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
XLSX = RAW / "DILIst Supplementary Table.xlsx"
FDA_NAMES = RAW / "dilirank2_pubchem_name_cache.csv"
FDA_PROPS = RAW / "dilirank2_pubchem_properties.csv"
NAME_CACHE = RAW / "dilist_pubchem_name_cache.csv"
PROP_CACHE = RAW / "dilist_pubchem_properties.csv"

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
S = requests.Session()
S.headers.update({"User-Agent": "CSE491-DILIst-mapping/1.0"})


def norm(x):
    return re.sub(r"\s+", " ", str(x)).strip().casefold()


def smiles_of(p):
    for k in ("SMILES", "IsomericSMILES",
              "CanonicalSMILES", "ConnectivitySMILES"):
        v = p.get(k, "")
        if v and str(v).strip() and str(v).lower() != "nan":
            return str(v).strip()
    return ""


def readcsv(p):
    return pd.read_csv(p, dtype=str).fillna("") if p.exists() else pd.DataFrame()


def getjson(url):
    for n in range(5):
        try:
            r = S.get(url, timeout=45)
            if r.status_code == 404:
                return {"_404": True}
            if r.status_code in (202, 429, 500, 502, 503, 504):
                time.sleep(min(30, 2 ** n))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if n == 4:
                print("PubChem request failed:", e)
                return None
            time.sleep(min(30, 2 ** n))
    return None


def load_names(path):
    d, out = readcsv(path), {}
    if not d.empty and "CompoundName" in d.columns:
        for _, r in d.iterrows():
            out[norm(r["CompoundName"])] = (
                r.get("Candidate_CIDs", ""),
                r.get("Lookup_Status", "")
            )
    return out


def main():

    print("DILIst PubChem mapping; reusing DILIrank caches")

    if not XLSX.exists():
        raise SystemExit(f"Workbook not found: {XLSX}")

    df = pd.read_excel(
        XLSX, sheet_name="DILIst", engine="openpyxl"
    )

    df.columns = [str(c).strip() for c in df.columns]

    cols = {
        re.sub(r"[^a-z0-9]", "", c.casefold()): c
        for c in df.columns
    }

    req = ("dilistid", "compoundname", "dilistclassification")

    if not all(k in cols for k in req):
        raise SystemExit(f"Unexpected columns: {df.columns.tolist()}")

    df["DILIST_ID"] = (
        df[cols["dilistid"]].astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )

    df["CompoundName"] = df[cols["compoundname"]].astype(str).str.strip()

    df["Y"] = pd.to_numeric(
        df[cols["dilistclassification"]], errors="coerce"
    )

    route = cols.get("routsofadministration")
    df["Route_of_Administration"] = df[route] if route else ""

    if df["Y"].isna().any() or not set(df["Y"].unique()).issubset({0, 1}):
        raise SystemExit("Missing or unexpected DILIst labels.")

    df["Y"] = df["Y"].astype(int)
    df.to_csv(RAW / "dilist_original.csv", index=False)

    # Reuse exact-name CID lookups from DILIrank.
    names = load_names(FDA_NAMES)
    names.update(load_names(NAME_CACHE))

    unique = {norm(x): str(x).strip() for x in df["CompoundName"].unique()}

    todo = [
        k for k in unique
        if k not in names or names[k][1] in ("", "api_error")
    ]

    print(f"DILIst rows: {len(df)} | unique names: {len(unique)}")
    print(f"Name lookups reused: {len(unique)-len(todo)} | new lookups: {len(todo)}")

    for i, k in enumerate(todo, 1):

        obj = getjson(
            f"{PUG}/compound/name/{quote(unique[k], safe='')}/cids/JSON"
        )

        cids = [] if not obj or obj.get("_404") else (
            obj.get("IdentifierList", {}).get("CID", [])
        )

        cids = list(dict.fromkeys(str(x) for x in cids))

        status = (
            "unique_cid_candidate" if len(cids) == 1
            else "multiple_cids_manual_review" if cids
            else "not_found"
        )

        names[k] = (",".join(cids), status)

        if i % 20 == 0 or i == len(todo):

            pd.DataFrame([
                {
                    "CompoundName": unique.get(a, a),
                    "Candidate_CIDs": b[0],
                    "Lookup_Status": b[1]
                }
                for a, b in names.items()
            ]).to_csv(NAME_CACHE, index=False)

            print(f"  Name lookups: {i}/{len(todo)}")

        time.sleep(0.3)

    pd.DataFrame([
        {
            "CompoundName": unique.get(a, a),
            "Candidate_CIDs": b[0],
            "Lookup_Status": b[1]
        }
        for a, b in names.items()
    ]).to_csv(NAME_CACHE, index=False)

    # Merge PubChem property caches.
    props = {}

    for path in (FDA_PROPS, PROP_CACHE):

        d = readcsv(path)

        if not d.empty and "CID" in d.columns:

            for _, r in d.iterrows():

                cid = str(r["CID"]).strip()

                if cid:
                    props.setdefault(cid, {}).update(r.to_dict())

    row_cids, row_status, row_candidates = [], [], []

    for _, r in df.iterrows():

        ctext, _ = names.get(
            norm(r["CompoundName"]), ("", "not_found")
        )

        candidates = [x for x in ctext.split(",") if x]

        cid = candidates[0] if len(candidates) == 1 else ""

        status = (
            "unique_name_candidate" if len(candidates) == 1
            else "multiple_cids_manual_review" if len(candidates) > 1
            else "not_found"
        )

        row_cids.append(cid)
        row_status.append(status)
        row_candidates.append(ctext)

    # Retrieve only properties not already present in cache.
    missing = sorted(set(
        c for c in row_cids
        if c and not smiles_of(props.get(c, {}))
    ))

    print("CID properties to retrieve:", len(missing))

    for start in range(0, len(missing), 50):

        batch = missing[start:start+50]

        obj = getjson(
            f"{PUG}/compound/cid/{','.join(batch)}"
            "/property/Title,SMILES,InChIKey/JSON"
        )

        rows = (obj or {}).get(
            "PropertyTable", {}
        ).get("Properties", [])

        if not rows:

            for cid in batch:

                one = getjson(
                    f"{PUG}/compound/cid/{cid}"
                    "/property/Title,SMILES,InChIKey/JSON"
                )

                for item in (one or {}).get(
                    "PropertyTable", {}
                ).get("Properties", []):

                    props[str(item.get("CID"))] = item

                time.sleep(0.25)

        else:

            for item in rows:
                props[str(item.get("CID"))] = item

        pd.DataFrame(list(props.values())).to_csv(
            PROP_CACHE, index=False
        )

        print(
            f"  Property batches: "
            f"{min(start+len(batch), len(missing))}/{len(missing)}"
        )

        time.sleep(0.25)

    # Validate all mapped structures using RDKit.
    audit = df.copy()

    audit["PubChem_Candidate_CIDs"] = row_candidates
    audit["PubChem_CID"] = row_cids
    audit["Mapping_Status"] = row_status

    titles, smi_list, keys, canon, rdstatus = [], [], [], [], []

    for cid in row_cids:

        p = props.get(cid, {}) if cid else {}
        smi = smiles_of(p)

        titles.append(p.get("Title", ""))
        smi_list.append(smi)
        keys.append(p.get("InChIKey", ""))

        mol = Chem.MolFromSmiles(smi) if smi else None

        if not smi:
            canon.append("")
            rdstatus.append("no_smiles")

        elif mol is None:
            canon.append("")
            rdstatus.append("invalid_smiles")

        else:
            canon.append(
                Chem.MolToSmiles(
                    mol, canonical=True, isomericSmiles=True
                )
            )
            rdstatus.append("valid")

    audit["PubChem_Title"] = titles
    audit["Drug"] = smi_list
    audit["InChIKey"] = keys
    audit["Canonical_SMILES"] = canon
    audit["RDKit_Status"] = rdstatus

    for i in audit.index:

        if audit.at[i, "Mapping_Status"] == "unique_name_candidate":

            audit.at[i, "Mapping_Status"] = (
                "mapped_candidate_identity_review_required"
                if audit.at[i, "RDKit_Status"] == "valid"
                else audit.at[i, "RDKit_Status"]
            )

    audit.to_csv(RAW / "dilist_mapping_audit.csv", index=False)

    provisional = audit[
        audit["Mapping_Status"] ==
        "mapped_candidate_identity_review_required"
    ].copy()

    provisional["Drug_ID"] = provisional["DILIST_ID"]

    provisional[[
        "Drug_ID", "DILIST_ID", "CompoundName", "Drug", "Y",
        "Route_of_Administration", "PubChem_CID", "PubChem_Title",
        "InChIKey", "Canonical_SMILES", "Mapping_Status"
    ]].to_csv(RAW / "dilist_provisional.csv", index=False)

    provisional[[
        "DILIST_ID", "CompoundName", "Y", "PubChem_CID",
        "PubChem_Title", "Drug", "InChIKey", "Canonical_SMILES"
    ]].to_csv(RAW / "dilist_identity_review.csv", index=False)

    review = audit[
        audit["Mapping_Status"] !=
        "mapped_candidate_identity_review_required"
    ]

    review.to_csv(RAW / "dilist_unmapped_review.csv", index=False)

    valid = audit[audit["Canonical_SMILES"] != ""].copy()

    dup = valid[
        valid.duplicated("Canonical_SMILES", keep=False)
    ].sort_values("Canonical_SMILES")

    dup.to_csv(RAW / "dilist_duplicate_structure_review.csv", index=False)

    ck = valid.groupby("Canonical_SMILES")["Y"].nunique()
    ck = ck[ck > 1].index

    valid[
        valid["Canonical_SMILES"].isin(ck)
    ].sort_values("Canonical_SMILES").to_csv(
        RAW / "dilist_conflicting_label_review.csv", index=False
    )

    print("\nDILIST MAPPING SUMMARY")
    print("Original rows:", len(audit))
    print(audit["Mapping_Status"].value_counts().to_string())
    print("Valid RDKit structures:", sum(audit["RDKit_Status"] == "valid"))
    print("Provisional mapped rows:", len(provisional))
    print("Unique canonical structures:", valid["Canonical_SMILES"].nunique())
    print("Duplicate-structure rows:", len(dup))
    print("Conflicting canonical structures:", len(ck))

    print("\nProvisional label counts:")
    if len(provisional):
        print(provisional["Y"].value_counts().sort_index().to_string())

    print("Manual review rows:", len(review))

    print("\nIMPORTANT: Provisional rows are not final training data.")
    print("Verify identities and cross-dataset overlap first.")


if __name__ == "__main__":
    main()
