"""Download FDA DILIrank 2.0 and map FDA compound names to PubChem structures.

Outputs under data/raw/:
  dilirank2_fda.xlsx                    Original FDA workbook
  dilirank2_fda_original.csv            Original rows/columns
  dilirank2_pubchem_name_cache.csv      Resumable name-to-CID lookup cache
  dilirank2_pubchem_properties.csv      PubChem properties for resolved CIDs
  dilirank2_mapping_audit.csv           Full row-level mapping audit
  dilirank2_unmapped_review.csv         Manual-review queue (can add CID overrides)
  dilirank2.csv                         Model-ready file: Drug_ID, Drug, Y + provenance

Binary study mapping (pre-declared decision):
  vMost-DILI-concern -> 1
  vLess-DILI-concern -> 1
  vNo-DILI-concern   -> 0
  Ambiguous          -> excluded from binary file (kept in audit)

The script never chooses the first PubChem CID when a name returns multiple CIDs.
Those rows are sent to the manual-review file instead.
"""
from __future__ import annotations

import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).parent.name == "scripts" else Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

FDA_URL = "https://www.fda.gov/media/113052/download?attachment="
FDA_XLSX = RAW / "dilirank2_fda.xlsx"
ORIGINAL_CSV = RAW / "dilirank2_fda_original.csv"
NAME_CACHE_CSV = RAW / "dilirank2_pubchem_name_cache.csv"
PROPERTY_CACHE_CSV = RAW / "dilirank2_pubchem_properties.csv"
AUDIT_CSV = RAW / "dilirank2_mapping_audit.csv"
REVIEW_CSV = RAW / "dilirank2_unmapped_review.csv"
MODEL_CSV = RAW / "dilirank2.csv"

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
HEADERS = {"User-Agent": "CSE491-Project10-DILIrank2-research/1.0"}

session = requests.Session()
session.headers.update(HEADERS)


def get_json(url: str, attempts: int = 6):
    """GET JSON from PubChem with backoff for rate limits/transient errors."""
    for attempt in range(attempts):
        try:
            response = session.get(url, timeout=45)
            if response.status_code == 404:
                return {"_http_status": 404}
            if response.status_code in (202, 429, 500, 502, 503, 504):
                wait = min(60, 2 ** attempt)
                print(f"  PubChem HTTP {response.status_code}; waiting {wait}s")
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == attempts - 1:
                print(f"  Request failed: {exc}")
                return None
            wait = min(60, 2 ** attempt)
            print(f"  Request error; waiting {wait}s: {exc}")
            time.sleep(wait)
    return None


def download_fda_workbook():
    if FDA_XLSX.exists() and zipfile.is_zipfile(FDA_XLSX):
        print(f"Using existing FDA workbook: {FDA_XLSX}")
        return

    print("Downloading FDA DILIrank 2.0 workbook...")
    response = session.get(FDA_URL, timeout=90)
    response.raise_for_status()
    content = response.content
    if not zipfile.is_zipfile(io.BytesIO(content)):
        content_type = response.headers.get("Content-Type", "unknown")
        raise RuntimeError(
            "FDA response was not an XLSX workbook. "
            f"Content-Type={content_type}. Open the FDA DILIrank 2.0 page, "
            "click 'Download Complete List', and save the workbook as "
            f"{FDA_XLSX.name} in {RAW}."
        )
    FDA_XLSX.write_bytes(content)
    print(f"Saved FDA workbook: {FDA_XLSX}")


def normalize_category(value) -> str:
    return re.sub(r"[^a-z]", "", str(value).lower())


def binary_label(value):
    """Return 1, 0 or pd.NA using the study's stated binary mapping."""
    category = normalize_category(value)
    if "ambiguous" in category:
        return pd.NA
    if "mostdiliconcern" in category or "lessdiliconcern" in category:
        return 1
    if "nodiliconcern" in category:
        return 0
    return pd.NA


def load_existing_overrides() -> dict[str, str]:
    """Read user-entered PubChem CID overrides from the previous review file."""
    if not REVIEW_CSV.exists():
        return {}
    try:
        review = pd.read_csv(REVIEW_CSV, dtype=str).fillna("")
        if "LTKBID" not in review.columns or "PubChem_CID_override" not in review.columns:
            return {}
        return {
            str(row["LTKBID"]).strip(): str(row["PubChem_CID_override"]).strip()
            for _, row in review.iterrows()
            if str(row["PubChem_CID_override"]).strip()
        }
    except Exception as exc:
        print(f"Could not read previous manual overrides: {exc}")
        return {}


def load_name_cache() -> dict[str, dict]:
    if not NAME_CACHE_CSV.exists():
        return {}
    try:
        cache = pd.read_csv(NAME_CACHE_CSV, dtype=str).fillna("")
        result = {}
        for _, row in cache.iterrows():
            result[row["CompoundName"]] = {
                "candidate_cids": row.get("Candidate_CIDs", ""),
                "lookup_status": row.get("Lookup_Status", ""),
            }
        return result
    except Exception:
        return {}


def save_name_cache(cache: dict[str, dict]):
    rows = [
        {"CompoundName": name,
         "Candidate_CIDs": item.get("candidate_cids", ""),
         "Lookup_Status": item.get("lookup_status", "")}
        for name, item in cache.items()
    ]
    pd.DataFrame(rows).to_csv(NAME_CACHE_CSV, index=False)


def lookup_name(name: str) -> tuple[list[str], str]:
    encoded = quote(name, safe="")
    payload = get_json(f"{PUG}/compound/name/{encoded}/cids/JSON")
    if payload is None:
        return [], "api_error"
    if payload.get("_http_status") == 404:
        return [], "not_found"
    cids = payload.get("IdentifierList", {}).get("CID", [])
    cids = [str(cid) for cid in cids]
    if len(cids) == 1:
        return cids, "unique_cid"
    if len(cids) > 1:
        return cids, "multiple_cids_manual_review"
    return [], "not_found"


def load_property_cache() -> dict[str, dict]:
    if not PROPERTY_CACHE_CSV.exists():
        return {}
    try:
        props = pd.read_csv(PROPERTY_CACHE_CSV, dtype=str).fillna("")
        return {str(row["CID"]): row.to_dict() for _, row in props.iterrows()}
    except Exception:
        return {}


def fetch_properties(cids: list[str], property_cache: dict[str, dict]):
    missing = [cid for cid in sorted(set(cids)) if cid not in property_cache or not (property_cache[cid].get("SMILES") or property_cache[cid].get("IsomericSMILES"))]
    if not missing:
        return

    print(f"Retrieving PubChem properties for {len(missing)} unique CID(s)...")
    for start in range(0, len(missing), 50):
        batch = missing[start:start + 50]
        cid_text = ",".join(batch)
        url = f"{PUG}/compound/cid/{cid_text}/property/Title,CanonicalSMILES,IsomericSMILES,InChIKey/JSON"
        payload = get_json(url)
        rows = payload.get("PropertyTable", {}).get("Properties", []) if payload else []

        # If a batch request fails, retry each CID individually rather than silently dropping it.
        if not rows:
            for cid in batch:
                one = get_json(
                    f"{PUG}/compound/cid/{cid}/property/Title,CanonicalSMILES,IsomericSMILES,InChIKey/JSON"
                )
                one_rows = one.get("PropertyTable", {}).get("Properties", []) if one else []
                for item in one_rows:
                    property_cache[str(item.get("CID"))] = item
                time.sleep(0.30)
        else:
            for item in rows:
                property_cache[str(item.get("CID"))] = item
        time.sleep(0.30)

        pd.DataFrame(property_cache.values()).to_csv(PROPERTY_CACHE_CSV, index=False)
        print(f"  Property batches: {min(start + len(batch), len(missing))}/{len(missing)}")


def main():
    print("=" * 68)
    print("FDA DILIrank 2.0 download + PubChem structure mapping")
    print("=" * 68)

    try:
        download_fda_workbook()
        # FDA workbook has a merged title row above the actual field names.
        # Detect the header row rather than assuming row 1 contains column names.
        preview = pd.read_excel(
            FDA_XLSX, sheet_name=0, header=None, engine="openpyxl", nrows=30
        )
    except ImportError:
        raise SystemExit("Missing Excel reader. Run: python -m pip install openpyxl")
    except Exception as exc:
        raise SystemExit(f"Could not read/download the FDA workbook: {exc}")

    def norm_col(value):
        return re.sub(r"[^a-z0-9]", "", str(value).lower())

    required_norm = {"ltkbid", "compoundname", "vdiliconcern"}
    header_row = None
    for row_index, row in preview.iterrows():
        row_values = {norm_col(value) for value in row.tolist() if pd.notna(value)}
        if required_norm.issubset(row_values):
            header_row = int(row_index)
            break

    if header_row is None:
        print("Could not identify the FDA table header in the first 30 rows.")
        print("Workbook preview:")
        print(preview.head(12).to_string(index=False, header=False))
        raise SystemExit(
            "Expected fields LTKBID, CompoundName and vDILI-Concern were not found. "
            "Check that the official DILIrank 2.0 Complete List workbook is being used."
        )

    print(f"Detected FDA table header at Excel row {header_row + 1}.")
    fda = pd.read_excel(
        FDA_XLSX, sheet_name=0, header=header_row, engine="openpyxl"
    )
    fda = fda.dropna(how="all").copy()

    # Normalise column labels while retaining the FDA's original field names.
    rename = {}
    wanted = {
        "ltkbid": "LTKBID",
        "compoundname": "CompoundName",
        "vdiliconcern": "vDILI-Concern",
        "severityclass": "SeverityClass",
        "labelsection": "LabelSection",
        "comment": "Comment",
    }
    for column in fda.columns:
        key = norm_col(column)
        if key in wanted:
            rename[column] = wanted[key]
    fda = fda.rename(columns=rename)

    required = ["LTKBID", "CompoundName", "vDILI-Concern"]
    missing = [col for col in required if col not in fda.columns]
    if missing:
        print("Columns found:", fda.columns.tolist())
        raise SystemExit(f"Required FDA column(s) not found after header detection: {missing}")

    fda["LTKBID"] = fda["LTKBID"].astype(str).str.strip()
    fda["CompoundName"] = fda["CompoundName"].astype(str).str.strip()
    fda["Y"] = pd.array(fda["vDILI-Concern"].map(binary_label), dtype="Int64")
    fda.to_csv(ORIGINAL_CSV, index=False)

    print(f"FDA records: {len(fda)}")
    print("Original category counts:")
    print(fda["vDILI-Concern"].value_counts(dropna=False).to_string())
    print("\nStudy binary mapping: Most/Less=1, No=0, Ambiguous=excluded")

    overrides = load_existing_overrides()
    cache = load_name_cache()
    names = sorted({n for n in fda["CompoundName"].dropna().astype(str) if n and n.lower() != "nan"})
    todo = [n for n in names if n not in cache or cache[n]["lookup_status"] == "api_error"]
    print(f"Unique compound names: {len(names)}; lookups still required: {len(todo)}")

    for index, name in enumerate(todo, start=1):
        cids, status = lookup_name(name)
        cache[name] = {"candidate_cids": ",".join(cids), "lookup_status": status}
        if index % 20 == 0 or index == len(todo):
            save_name_cache(cache)
            print(f"  Name lookups: {index}/{len(todo)}")
        time.sleep(0.30)  # remain below PubChem's usual request-rate ceiling
    save_name_cache(cache)

    # Apply unique name matches first, then any CID explicitly entered by the user.
    row_cids = []
    row_status = []
    all_cids = []
    for _, row in fda.iterrows():
        name = str(row["CompoundName"]).strip()
        lt = str(row["LTKBID"]).strip()
        cached = cache.get(name, {"candidate_cids": "", "lookup_status": "not_found"})
        candidates = [x for x in cached.get("candidate_cids", "").split(",") if x]
        override = overrides.get(lt, "")
        if override:
            cid = override
            status = "manual_override"
        elif len(candidates) == 1:
            cid = candidates[0]
            status = "unique_cid"
        elif len(candidates) > 1:
            cid = ""
            status = "multiple_cids_manual_review"
        else:
            cid = ""
            status = cached.get("lookup_status", "not_found")
        row_cids.append(cid)
        row_status.append(status)
        if cid:
            all_cids.append(cid)

    property_cache = load_property_cache()
    fetch_properties(all_cids, property_cache)

    audit = fda.copy()
    audit["PubChem_Candidate_CIDs"] = [
        cache.get(str(name).strip(), {}).get("candidate_cids", "")
        for name in audit["CompoundName"]
    ]
    audit["PubChem_CID"] = row_cids
    audit["Mapping_Status"] = row_status

    titles, smiles, inchikeys = [], [], []
    for cid in row_cids:
        prop = property_cache.get(str(cid), {}) if cid else {}
        titles.append(prop.get("Title", ""))
        smiles.append(prop.get("SMILES") or prop.get("IsomericSMILES") or "")
        inchikeys.append(prop.get("InChIKey", ""))

    audit["PubChem_Title"] = titles
    audit["Drug"] = smiles
    audit["InChIKey"] = inchikeys
    audit["Drug_ID"] = audit["LTKBID"]
    audit["PubChem_CID"] = audit["PubChem_CID"].astype(str)

    # Mark a unique CID with no returned structure as unresolved.
    missing_structure = audit["PubChem_CID"].ne("") & audit["Drug"].fillna("").eq("")
    audit.loc[missing_structure, "Mapping_Status"] = "property_missing"
    audit.to_csv(AUDIT_CSV, index=False)

    # Model-ready file: unambiguous structure match + non-ambiguous FDA label.
    mapped_status = audit["Mapping_Status"].isin(["unique_cid", "manual_override"])
    usable = audit[mapped_status & audit["Drug"].notna() & audit["Drug"].ne("") & audit["Y"].notna()].copy()
    keep = [
        "Drug_ID", "Drug", "Y", "CompoundName", "vDILI-Concern",
        "PubChem_CID", "PubChem_Title", "InChIKey", "SeverityClass",
        "LabelSection", "Comment"
    ]
    keep = [col for col in keep if col in usable.columns]
    usable[keep].to_csv(MODEL_CSV, index=False)

    # Keep every unresolved structure match in a review file. Preserve user-entered overrides on reruns.
    review_mask = ~audit["Mapping_Status"].isin(["unique_cid", "manual_override"])
    review = audit.loc[review_mask, [
        "LTKBID", "CompoundName", "vDILI-Concern", "Y",
        "PubChem_Candidate_CIDs", "Mapping_Status"
    ]].copy()
    review["PubChem_CID_override"] = review["LTKBID"].map(overrides).fillna("")
    review.to_csv(REVIEW_CSV, index=False)

    print("\n" + "=" * 68)
    print("MAPPING SUMMARY")
    print("=" * 68)
    print("FDA original rows:", len(audit))
    print("Mapping statuses:")
    print(audit["Mapping_Status"].value_counts(dropna=False).to_string())
    print("\nModel-ready rows:", len(usable))
    if len(usable):
        print("Binary label counts:")
        print(usable["Y"].value_counts().sort_index().to_string())
    print("Manual review rows:", len(review))
    print("\nFiles created:")
    for path in [FDA_XLSX, ORIGINAL_CSV, NAME_CACHE_CSV, PROPERTY_CACHE_CSV,
                 AUDIT_CSV, REVIEW_CSV, MODEL_CSV]:
        if path.exists():
            print(f"  {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")

    print("\nIMPORTANT:")
    print("- Review dilirank2_unmapped_review.csv before considering mapping complete.")
    print("- If a name has multiple candidate CIDs, verify the compound identity and enter the chosen CID")
    print("  in PubChem_CID_override, then rerun this script.")
    print("- Do not merge DILIrank into TDC DILI training data. Keep it as external validation.")
    print("- The binary mapping is a study decision; preserve/report the original FDA categories.")


if __name__ == "__main__":
    main()
