"""NHTSA flat files: download, parse and load into a DuckDB database.

Two public files from https://static.nhtsa.gov/odi/ffdd/ are used:

* ``FLAT_RCL_POST_2010.zip``: recall campaigns reported since 2010, one row per (campaign, vehicle, component).
* ``COMPLAINTS_RECEIVED_2025-2026.zip``: owner complaints received since 2025-01-01, one row per
  (complaint, product, component).

Only vehicle records are kept (recall type ``V``, complaint product type ``V``). Personal or
free-text fields that the agent does not need (city, partial VIN, dealer, operator name,
complaint narrative) are never loaded.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import duckdb
import requests

BASE_URL = "https://static.nhtsa.gov/odi/ffdd"
RECALLS_FILE = "rcl/FLAT_RCL_POST_2010.zip"
COMPLAINTS_FILE = "cmpl/COMPLAINTS_RECEIVED_2025-2026.zip"

RCL_COLUMNS = [
    "RECORD_ID", "CAMPNO", "MAKETXT", "MODELTXT", "YEARTXT", "MFGCAMPNO", "COMPNAME", "MFGNAME", "BGMAN", "ENDMAN",
    "RCLTYPECD", "POTAFF", "ODATE", "INFLUENCED_BY", "MFGTXT", "RCDATE", "DATEA", "RPNO", "FMVSS", "DESC_DEFECT",
    "CONEQUENCE_DEFECT", "CORRECTIVE_ACTION", "NOTES", "RCL_CMPT_ID", "MFR_COMP_NAME", "MFR_COMP_DESC",
    "MFR_COMP_PTNO", "DO_NOT_DRIVE", "PARK_OUTSIDE",
]
CMPL_COLUMNS = [
    "CMPLID", "ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "CRASH", "FAILDATE", "FIRE", "INJURED", "DEATHS",
    "COMPDESC", "CITY", "STATE", "VIN", "DATEA", "LDATE", "MILES", "OCCURENCES", "CDESCR", "CMPL_TYPE",
    "POLICE_RPT_YN", "PURCH_DT", "ORIG_OWNER_YN", "ANTI_BRAKES_YN", "CRUISE_CONT_YN", "NUM_CYLS", "DRIVE_TRAIN",
    "FUEL_SYS", "FUEL_TYPE", "TRANS_TYPE", "VEH_SPEED", "DOT", "TIRE_SIZE", "LOC_OF_TIRE", "TIRE_FAIL_TYPE",
    "ORIG_EQUIP_YN", "MANUF_DT", "SEAT_TYPE", "RESTRAINT_TYPE", "DEALER_NAME", "DEALER_TEL", "DEALER_CITY",
    "DEALER_STATE", "DEALER_ZIP", "PROD_TYPE", "REPAIRED_YN", "MEDICAL_ATTN", "VEHICLES_TOWED_YN",
    "STATE_OF_INCIDENT", "VEHICLE_OPERATOR",
]

# NHTSA top-level component names collapsed into families. Complaint and recall taxonomies differ slightly
# (e.g. "ENGINE" vs "ENGINE AND ENGINE COOLING", "SERVICE BRAKES" vs "SERVICE BRAKES, HYDRAULIC"); a component
# belongs to the longest family name it starts with. The gloss is shown to the LLM planner.
COMPONENT_FAMILIES: dict[str, str] = {
    "AIR BAGS": "airbags, inflators, occupant classification",
    "BACK OVER PREVENTION": "rearview camera, backup display, rear sensors",
    "CHILD SEAT": "built-in or add-on child restraints",
    "COMMUNICATION": "telematics, eCall",
    "ELECTRICAL SYSTEM": "battery, wiring, instrument cluster, software, displays",
    "ELECTRONIC STABILITY CONTROL": "ESC",
    "ENGINE": "engine and engine cooling",
    "EQUIPMENT": "accessories and other equipment",
    "EXTERIOR LIGHTING": "headlights, taillights, turn signals",
    "FORWARD COLLISION AVOIDANCE": "automatic emergency braking, collision warning",
    "FUEL SYSTEM": "fuel tank, pump, lines, fuel leaks",
    "FUEL/PROPULSION SYSTEM": "EV or hybrid propulsion and charging (complaint taxonomy)",
    "HYBRID PROPULSION SYSTEM": "hybrid or EV high-voltage system",
    "INTERIOR LIGHTING": "interior lights",
    "LANE DEPARTURE": "lane-keeping assist, lane departure warning",
    "LATCHES/LOCKS/LINKAGES": "door, hood and tailgate latches and locks",
    "PARKING BRAKE": "parking brake",
    "POWER TRAIN": "transmission, drivetrain, axles, shifting",
    "SEAT BELTS": "seat belts and pretensioners",
    "SEATS": "seats and seat frames",
    "SERVICE BRAKES": "brakes (hydraulic, air or electric)",
    "STEERING": "steering",
    "STRUCTURE": "body, frame, hood, doors, bumpers",
    "SUSPENSION": "suspension",
    "TIRES": "tires",
    "TRACTION CONTROL SYSTEM": "traction control",
    "TRAILER HITCHES": "trailer hitches",
    "UNKNOWN OR OTHER": "unknown or other",
    "VEHICLE SPEED CONTROL": "cruise control, accelerator, unintended acceleration",
    "VISIBILITY": "windshield, wipers, mirrors, windows, defroster",
    "WHEELS": "wheels and wheel nuts",
}
OTHER_FAMILY = "UNKNOWN OR OTHER"


def component_family(component: str) -> str:
    """Map an NHTSA component description (``TOP:SUB:...``) to its family."""
    top = component.split(":", 1)[0].strip().upper()
    matches = [f for f in COMPONENT_FAMILIES if top.startswith(f)]
    return max(matches, key=len) if matches else OTHER_FAMILY


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(relative: str, raw_dir: Path, timeout: int = 600) -> Path:
    """Download one NHTSA zip into ``raw_dir`` (skipped when present) and extract its .txt file."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / Path(relative).name
    if not zip_path.exists():
        tmp = zip_path.with_suffix(".part")
        with requests.get(f"{BASE_URL}/{relative}", stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        tmp.replace(zip_path)
    with zipfile.ZipFile(zip_path) as z:
        (name,) = [n for n in z.namelist() if n.lower().endswith(".txt")]
        txt = raw_dir / name
        if not txt.exists() or txt.stat().st_size != z.getinfo(name).file_size:
            z.extract(name, raw_dir)
    return txt


def manifest(raw_dir: Path) -> dict:
    """Size, sha256 and inner-file timestamp of each downloaded zip (cached in raw_dir/manifest.json)."""
    path = raw_dir / "manifest.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    out = {}
    for rel in (RECALLS_FILE, COMPLAINTS_FILE):
        zp = raw_dir / Path(rel).name
        key = f"{zp.name}:{zp.stat().st_size}:{int(zp.stat().st_mtime)}"
        if old.get(zp.name, {}).get("key") == key:
            out[zp.name] = old[zp.name]
            continue
        with zipfile.ZipFile(zp) as z:
            info = z.infolist()[0]
        out[zp.name] = {"key": key, "url": f"{BASE_URL}/{rel}", "bytes": zp.stat().st_size, "sha256": sha256(zp),
                        "inner_file": info.filename, "inner_timestamp": "%04d-%02d-%02d %02d:%02d" % info.date_time[:5]}
    path.write_text(json.dumps(out, indent=2))
    return out


def _read_tsv(con: duckdb.DuckDBPyConnection, path: Path, columns: list[str], table: str) -> None:
    cols = "{" + ", ".join(f"'{c}': 'VARCHAR'" for c in columns) + "}"
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE {table} AS SELECT * FROM read_csv('{path.as_posix()}', delim='\t', header=false, "
        f"quote='', escape='', columns={cols}, strict_mode=false, encoding='utf-8')"
    )


def _family_table(con: duckdb.DuckDBPyConnection, source: str, column: str, name: str) -> None:
    names = [r[0] for r in con.execute(f"SELECT DISTINCT {column} FROM {source} WHERE {column} IS NOT NULL").fetchall()]
    con.execute(f"CREATE OR REPLACE TEMP TABLE {name} (component VARCHAR, family VARCHAR)")
    con.executemany(f"INSERT INTO {name} VALUES (?, ?)", [(n, component_family(n)) for n in names])


def build_database(recalls_txt: Path, complaints_txt: Path, db_path: Path) -> dict[str, int]:
    """Create the five analysis tables in ``db_path``; returns row counts."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".building")
    tmp.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))
    con.execute("SET threads TO 2")

    _read_tsv(con, recalls_txt, RCL_COLUMNS, "raw_rcl")
    con.execute("CREATE OR REPLACE TEMP TABLE rcl AS SELECT * FROM raw_rcl WHERE RCLTYPECD = 'V'")
    con.execute(
        """
        CREATE TABLE recalls AS
        SELECT CAMPNO AS campaign,
               any_value(trim(MFGNAME)) AS manufacturer,
               min(strptime(RCDATE, '%Y%m%d'))::DATE AS report_date,
               max(TRY_CAST(POTAFF AS BIGINT)) AS potential_units,
               any_value(INFLUENCED_BY) AS initiated_by,
               bool_or(DO_NOT_DRIVE = 'Yes') AS do_not_drive,
               bool_or(PARK_OUTSIDE = 'Yes') AS park_outside,
               any_value(DESC_DEFECT) AS defect,
               any_value(CONEQUENCE_DEFECT) AS consequence,
               any_value(CORRECTIVE_ACTION) AS remedy
        FROM rcl GROUP BY CAMPNO ORDER BY CAMPNO
        """
    )
    con.execute(
        """
        CREATE TABLE recall_vehicles AS
        SELECT DISTINCT CAMPNO AS campaign, upper(trim(MAKETXT)) AS make, upper(trim(MODELTXT)) AS model,
               TRY_CAST(YEARTXT AS INTEGER) AS model_year
        FROM rcl WHERE TRY_CAST(YEARTXT AS INTEGER) BETWEEN 1900 AND 2100
        ORDER BY 1, 2, 3, 4
        """
    )
    _family_table(con, "rcl", "COMPNAME", "rcl_fam")
    con.execute(
        """
        CREATE TABLE recall_components AS
        SELECT DISTINCT r.CAMPNO AS campaign, r.COMPNAME AS component, f.family
        FROM rcl r JOIN rcl_fam f ON f.component = r.COMPNAME ORDER BY 1, 2
        """
    )

    _read_tsv(con, complaints_txt, CMPL_COLUMNS, "raw_cmpl")
    con.execute("CREATE OR REPLACE TEMP TABLE cmpl AS SELECT * FROM raw_cmpl WHERE PROD_TYPE = 'V'")
    # One row per (complaint, vehicle). Complaint-level flags are constant across a complaint's rows.
    con.execute(
        """
        CREATE TABLE complaints AS
        WITH per_complaint AS (
            SELECT ODINO,
                   min(strptime(LDATE, '%Y%m%d'))::DATE AS received,
                   min(TRY_STRPTIME(FAILDATE, '%Y%m%d'))::DATE AS incident_date,
                   bool_or(CRASH = 'Y') AS crash, bool_or(FIRE = 'Y') AS fire,
                   max(TRY_CAST(INJURED AS INTEGER)) AS injured, max(TRY_CAST(DEATHS AS INTEGER)) AS deaths,
                   bool_or(MEDICAL_ATTN = 'Y') AS medical_attention, bool_or(VEHICLES_TOWED_YN = 'Y') AS towed,
                   any_value(nullif(trim(STATE), '')) AS state, max(TRY_CAST(MILES AS INTEGER)) AS miles
            FROM cmpl GROUP BY ODINO
        )
        SELECT DISTINCT CAST(c.ODINO AS BIGINT) AS odino, upper(trim(c.MAKETXT)) AS make, upper(trim(c.MODELTXT)) AS model,
               TRY_CAST(c.YEARTXT AS INTEGER) AS model_year, p.received, p.incident_date, p.crash, p.fire,
               coalesce(p.injured, 0) AS injured, coalesce(p.deaths, 0) AS deaths, p.medical_attention, p.towed,
               p.state, p.miles
        FROM cmpl c JOIN per_complaint p USING (ODINO)
        ORDER BY 1, 2, 3, 4
        """
    )
    _family_table(con, "cmpl", "COMPDESC", "cmpl_fam")
    con.execute(
        """
        CREATE TABLE complaint_components AS
        SELECT DISTINCT CAST(c.ODINO AS BIGINT) AS odino, c.COMPDESC AS component, f.family
        FROM cmpl c JOIN cmpl_fam f ON f.component = c.COMPDESC ORDER BY 1, 2
        """
    )
    counts = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
              for t in ("recalls", "recall_vehicles", "recall_components", "complaints", "complaint_components")}
    con.close()
    db_path.unlink(missing_ok=True)
    tmp.replace(db_path)
    return counts


def coverage(con: duckdb.DuckDBPyConnection) -> dict:
    """Date ranges and sizes of the loaded tables (used in prompts, README and run metadata)."""
    c = con.execute("SELECT min(received), max(received), count(DISTINCT odino) FROM complaints").fetchone()
    r = con.execute("SELECT min(report_date), max(report_date), count(*) FROM recalls").fetchone()
    return {
        "complaints_from": str(c[0]), "complaints_to": str(c[1]), "complaints": int(c[2]),
        "recalls_from": str(r[0]), "recalls_to": str(r[1]), "recall_campaigns": int(r[2]),
        "makes": int(con.execute("SELECT count(DISTINCT make) FROM (SELECT make FROM complaints UNION SELECT make FROM recall_vehicles)").fetchone()[0]),
    }
