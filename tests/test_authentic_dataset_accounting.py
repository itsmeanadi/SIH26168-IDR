"""Unit and forensic validation tests for authentic IO-VNBD dataset accounting and driver disjointness."""

import json
import zipfile
import pytest
from pathlib import Path
import pandas as pd
import numpy as np
from idr.config import BASE_DIR, CONFIG, DRIVER_GROUP_MAP, DatasetConfig

RAW_ZIP = BASE_DIR / "data" / "raw" / "Synchronised_V_and_S_datasets.zip"
FINGERPRINT_JSON = BASE_DIR / "data" / "raw" / "dataset_fingerprint.json"


def test_fingerprint_file_exists_and_consistent():
    """Verify dataset fingerprint JSON exists and has all required forensic keys."""
    assert FINGERPRINT_JSON.exists(), f"Fingerprint missing: {FINGERPRINT_JSON}"
    with open(FINGERPRINT_JSON, "r") as f:
        data = json.load(f)

    assert data["archive_sha256"] == "624003b0bfb3d221114eb262dd02f21f7dba74fb25d8045b4b8ac684956d2855"
    assert data["archive_byte_size"] == 203606286
    assert data["zip_entry_count"] == 442
    assert data["actual_file_count"] == 360
    assert data["directory_count"] == 82
    assert data["total_uncompressed_bytes"] == 864108895
    assert data["csv_file_count_total"] == 288
    assert data["csv_categorised_count"] == 144
    assert data["csv_uncategorised_count"] == 144
    assert data["jpg_file_count"] == 72
    assert data["categorised_drive_pair_count"] == 72
    assert data["phone_csv_count_categorised"] == 72
    assert data["vehicle_csv_count_categorised"] == 72
    assert data["orphan_files_count"] == 0
    assert data["duplicate_pair_count"] == 0
    assert data["total_samples_all_drives"] == 1070745


def test_driver_disjoint_split_policy():
    """Verify that train, val, and test splits have zero driver overlap."""
    cfg = CONFIG["dataset"]
    assert cfg.split_policy == "driver_disjoint"

    def get_drivers(groups):
        drivers = set()
        for g in groups:
            for d_name, d_groups in DRIVER_GROUP_MAP.items():
                if any(g.startswith(dg) or dg.startswith(g) for dg in d_groups):
                    drivers.add(d_name)
        return drivers

    train_drivers = get_drivers(cfg.train_drives)
    val_drivers = get_drivers(cfg.val_drives)
    test_drivers = get_drivers(cfg.test_drives)

    # Assert driver sets are strictly disjoint
    assert len(train_drivers & val_drivers) == 0, f"Train and Val share drivers: {train_drivers & val_drivers}"
    assert len(train_drivers & test_drivers) == 0, f"Train and Test share drivers: {train_drivers & test_drivers}"
    assert len(val_drivers & test_drivers) == 0, f"Val and Test share drivers: {val_drivers & test_drivers}"


@pytest.mark.skipif(not RAW_ZIP.exists(), reason="Authentic raw zip not present locally")
def test_archive_inventory_and_drive_pairing():
    """Verify exact ZIP file counts and 100% drive pairing without full extraction."""
    with zipfile.ZipFile(RAW_ZIP, "r") as z:
        infolist = z.infolist()
        assert len(infolist) == 442

        cat_files = [f for f in z.namelist() if f.startswith("Synchronised V abd S datasets/Categorised IOVNB Dataset/") and not f.endswith("/")]
        cat_csvs = [f for f in cat_files if f.endswith(".csv")]
        assert len(cat_csvs) == 144

        # Verify basenames match
        phone_ids = set()
        vehicle_ids = set()
        for f in cat_csvs:
            basename = Path(f).name
            if basename.startswith("S-") or basename.startswith("S_"):
                phone_ids.add(basename[2:-4].lower())
            elif basename.lower().startswith("v-") or basename.lower().startswith("v_"):
                vehicle_ids.add(basename[2:-4].lower())

        assert len(phone_ids) == 72
        assert len(vehicle_ids) == 72
        assert phone_ids == vehicle_ids, f"Mismatch in paired drive IDs: {phone_ids ^ vehicle_ids}"
