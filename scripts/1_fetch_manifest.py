#!/usr/bin/env python3
"""Fetch the Destiny 2 manifest and export lore entries to CSV."""

from __future__ import annotations

import json
import os
import sqlite3
import zipfile
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import requests
from dotenv import load_dotenv

BUNGIE_MANIFEST_URL = "https://www.bungie.net/Platform/Destiny2/Manifest/"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MANIFEST_ZIP_PATH = DATA_DIR / "manifest.zip"
LORE_CSV_PATH = DATA_DIR / "lore_raw.csv"


def ensure_data_dir() -> None:
    """Ensure the target data directory exists."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_api_key() -> str:
    """Load the Bungie API key from environment variables."""
    load_dotenv()
    api_key = os.getenv("BUNGIE_API_KEY")
    if not api_key:
        raise EnvironmentError("BUNGIE_API_KEY is not set in the environment.")
    return api_key


def fetch_manifest_metadata(api_key: str) -> Dict[str, Any]:
    """Fetch manifest metadata from Bungie's API."""
    response = requests.get(
        BUNGIE_MANIFEST_URL,
        headers={"X-API-Key": api_key},
        timeout=30,
    )
    response.raise_for_status()
    payload: Dict[str, Any] = response.json()
    if "Response" not in payload:
        raise ValueError("Unexpected response payload: missing 'Response' key.")
    return payload["Response"]


def get_world_content_url(metadata: Dict[str, Any]) -> str:
    """Extract the English world content URL from manifest metadata."""
    try:
        relative_path = metadata["mobileWorldContentPaths"]["en"]
    except KeyError as exc:
        raise KeyError("English world content path not found in manifest metadata.") from exc
    return f"https://www.bungie.net{relative_path}"


def download_manifest(url: str, api_key: str) -> Path:
    """Download the manifest zip file to disk."""
    with requests.get(url, headers={"X-API-Key": api_key}, timeout=60, stream=True) as resp:
        resp.raise_for_status()
        with MANIFEST_ZIP_PATH.open("wb") as handle:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)
    return MANIFEST_ZIP_PATH


def extract_sqlite(zip_path: Path) -> Path:
    """Extract the SQLite database from the manifest zip."""
    with zipfile.ZipFile(zip_path, "r") as archive:
        members = archive.namelist()
        if not members:
            raise ValueError("Manifest archive is empty.")
        sqlite_member = next((m for m in members if m.endswith(".sqlite")), None)
        if not sqlite_member:
            raise ValueError("No SQLite file found in manifest archive.")
        extracted_path = DATA_DIR / Path(sqlite_member).name
        archive.extract(sqlite_member, path=DATA_DIR)
    return extracted_path


def export_lore_to_csv(sqlite_path: Path) -> None:
    """Export the DestinyLoreDefinition table to CSV."""
    with sqlite3.connect(sqlite_path) as conn:
        lore_df = pd.read_sql_query("SELECT * FROM DestinyLoreDefinition", conn)
    lore_df.to_csv(LORE_CSV_PATH, index=False)


def main() -> None:
    """Run the manifest fetch and export pipeline."""
    ensure_data_dir()
    api_key = load_api_key()
    metadata = fetch_manifest_metadata(api_key)
    world_content_url = get_world_content_url(metadata)

    print(f"Downloading manifest from {world_content_url}")
    zip_path = download_manifest(world_content_url, api_key)

    print(f"Extracting manifest to {DATA_DIR}")
    sqlite_path = extract_sqlite(zip_path)

    print(f"Exporting DestinyLoreDefinition to {LORE_CSV_PATH}")
    export_lore_to_csv(sqlite_path)

    print("Done.")


if __name__ == "__main__":
    main()

