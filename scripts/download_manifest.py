import requests, zipfile, json, os, pathlib
from dotenv import load_dotenv

load_dotenv()  # reads .env into environment variables

BUNGIE_API_KEY = os.getenv("BUNGIE_API_KEY")


BASE = "https://www.bungie.net"
MANIFEST_URL = f"{BASE}/Platform/Destiny2/Manifest/"

def download_manifest():
    os.makedirs("data/lore_raw", exist_ok=True)
    resp = requests.get(MANIFEST_URL, headers={"X-API-Key": BUNGIE_API_KEY})
    manifest = resp.json()["Response"]
    
    # Get the SQLite DB path (contains all definitions)
    db_path = manifest["mobileWorldContentPaths"]["en"]
    db_url = BASE + db_path
    
    print(f"Downloading content DB from {db_url}...")
    r = requests.get(db_url)
    zip_path = "data/content.zip"
    
    with open(zip_path, "wb") as f:
        f.write(r.content)
    
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall("data/lore_raw/")
    
    print("Done. Check data/lore_raw/ for the .content file.")

if __name__ == "__main__":
    download_manifest()