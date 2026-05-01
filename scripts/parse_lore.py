import sqlite3, json, pathlib, os

DB_PATH = next(pathlib.Path("data/lore_raw").glob("*.content"))
OUT_PATH = pathlib.Path("data/lore_entries.json")

def parse_lore():
    os.makedirs(OUT_PATH.parent, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # The table name for lore definitions
    cursor.execute("SELECT json FROM DestinyLoreDefinition")
    rows = cursor.fetchall()
    conn.close()
    
    entries = []
    for (raw,) in rows:
        data = json.loads(raw)
        title = data.get("displayProperties", {}).get("name", "")
        body  = data.get("displayProperties", {}).get("description", "")
        
        # Skip empty or very short entries
        if len(body.strip()) < 50:
            continue
        
        entries.append({
            "title": title,
            "body": body.strip(),
            "hash": data.get("hash"),
        })
    
    OUT_PATH.write_text(json.dumps(entries, indent=2))
    print(f"Parsed {len(entries)} lore entries → {OUT_PATH}")

if __name__ == "__main__":
    parse_lore()