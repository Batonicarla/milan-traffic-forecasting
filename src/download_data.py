import getpass
from pathlib import Path
import requests

SERVER = "https://dataverse.harvard.edu"
TELECOM_DOI = "doi:10.7910/DVN/EGZHFV"
GRID_DOI = "doi:10.7910/DVN/QJWLFU"

API_TOKEN = getpass.getpass("Paste your Dataverse API token: ")
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "X-Dataverse-key": API_TOKEN,
})

RAW_DIR = Path("data/raw")
GRID_DIR = Path("data/grid")

def get_file_list(doi):
    url = f"{SERVER}/api/datasets/:persistentId/?persistentId={doi}"
    resp = session.get(url, timeout=60)
    resp.raise_for_status()
    return [(f["dataFile"]["id"], f["dataFile"]["filename"], f["dataFile"]["filesize"])
            for f in resp.json()["data"]["latestVersion"]["files"]]

def download_all(doi, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    files = get_file_list(doi)
    print(f"Found {len(files)} files for {doi}")
    for i, (file_id, filename, filesize) in enumerate(files, 1):
        dest = output_dir / filename
        if dest.exists() and dest.stat().st_size == filesize:
            print(f"[{i}/{len(files)}] {filename} already present, skipping")
            continue
        with session.get(f"{SERVER}/api/access/datafile/{file_id}", stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        print(f"[{i}/{len(files)}] {filename} done")
    print("Finished.")

if __name__ == "__main__":
    download_all(TELECOM_DOI, RAW_DIR)
    download_all(GRID_DOI, GRID_DIR)