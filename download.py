from tqdm import tqdm
import requests


pan12_url = "https://zenodo.org/api/records/3713273/files-archive"

with requests.get(pan12_url, stream=True) as r:
    r.raise_for_status()
    total = int(r.headers.get('content-length', 0))
    with open("pan12-authorship-attribution-corpora.zip", "wb") as f, tqdm(total=total if total > 0 else None, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))