import zipfile
from pathlib import Path

top_zip = Path("./pan12-authorship-attribution-corpora.zip")
out_dir  = Path("pan12-authorship-attribution-corpora")

# 1) Extract the top-level archive
out_dir.mkdir(exist_ok=True)
with zipfile.ZipFile(top_zip, "r") as zf:
    zf.extractall(out_dir)

# 2) Recursively extract any nested .zip files
def extract_nested_zips(root: Path, delete_archives=True):
    # find all .zip files under root, recursively
    for zippath in list(root.rglob("*.zip")):
        print(zippath)
        dest = zippath.with_suffix("")  # folder named after the zip
        dest.mkdir(exist_ok=True)
        with zipfile.ZipFile(zippath, "r") as zf:
            zf.extractall(dest)
        if delete_archives:
            zippath.unlink()  
        extract_nested_zips(dest, delete_archives=delete_archives)

extract_nested_zips(out_dir)
