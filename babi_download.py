# need datasets < 0.0.4
from datasets import load_dataset, load_from_disk
from pathlib import Path 

ds = load_dataset("facebook/babi_qa", "en-10k-qa1")
babi_path = Path("/content/drive/MyDrive/babi_data") 
babi_path.mkdir(exist_ok=True)
ds.save_to_disk(babi_path)

# ds = load_from_disk(babi_path)