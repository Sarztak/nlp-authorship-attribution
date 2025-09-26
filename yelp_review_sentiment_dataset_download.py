import kagglehub
import os 
import shutil
from pathlib import Path 


path = kagglehub.dataset_download("thedevastator/yelp-reviews-sentiment-dataset")

print("Path to dataset files:", path)



path = Path(path)
drive_path = Path('/content/drive/MyDrive/yelp-restaurant-reviews-sentiments')
drive_path.mkdir(exist_ok=True)

for file in os.listdir(path):
    shutil.copy(path / file, drive_path)

