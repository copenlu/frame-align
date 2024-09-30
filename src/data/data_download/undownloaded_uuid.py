from tqdm import tqdm
from pathlib import Path
import shutil, logging, pickle
import pandas as pd

all_months = [
    "2023-05-01_2023-05-31", "2023-06-01_2023-06-30","2023-07-01_2023-07-31", 
    "2023-08-01_2023-08-31", "2023-09-01_2023-09-30", "2023-10-01_2023-10-31", 
    "2023-11-01_2023-11-30", "2023-12-01_2023-12-31", "2024-01-01_2024-01-31", 
    "2024-02-01_2024-02-29", "2024-03-01_2024-03-31", "2024-04-01_2024-04-30"
]

# all_months = ["2023-06-01_2023-06-30"]


base_img_dir = Path("/projects/frame_align/data/img_data/")
base_raw_dir = Path("/projects/frame_align/data/raw/2023-2024/")

month_imgdir_paths = [base_img_dir / month for month in all_months]
month_rawdir_paths = [base_raw_dir / month for month in all_months]

sorted_month_imgdir_paths = sorted(month_imgdir_paths)
sorted_month_rawdir_paths = sorted(month_rawdir_paths)

logging.basicConfig(level=logging.INFO)
logging.info("month_dir_paths")

downloaded_uuids = {}
# Load pikle file with downloaded uuids and save to a list
for idx, month_imgdir in enumerate(sorted_month_imgdir_paths):
    with open(month_imgdir / f"{month_imgdir.name}_downloaded_uuids.pkl", "rb") as f:
        downloaded_uuids[month_imgdir.name] = pickle.load(f)
        

undownloaded_uuids = {}

# use tqdm to show progress bar
for img_monthy_dir, raw_monthy_dir in zip(tqdm(sorted_month_imgdir_paths), sorted_month_rawdir_paths):
    csv_file_path = raw_monthy_dir / "datawithtopiclabels.csv" 
     
    undownloaded_csv_path = img_monthy_dir / "undownloaded_uuids.csv"
    # delete the file if it exists
    if undownloaded_csv_path.exists():
        print(f"Existed! Deleting {undownloaded_csv_path}")
        undownloaded_csv_path.unlink()

    # read the csv file in chunks to avoid memory error if needed
    csv_df = pd.read_csv(csv_file_path)
    csv_df = csv_df.dropna(subset=['image_url'])
    total_urls = csv_df['image_url'].tolist()
    
    # check if id is in downloaded_uuids
    filtered_csv = csv_df[~csv_df['id'].isin(downloaded_uuids[raw_monthy_dir.name])]
    filtered_csv.to_csv(undownloaded_csv_path, index=False, header=True)
    print(f"Total urls: {len(total_urls)}, Undownloaded urls: {len(filtered_csv)}")
    
    # read the undownloaded csv file and count the number of rows
    undownloaded_csv_path = img_monthy_dir / "undownloaded_uuids.csv"
    undownloaded_df = pd.read_csv(undownloaded_csv_path)
    logging.info(f" Downloaded {raw_monthy_dir.name}: {len(downloaded_uuids[raw_monthy_dir.name])}")
    logging.info(f" Undownloaded {raw_monthy_dir.name}: {len(undownloaded_df)}")
