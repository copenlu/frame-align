import os, json
import pandas as pd
import requests
from PIL import Image
import numpy as np
import torch, random
import logging
from tqdm import tqdm
import argparse
from urllib.parse import urlparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_images(og_img_dir, directory_name, id_list, image_url_list, image_dir):
    # shuffle urls
    # random.shuffle(image_url_list)


    failed_urls = {}
    failed_ids = {"id": [], "error": []}

    for i in tqdm(range(len(id_list))):
        id = id_list[i]
        image_url = image_url_list[i]
        base_url = urlparse(image_url).netloc

        if base_url in failed_urls and failed_urls[base_url] > 100:
            logger.error(f"Too many failed URLs for {base_url}. Skipping.")
            continue
        # download image using url and save to image_dir/directory_path. Create directory if it doesn't exist
        os.makedirs(os.path.join(image_dir, directory_name), exist_ok=True)
        # image_path = os.path.join(image_dir, directory_name, f"{id}.jpg")
        # download image url to image_path. Add timeout to prevent hanging
        try:
            response = requests.get(image_url, stream=True, timeout=2)  # Add a timeout (in seconds)
            response.raise_for_status()  # Raise an HTTPError if the status is not 200
            raw_image = Image.open(response.raw).convert("RGB")
            
            # Check the shape of the image tensor
            image_tensor = torch.tensor(np.array(raw_image))
            if image_tensor.shape[-1] != 3:
                raise ValueError(f"Unexpected image shape: {image_tensor.shape}")
            if image_tensor.shape[0] == 1 and image_tensor.shape[1] == 1:
                logger.info(f"Skipping image with shape {image_tensor.shape} - id: {id}")
                continue

        except Exception as e:

            if base_url in failed_urls:
                failed_urls[base_url] += 1
            else:
                failed_urls[base_url] = 1

            failed_ids["id"].append(id)
            failed_ids["error"].append(str(e))

            # Save failed ids as a json file
            failed_ids_path = os.path.join(og_img_dir, directory_name, "failed_ids.json")
            json.dump(failed_ids, open(failed_ids_path, "w"))

            # Save failed urls as a json file
            failed_url_path = os.path.join(og_img_dir, directory_name, "failed_urls.json")
            json.dump(failed_urls, open(failed_url_path, "w"))

            logger.info(f"Image URL: {image_url}")
            logger.error(f"Image error {e} - id: {id}")

            continue

        image_path = os.path.join(image_dir, directory_name, f"{id}.jpg")
        raw_image.save(image_path)
        logger.info(f"Saved image to {image_path}")

    return

def filter_urls(base_dir, directory_name):
    directory_path = os.path.join(base_dir, directory_name)
    # csv_file = "datawithtopics_merged.csv"
    csv_file = "undownloaded_uuids.csv"
    logging.info(f"Reading {csv_file} from {directory_path}")
    df = pd.read_csv(os.path.join(directory_path, csv_file))

    # shuffle them
    df = df.sample(frac=1).reset_index(drop=True)

    print(df.columns)
    # get 'id' and 'image_url' columns and drop rows with missing image_url
    df_image = df[['id', 'image_url']].dropna(subset=['image_url']) 

    # df_image = df_image[0:10]

    id_list = df_image['id'].tolist()
    image_url_list = df_image['image_url'].tolist()

    return df, id_list, image_url_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download images from URLs.")
    # base_dir = "/home/vsl333/datasets/news-bert-data/bertopic/allcsvtopics"
    parser.add_argument("base_dir", type=str, help="The base directory to save images.")
    parser.add_argument("directory_name", type=str, help="The directory name to save images.")
    # image_dir = "/projects/belongielab/data/frame-align"
    parser.add_argument("image_dir", type=str, help="The directory to save images.")
    # base_dir = "/home/vsl333/datasets/news-bert-data/bertopic/allcsvtopics"
    args = parser.parse_args()

    # og_img_dir = the bath where "undownloaded_uuids.csv" is located
    og_img_dir = args.base_dir + "/img_data"

    data_df, id_list, image_url_list = filter_urls(og_img_dir, args.directory_name)
    
    logging.info(f"Downloading images for {args.directory_name}")
    download_images(og_img_dir, args.directory_name, id_list, image_url_list, args.image_dir)
    logging.info(f"Download complete for {args.directory_name}")