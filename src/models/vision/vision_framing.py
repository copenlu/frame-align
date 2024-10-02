import json, pickle
import torch, os
import random, shutil
import argparse
import numpy as np
import pandas as pd
import logging
import requests
from PIL import Image
from pathlib import Path
from pdb import set_trace
from vllm import LLM, SamplingParams
from prompts_llava import PROMPT_DICT_LLAVA

random.seed(42)
torch.manual_seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

script_dir = Path(__file__).resolve().parent    
logger.info(f"Script directory: {script_dir}")

data="new_data"

if data=="old_data":
    og_img_path = f"/projects/frame_align/data/img_data/"
if data=="new_data":
    og_img_path = f"/projects/frame_align/data/news_img_data/"

logger.info(f"Image path: {og_img_path}")
data_csv_path = f"/projects/frame_align/data/raw/2023-2024/"

PROMPT_MAPPING = {
        "llava-hf/llava-1.5-7b-hf": PROMPT_DICT_LLAVA
        }

sampling_params = SamplingParams(temperature=0.2, max_tokens=1000)

def annotate_frames(model_code, dir_name)-> None:

    model_name_short = model_code.split('/')[1].split('-')[0]

    logging.info(f"Directory name: {dir_name}")
    # type of dir name
    logging.info(f"Type of dir name: {type(dir_name)}")

    if data=="old_data":
        pkl_path = os.path.join(og_img_path, dir_name, f"{dir_name}_downloaded_uuids.pkl")
    if data=="new_data":
        pkl_path = os.path.join(og_img_path, f"{dir_name}_downloaded_uuids.pkl")

    logging.info(f"Loading pickle file from path: {pkl_path}")
    
    data_csv = os.path.join(data_csv_path, dir_name, "datawithtopiclabels.csv")

    img_dir_path = os.path.join(og_img_path, dir_name)

    unfound_images = []
    # load pickle file
    with open(pkl_path, "rb") as f:
        downloaded_uuids = pickle.load(f)

    logger.info(f"Downloaded UUIDs: {len(downloaded_uuids)}")
    logger.info(f"Loading CSV from path: {data_csv}")
    df = pd.read_csv(data_csv)
    # only choose the one in downloaded uuids
    data_df = df[df["id"].isin(downloaded_uuids)]

    
    logger.info(f"Will rundata for downloaded images, DF shape: {data_df.shape}")

    if data=="old_data":
        pass
        # output_file = os.path.join(og_img_path, dir_name , "annotated/vision/", f"{dir_name}_{model_name_short}.jsonl")
    if data=="new_data":
        output_file = os.path.join(f"/projects/frame_align/data/annotated/vision/", f"{dir_name}_{model_name_short}.jsonl")
        output_fail_file = os.path.join(f"/projects/frame_align/data/annotated/vision/", f"{dir_name}_{model_name_short}_fail.tsv")
    # delete the file if it exists
    if os.path.exists(output_file):
        logging.info(f"Existed! Deleting existing file: {output_file}")
        os.remove(output_file)


    vlm = LLM(model=model_code)

    ids, image_urls, headlines = data_df["id"].tolist(), data_df["image_url"].tolist(), data_df["title"].tolist()
    logging.info(f"Number of images to process: {len(ids)}")

    model_prompt_dict = PROMPT_MAPPING[model_code]

    # ADD tqdm here to see progress
    for uuid, image_file, headline in zip(ids, image_urls, headlines):

        # load image from img_dir_path + uuid + ".jpg"
        image_file_name = os.path.join(img_dir_path, f"{uuid}.jpg")
        if os.path.exists(image_file_name):
            raw_image = Image.open(image_file_name).convert("RGB")
        else:
            unfound_images.append(uuid)
            logger.info(f"Image file not found: {image_file_name}")
            continue

        img_annotations = {}
        # try:
        #     logger.info(f"Opening image")
        #     response = requests.get(image_file, stream=True, timeout=20)  # Add a timeout (in seconds)
        #     response.raise_for_status()  # Raise an HTTPError if the status is not 200
        #     raw_image = Image.open(response.raw).convert("RGB")
        #     logger.info(f"Image opened")
            
        #     # Check the shape of the image tensor
        #     image_tensor = torch.tensor(np.array(raw_image))
        #     if image_tensor.shape[-1] != 3:
        #         raise ValueError(f"Unexpected image shape: {image_tensor.shape}")
        #     if image_tensor.shape[0] == 1 and image_tensor.shape[1] == 1:
        #         logger.info(f"Skipping image with shape {image_tensor.shape} - uuid: {uuid}")
        #         continue
        #     del image_tensor

        # except Exception as e:
        #     logger.info(f"Image URL: {image_file}")
        #     logger.error(f"Image error {e} - uuid: {uuid}")
        #     continue

        logger.info(f"Processing uuid: {uuid}")
        logger.info(f"Image URL: {image_file}")

        for task, prompt in model_prompt_dict.items(): 
            # Inference with image embeddings as input
            inputs = {
                "prompt": prompt,
                "multi_modal_data": {"image": raw_image}
            }
            outputs = vlm.generate(inputs, sampling_params=sampling_params)
            output_text = outputs[0].outputs[0].text
            try:
                output_json = json.loads(output_text)
                img_annotations.update(output_json)
            except Exception as e:
                try:
                    output_json = json.loads(output_text[output_text.index('{'):output_text.rindex('}')+1])
                    img_annotations.update(output_json)
                except Exception as e:
                    print(f"Skipped-uuid-{uuid}-{task}: {e}")
                    with open(output_fail_file, "a") as f:
                        f.write(f"{uuid}\t{task}\t{output_text}\n")
                    continue
        img_annotations["image_url"] = image_file
        img_annotations["title"] = headline
        img_annotations["uuid"] = uuid

        # logging location of current directory
        logger.info(f"context: {Path.cwd()}")
        
  
        # create the directory if it does not exist else write to the file
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "a") as f:
            json.dump(img_annotations, f)
            f.write("\n")

    # count number of lines in the jsonl file
    with open(output_file, "r") as f:
        lines = f.readlines()
        logger.info(f"Number of lines in the jsonl file: {len(lines)}")

    logger.info(f"Unfound images: {unfound_images}")



def main():
    parser = argparse.ArgumentParser(description='Annotate image frames using a VLLM model')
    parser.add_argument('--model_name', type=str, help='Model name', default='llava-hf/llava-1.5-7b-hf')
    parser.add_argument('--data_file', type=str, help='Data file with image urls', default="/projects/frame_align/data/raw/2023-24/default/datawithtopiclabels.csv")
    parser.add_argument('--dir_name', type=str, help='Directory name for saving annotated frames', default="default")
    args = parser.parse_args()

    dir_name = args.dir_name.split("/")[-1]
    # annotate_frames(args.model_name, args.data_file, args.dir_name)
    annotate_frames(args.model_name, dir_name)

if __name__ == "__main__":
    main()
    print("Done")
