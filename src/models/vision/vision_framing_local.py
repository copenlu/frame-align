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
from prompts_pixtral import PROMPT_DICT_PIXTRAL
from prompts_pixtral_small_prompts import PROMPT_DICT_PIXTRAL_SMALL
from prompts_pixtral_medium_prompts import PROMPT_DICT_PIXTRAL_MEDIUM
from prompts_pixtral_underrevision import PROMPT_DICT_PIXTRAL_REVISION

import base64

random.seed(42)
torch.manual_seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

script_dir = Path(__file__).resolve().parent    
logger.info(f"Script directory: {script_dir}")

img_path = f"data_pixtral_llava/images"
# data_csv_path = f"data_pixtral_llava/sampled_annotated_articles_600.csv"

        
logger.info(f"Image path: {img_path}")

file_suffix = "revision"
PROMPT_MAPPING = {
        "llava-hf/llava-hf/llava-1.5-13b-hf": PROMPT_DICT_LLAVA,
        "mistralai/Pixtral-12B-2409": PROMPT_DICT_PIXTRAL_REVISION,
        }


def file_to_data_url(file_path: str):
    """
    Convert a local image file to a data URL.
    """    
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    _, extension = os.path.splitext(file_path)
    mime_type = f"image/{extension[1:].lower()}"
    
    return f"data:{mime_type};base64,{encoded_string}"

def annotate_frames(model_code, dir_name, data_csv_path)-> None:

    model_name_short = model_code.split('/')[1].split('-')[0]

    logging.info(f"Directory name: {dir_name}")
    logging.info(f"Type of dir name: {type(dir_name)}")

    unfound_images = []

    data_csv_df = pd.read_csv(data_csv_path)
    # data_csv_df = data_csv_df.sample(n=100, random_state=42)
    downloaded_uuids = data_csv_df["id"].tolist()

    logger.info(f"Processing UUIDs: {len(downloaded_uuids)}")
    logger.info(f"Loading CSV from path: {data_csv_path}")


    logger.info(f"Will rundata for downloaded images, DF shape: {data_csv_df.shape}")

    output_file = f"data_pixtral_llava/{model_name_short}_annotations_{file_suffix}.jsonl"
    if os.path.exists(output_file):
        logging.info(f"Existed! Deleting existing file: {output_file}")
        os.remove(output_file)
        
    output_fail_file = f"data_pixtral_llava/{model_name_short}_fail_annotations_{file_suffix}.txt"
    if os.path.exists(output_fail_file):
        logging.info(f"Existed! Deleting existing file: {output_fail_file}")
        os.remove(output_fail_file)

    
    if model_code == "mistralai/Pixtral-12B-2409":
        # sampling_params = SamplingParams(max_tokens=8192)
        sampling_params = SamplingParams(max_tokens=1024)
        vlm = LLM(model=model_code, tokenizer_mode="mistral", dtype="half", max_model_len =7000)
    else:
        vlm = LLM(model=model_code)
        sampling_params = SamplingParams(temperature=0.2, max_tokens=2000)

    # Issues: https://github.com/vllm-project/vllm/issues/8863
    
    
    # ids, headlines, text_frame_names = data_csv_df["text_id"].tolist(), data_csv_df["title"].tolist(), data_csv_df["text_frame_name"].tolist()
    ids, headlines, topics = data_csv_df["id"].tolist(), data_csv_df["title"].tolist(), data_csv_df["topic_label"].tolist()

    logging.info(f"Number of images to process: {len(ids)}")

    model_prompt_dict = PROMPT_MAPPING[model_code]

    # ADD tqdm here to see progress
    logger.info(f"Number of data points processing: {len(ids)}")
    for uuid, headline, topic in zip(ids, headlines, topics):

        # load image from image directory
        image_file_name = os.path.join(img_path, f"{uuid}.jpg")
        if os.path.exists(image_file_name):
            raw_image = Image.open(image_file_name).convert("RGB")
            # resize image to 512x512
            raw_image = raw_image.resize((512, 512))
        else:
            unfound_images.append(uuid)
            logger.info(f"Image file not found: {image_file_name}")
            continue

        img_annotations = {}

        logger.info(f"Processing uuid: {uuid}")
        logger.info(f"Image file: {image_file_name}")

        for task, prompt in model_prompt_dict.items(): 
            # Inference with image embeddings as input
            if model_code == "mistralai/Pixtral-12B-2409":
                
                image_source = file_to_data_url(image_file_name)
                messages = [
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_source}}]
                            },
                        ]
                outputs = vlm.chat(messages, sampling_params=sampling_params)

            else:
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
        img_annotations["image_url"] = f"images/{uuid}.jpg"
        img_annotations["title"] = headline
        img_annotations["uuid"] = uuid
        img_annotations["topic_label"] = topic
        # img_annotations["text_frame_name"]= text_frame

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
    parser.add_argument('--model_name', type=str, help='Model name', default='mistralai/Pixtral-12B-2409')
    parser.add_argument('--data_file', type=str, help='Data file with image urls', default="/projects/frame_align/data/raw/2023-24/default/datawithtopiclabels.csv")
    parser.add_argument('--dir_name', type=str, help='Directory name for saving annotated frames', default="default")
    args = parser.parse_args()

    dir_name = args.dir_name.split("/")[-1]
    # annotate_frames(args.model_name, args.data_file, args.dir_name)
    annotate_frames(args.model_name, dir_name, args.data_file)

if __name__ == "__main__":
    main()
    print("Done")
