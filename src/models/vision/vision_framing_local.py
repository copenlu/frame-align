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
from prompts_pixtral_underrevision import PROMPT_DICT_PIXTRAL

import base64

random.seed(42)
torch.manual_seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

script_dir = Path(__file__).resolve().parent    
logger.info(f"Script directory: {script_dir}")

img_path = f"data_pixtral_llava/images/"
        
logger.info(f"Image path: {img_path}")

def file_to_data_url(file_path: str):
    """
    Convert a local image file to a data URL.
    """    
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    _, extension = os.path.splitext(file_path)
    mime_type = f"image/{extension[1:].lower()}"
    
    return f"data:{mime_type};base64,{encoded_string}"

def annotate_frames(model_code, output_dir, input_file_path)-> None:

    model_name_short = model_code.lower().split('/')[1].split('-')[0]
    unfound_images = []

    data_df = pd.read_csv(input_file_path)
    input_file_path = Path(input_file_path)
    output_dir = Path(output_dir)
    output_file = output_dir/f"multipleframes_{input_file_path.stem}_{model_name_short}_anno.jsonl"
    output_fail_file = output_dir/f"{model_name_short}_anno_fail.tsv"
    # data_csv_df = data_csv_df.sample(n=100, random_state=42)
    uuids = data_df["uuid"].tolist()

    logger.info(f"Processing UUIDs: {len(uuids)}")
    logger.info(f"Loading CSV from path: {input_file_path}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Output file: {output_file}")

    if output_file.exists():
        logging.info(f"Existed! Deleting existing file: {output_file}")
        os.remove(output_file)
    if output_fail_file.exists():
        logging.info(f"Existed! Deleting existing file: {output_fail_file}")
        os.remove(output_fail_file)
    
    if model_code == "mistralai/Pixtral-12B-2409":
        sampling_params = SamplingParams(temperature=0.2, max_tokens=1024)
        vlm = LLM(model=model_code, tokenizer_mode="mistral", dtype="half", max_model_len=7000)
    else:
        vlm = LLM(model=model_code)
        sampling_params = SamplingParams(temperature=0.2, max_tokens=2000)

    # Issues: https://github.com/vllm-project/vllm/issues/8863
    
    model_prompt_dict = PROMPT_DICT_PIXTRAL

    for uuid in uuids:

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

        for prompt_setting, prompt in model_prompt_dict.items(): 
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
                output_json = {f"{prompt_setting}_{k}":v for k,v in output_json.items()}
                img_annotations.update(output_json)
            except Exception as e:
                try:
                    output_json = json.loads(output_text[output_text.index('{'):output_text.rindex('}')+1])
                    output_json = {f"{prompt_setting}_{k}":v for k,v in output_json.items()}
                    img_annotations.update(output_json)
                except Exception as e:
                    print(f"Skipped-uuid-{uuid}-{prompt_setting}: {e}")
                    with open(output_fail_file, "a") as f:
                        f.write(f"{uuid}\t{prompt_setting}\t{output_text}\n")
                    continue
        img_annotations["image_url"] = f"{img_path}{uuid}.jpg"
        img_annotations["uuid"] = uuid
          
        with open(output_file, "a") as f:
            json.dump(img_annotations, f)
            f.write("\n")
    logger.info(f"No. of unfound images: {len(unfound_images)}")

    # count accuracy
    # output_df = pd.read_json(output_file, lines=True)
    # merged_df = output_df[['uuid','frame-name']].merge(data_df[['uuid','merged_labels']], on='uuid', how='inner')
    # merged_df['frame-name'] = merged_df['frame-name'].str.lower()
    # for i, row in merged_df.iterrows():
    #     if row['frame-name'] in row['merged_labels']:
    #         merged_df.loc[i, 'correct'] = 1
    #     else:
    #         merged_df.loc[i, 'correct'] = 0
    # accuracy = merged_df['correct'].sum()/merged_df['correct'].count()
    # logger.info(f"Accuracy: {accuracy} for {len(merged_df)} samples")

def main():
    parser = argparse.ArgumentParser(description='Annotate image frames using a VLLM model')
    parser.add_argument('--model_name', type=str, help='Model name', default='mistralai/Pixtral-12B-2409')
    parser.add_argument('--data_file', type=str, help='Data file with image urls', default="/projects/frame_align/data/raw/2023-24/default/datawithtopiclabels.csv")
    parser.add_argument('--output_dir', type=str, help='Directory name for saving annotated frames', default="default")
    args = parser.parse_args()

    # annotate_frames(args.model_name, args.data_file, args.dir_name)
    annotate_frames(args.model_name, args.output_dir, args.data_file)

if __name__ == "__main__":
    main()
    print("Done")
