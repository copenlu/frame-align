import json
import torch
import random
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

# llava model
model_name = "llava-hf/llava-1.5-7b-hf"

PROMPT_MAPPING = {
        "llava-hf/llava-1.5-7b-hf": PROMPT_DICT_LLAVA
        }

sampling_params = SamplingParams(temperature=0.2,
                                max_tokens=1000)

vlm = LLM(model=model_name)

def annotate_frames(model_code)-> None:
    model_name_short = model_code.split('/')[1].split('-')[0]

    # script_dir = Path(__file__).resolve().parent

    # logger.info(f"Script directory: {script_dir}")
    data_csv = "./data/raw/2023-24/July-23/topic_samples.csv"
    logger.info(f"Data CSV path: {data_csv}")
    data_df = pd.read_csv(data_csv)
    ids, image_urls, headlines = data_df["id"].tolist(), data_df["image_url"].tolist(), data_df["title"].tolist()

    model_prompt_dict = PROMPT_MAPPING[model_code]

    # news_df = pd.read_csv(data_path/"July-23"/"topic_samples.csv")

    for uuid, image_file, headline in zip(ids, image_urls, headlines):
        img_annotations = {}
        try:
            logger.info(f"Opening image")
            response = requests.get(image_file, stream=True, timeout=20)  # Add a timeout (in seconds)
            response.raise_for_status()  # Raise an HTTPError if the status is not 200
            raw_image = Image.open(response.raw).convert("RGB")
            logger.info(f"Image opened")
            
            # Check the shape of the image tensor
            image_tensor = torch.tensor(np.array(raw_image))
            if image_tensor.shape[-1] != 3:
                raise ValueError(f"Unexpected image shape: {image_tensor.shape}")
            if image_tensor.shape[0] == 1 and image_tensor.shape[1] == 1:
                logger.info(f"Skipping image with shape {image_tensor.shape} - uuid: {uuid}")
                continue
            del image_tensor

        except Exception as e:
            logger.info(f"Image URL: {image_file}")
            logger.error(f"Image error {e} - uuid: {uuid}")
            continue

        logger.info(f"Processing uuid: {uuid}")
        logger.info(f"Image URL: {image_file}")

        for task, prompt in model_prompt_dict.items(): 

            # Inference with image embeddings as input
            inputs = {
                "prompt": prompt,
                "multi_modal_data": {"image": raw_image}
            }
            outputs = vlm.generate(inputs, sampling_params=sampling_params)
            
            try:
                output_json = json.loads(outputs[0].outputs[0].text)
                img_annotations.update(output_json)

            except Exception as e:
                print(f"Skipped-{uuid}-{task}: {e}")
                pass
        img_annotations["image_url"] = image_file
        img_annotations["title"] = headline
        img_annotations["uuid"] = uuid

        with open(f"./data/annotated/topic_sampled_jul23_annotated_{model_name_short}_vllm.jsonl", "a") as f:
            json.dump(img_annotations, f)
            f.write("\n")

def main():
    annotate_frames(model_name)

if __name__ == "__main__":
    main()