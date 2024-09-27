import json
import torch
import random
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

# To Fix: Replace this with args data_file
month_dir = Path("/projects/frame_align/data/raw/2023-11-01_2023-11-30/")
pickle_path = month_dir / "selected_uuids.pkl"
load_uuids = pd.read_pickle(pickle_path)
# convert to list
selected_uuids = list(load_uuids)


PROMPT_MAPPING = {
        "llava-hf/llava-1.5-7b-hf": PROMPT_DICT_LLAVA
        }

sampling_params = SamplingParams(temperature=0.2,
                                max_tokens=1000)

def annotate_frames(model_code, data_file)-> None:
    model_name_short = model_code.split('/')[1].split('-')[0]
    vlm = LLM(model=model_code)

    data_df = pd.read_csv(data_file)

    data_df = data_df[data_df["id"].isin(selected_uuids)]
    ids, image_urls, headlines = data_df["id"].tolist(), data_df["image_url"].tolist(), data_df["title"].tolist()

    model_prompt_dict = PROMPT_MAPPING[model_code]

    # ADD tqdm here to see progress
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

        # logging location of current directory
        logger.info(f"context: {Path.cwd()}")
        
        # ALso vision dir doesnt exist. Create it. Update this to use the month_dir variable
        output_file = f"/projects/frame_align/data/raw/2023-11-01_2023-11-30/annotated/vision/topic_samples_{model_name_short}.jsonl"
        # create the directory if it does not exist else write to the file
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "a") as f:
            json.dump(img_annotations, f)
            f.write("\n")
        # with open(f"data/annotated/vision/topic_samples_{model_name_short}.jsonl", "a") as f:
        #     json.dump(img_annotations, f)
        #     f.write("\n")

def main():
    parser = argparse.ArgumentParser(description='Annotate image frames using a VLLM model')
    parser.add_argument('--model_name', type=str, help='Model name', default='llava-hf/llava-1.5-7b-hf')
    parser.add_argument('--data_file', type=str, help='Data file with image urls', default="/projects/frame_align/data/raw/2023-24/topic_samples.csv")
    args = parser.parse_args()
    annotate_frames(args.model_name, args.data_file)

if __name__ == "__main__":
    main()