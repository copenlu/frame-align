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
from openai import OpenAI
from vllm import LLM, SamplingParams
from prompts_vllm_llava import PROMPT_LIST_LLAVA_vLLM

random.seed(42)
torch.manual_seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# llava model
model_name = "llava-hf/llava-1.5-7b-hf"
PROMPT_MAPPING = {
        "llava-hf/llava-1.5-7b-hf": PROMPT_LIST_LLAVA_vLLM
    }

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="framing") # check this

def get_messages(model_id:str, prompt:str, image_file:str) -> list:
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_file}},
        ],
    }]
    return messages


def annotate_frames(model_code)-> None:
    model_name_short = model_code.split('/')[1].split('-')[0]

    script_dir = Path(__file__).resolve().parent

    logger.info(f"Script directory: {script_dir}")
    data_csv = script_dir / "../../../data/raw/topic_samples.csv"
    logger.info(f"Data CSV path: {data_csv}")
    data_df = pd.read_csv(data_csv)
    ids, image_urls, headlines = data_df["id"].tolist(), data_df["image_url"].tolist(), data_df["title"].tolist()

    # news_df = pd.read_csv(data_path/"July-23"/"topic_samples.csv")

    img_annotations = {}
    for uuid, image_file, headline in zip(ids, image_urls, headlines):
        decoded_texts = []
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

        except Exception as e:
            logger.info(f"Image URL: {image_file}")
            logger.error(f"Image error {e} - uuid: {uuid}")
            continue

        logger.info(f"Processing uuid: {uuid}")
        logger.info(f"Image URL: {image_file}")

        for index, prompt in enumerate(PROMPT_MAPPING[model_id]):

            messages = get_messages(model_code, prompt, image_file)
            completion = client.chat.completions.create(
                model=model_code,
                messages=messages
            )

            import pdb; pdb.set_trace()
            print(f"Chat completion output-{index}: {completion.choices[0].message.content}")
            output_text = completion.choices[0].message
            try:
                output_json = json.loads(output_text.content)
                img_annotations.update(output_json)

            except Exception as e:
                print(f"Skipped-{i}-{task}: {e}")
                pass
        # article_annotations["article_text"] = article_text
        # article_annotations["title"] = title
        # article_annotations["id"] = i
        # article_annotations["uuid"] = uuid 

        # with open(f"./data/annotated/topic_sampled_jul23_annotated_{model_name_short}_vllm.jsonl", "a") as f:
        #     json.dump(article_annotations, f)
        #     f.write("\n")

def main():
    annotate_frames(model_name)

if __name__ == "__main__":
    main()