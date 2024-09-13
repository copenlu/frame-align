import json
import torch
import random
import argparse
import numpy as np
import pandas as pd
import requests, os

from PIL import Image
from pathlib import Path
from pdb import set_trace
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModel
from transformers import AutoProcessor, LlavaForConditionalGeneration, PaliGemmaForConditionalGeneration

from prompts_llava import PROMPT_LIST_LLAVA
from prompts_gemma import PROMPT_LIST_GEMMA
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

def enforce_reproducibility(seed=1000):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    random.seed(seed)
    np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print current path
# Determine the path to the annotated data directory relative to the script's location
script_dir = Path(__file__).resolve().parent

logger.info(f"Script directory: {script_dir}")

data_csv = script_dir / "../../../data/raw/topic_samples.csv"

logger.info(f"Data CSV path: {data_csv}")

data_df = pd.read_csv(data_csv)

# data_df = data_df.iloc[2:10] #1st image still gives error. check the error: The channel dimension is ambiguous. Got image shape (1, 1, 3). Assuming channels are the first dimension. 


ids, image_urls, headlines = data_df["id"].tolist(), data_df["image_url"].tolist(), data_df["title"].tolist()

def vlm_with_prompt(model_id):

    MODEL_CLASS_MAPPING = {
        "llava-hf/llava-1.5-7b-hf": LlavaForConditionalGeneration,
        "google/paligemma-3b-mix-224": PaliGemmaForConditionalGeneration,
        "google/paligemma-3b-mix-448": PaliGemmaForConditionalGeneration,
        "google/paligemma-3b-pt-448": PaliGemmaForConditionalGeneration,
        # "OpenGVLab/InternVL-Chat-V1-5": AutoModel
    }

    ModelClass = MODEL_CLASS_MAPPING[model_id]

    model = ModelClass.from_pretrained(
            model_id, 
            device_map=device,
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True,
            quantization_config = quantization_config,
            # parameter only exists for intern-vl
            # fix this
            # trust_remote_code=True if model_id == "OpenGVLab/InternVL-Chat-V1-5" else False
        ).eval() # check if we need this for llava

    processor = AutoProcessor.from_pretrained(model_id)

    PROMPT_MAPPING = {
            "llava-hf/llava-1.5-7b-hf": PROMPT_LIST_LLAVA,
            "google/paligemma-3b-mix-224": PROMPT_LIST_GEMMA,
            "google/paligemma-3b-mix-448": PROMPT_LIST_GEMMA
        }
    
    output_file = script_dir / "../../../data/processed" / f"topic_sampled_jul23_vision_{model_id.split('/')[-1].split('-')[0]}.jsonl"
    os.remove(output_file) if os.path.exists(output_file) else None

    for uuid, image_file, headline in zip(ids, image_urls, headlines):

        decoded_texts = []
        logger.info(f"Processing uuid: {uuid}")
        logger.info(f"Image URL: {image_file}")

        # Check if the image is loaded
        try:
            raw_image = Image.open(requests.get(image_file, stream=True).raw).convert("RGB")
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

        # Process the image with the model
        for prompt in PROMPT_MAPPING[model_id]:
            if prompt not in [PROMPT_MAPPING[model_id][i] for i in [0, 4, 8, 13]]: #these are: caption, symbols, gender, frame
                continue
            
            inputs = processor(prompt, raw_image, return_tensors='pt').to(model.device, torch.float16)
            output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            input_len = inputs["input_ids"].shape[-1]

            # append the decoded text to the list
            decoded_output = processor.decode(output[0][input_len:], skip_special_tokens=True)

            # To do: remove ASSISTANT for llava
            answer = decoded_output.split("ASSISTANT:")[1].strip().replace("\n", "") if "ASSISTANT:" in decoded_output else decoded_output
            # json_answer = json.loads(answer)
            decoded_texts.append(answer)

        # Combine all key-value pairs into a single dictionary
        data_entry = {
            "uuid" : uuid,
            "image_file": image_file,
            "headline": headline,
            "caption": decoded_texts[0],
            # "category": decoded_texts[1],
            # "actors": decoded_texts[2],
            # "actor_roles": decoded_texts[3],
            "symbols": decoded_texts[4],
            # "representation": decoded_texts[5],
            # "numbers": decoded_texts[6],
            # "expressions": decoded_texts[7],
            "gender": decoded_texts[8],
            # "power": decoded_texts[9],
            # "intimacy": decoded_texts[10],
            # "image_emotion": decoded_texts[11],
            # "people_emotion": decoded_texts[12],
            "frame": decoded_texts[13],
        }

        with open(output_file, "a") as f:
            f.write(json.dumps(data_entry) + "\n")

    # Read the JSONL file
    data = []
    with open(output_file, "r") as f:
        for line in f:
            # Parse each JSON object
            json_obj = json.loads(line)
            data.append(json_obj)

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.json_normalize(data)
    csv_file = output_file.replace(".jsonl", ".csv")
    os.remove(csv_file) if os.path.exists(csv_file) else None
    df.to_csv(csv_file, index=False)


if __name__ == '__main__':
    # enforce_reproducibility()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, help="The name of the model whose data to convert", default='llava-hf/llava-1.5-7b-hf',
                        choices=['llava-hf/llava-1.5-7b-hf', "google/paligemma-3b-mix-224", "google/paligemma-3b-mix-448", "google/paligemma-3b-pt-448"]) # "OpenGVLab/InternVL-Chat-V1-5" is too heavy
    args = parser.parse_args()
    
    vlm_with_prompt(args.model_id)