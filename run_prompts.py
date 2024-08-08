import json
import torch
import random
import argparse
import numpy as np
from PIL import Image
import pandas as pd

from prompts import PROMPT_LIST
from prompts_gemma import PROMPT_LIST_GEMMA

from pdb import set_trace
import requests, os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModel
from transformers import AutoProcessor, LlavaForConditionalGeneration, PaliGemmaForConditionalGeneration


quantization_config = BitsAndBytesConfig(load_in_8bit=True)

def enforce_reproducibility(seed=1000):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    random.seed(seed)
    np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_csv = "merged_gpt4_llava_mistral.csv"
data_df = pd.read_csv(data_csv)
image_urls, texts, headlines = data_df["img_url"].tolist(), data_df["text"].tolist(), data_df["title"].tolist()

def vlm_with_prompt(model_id):

    MODEL_CLASS_MAPPING = {
        "llava-hf/llava-1.5-7b-hf": LlavaForConditionalGeneration,
        "google/paligemma-3b-mix-224": PaliGemmaForConditionalGeneration,
        "google/paligemma-3b-pt-448": PaliGemmaForConditionalGeneration,
        "OpenGVLab/InternVL-Chat-V1-5": AutoModel
    }

    output_file = model_id.split("/")[-1] + "frame.jsonl"
    os.remove(output_file) if os.path.exists(output_file) else None

    for image_file, text, headline in zip(image_urls, texts, headlines):
        decoded_texts = []
        raw_image = Image.open(requests.get(image_file, stream=True).raw)
        
        PROMPT_MAPPING = {
            "llava-hf/llava-1.5-7b-hf": PROMPT_LIST,
            "google/paligemma-3b-mix-224": PROMPT_LIST_GEMMA,
            "google/paligemma-3b-pt-448": PROMPT_LIST_GEMMA,
            "OpenGVLab/InternVL-Chat-V1-5": PROMPT_LIST
        }
        for prompt in PROMPT_MAPPING[model_id]:
        
            ModelClass = MODEL_CLASS_MAPPING[model_id]
            print(f"ModelClass: {ModelClass}")

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
            "image_file": image_file,
            "headline": headline,
            "text": text,
            "caption": decoded_texts[0],
            "category": decoded_texts[1],
            "actors": decoded_texts[2],
            "actor_roles": decoded_texts[3],
            "symbols": decoded_texts[4],
            "representation": decoded_texts[5],
            "numbers": decoded_texts[6],
            "expressions": decoded_texts[7],
            "gender": decoded_texts[8],
            "power": decoded_texts[9],
            "intimacy": decoded_texts[10],
            "image_emotion": decoded_texts[11],
            "people_emotion": decoded_texts[12],
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
    parser.add_argument("--model_id", type=str, help="The name of the model whose data to convert", default='OpenGVLab/InternVL-Chat-V1-5',
                        choices=['llava-hf/llava-1.5-7b-hf', "google/paligemma-3b-mix-224", "google/paligemma-3b-pt-448", "OpenGVLab/InternVL-Chat-V1-5"])
    args = parser.parse_args()
    
    vlm_with_prompt(args.model_id)