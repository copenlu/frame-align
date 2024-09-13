import json
import torch
import random
import numpy as np
import pandas as pd

from pathlib import Path
from text_prompts import SYS_PROMPT, POST_PROMPT, text_prompt_dict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

random.seed(42)
torch.manual_seed(42)

bnb_config = BitsAndBytesConfig(load_in_8bit=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_code = 'mistralai/Mistral-7B-Instruct-v0.2'

def get_messages(model_name:str, article:str, task_prompt:str) -> list:
    if 'mistral' in model_name:
        messages = [{"role": "user", "content": SYS_PROMPT + task_prompt + f"Article: {article}\n" + POST_PROMPT}]
    else:
        messages = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": task_prompt + f"Article: {article}\n" + POST_PROMPT}]
    return messages

def annotate_frames(model_code)-> None:
    model_name_short = model_code.split('/')[1].split('-')[0]

    tokenizer = AutoTokenizer.from_pretrained(model_code, cache_dir="/projects/copenlu/data/models/")
    model = AutoModelForCausalLM.from_pretrained(model_code,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir="/projects/copenlu/data/models/")
        
    tokenizer.pad_token = tokenizer.eos_token

    data_path = Path("~/projects/frame-align/data/raw/2023-24/")
    news_df = pd.read_csv(data_path/"July-23"/"topic_samples.csv")

    for i, (uuid, row) in enumerate(news_df.iterrows()):
        article_text = row['maintext']
        title = row['title']

        article_annotations = {}
        for task, task_prompt in text_prompt_dict.items():

            messages = get_messages(model_code, article_text, task_prompt)
            inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

            inputs = inputs.to(device)
            input_length = inputs.shape[1]

            outputs = model.generate(inputs, max_new_tokens=5000, pad_token_id=tokenizer.pad_token_id)
            output_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            try:
                output_json = json.loads(output_text)
                article_annotations.update(output_json)
            except Exception as e:
                print(f"Skipped-{i}-{task}: {e}")
                pass
        article_annotations["article_text"] = article_text
        article_annotations["title"] = title
        article_annotations["id"] = i
        article_annotations["uuid"] = uuid 

        with open(f"./data/annotated/topic_sampled_jul23_annotated_{model_name_short}.jsonl", "a") as f:
            json.dump(article_annotations, f)
            f.write("\n")

def main():
    # for model_code in ['meta-llama/Meta-Llama-3-8B-Instruct', 'meta-llama/Meta-Llama-3.1-8B']:
    for model_code in ['mistralai/Mistral-7B-Instruct-v0.2']:
        annotate_frames(model_code)

if __name__ == "__main__":
    main()