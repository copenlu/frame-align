import json
import torch
import random
import numpy as np
import pandas as pd

from pathlib import Path
from pdb import set_trace
from openai import OpenAI
from text_prompts import SYS_PROMPT, POST_PROMPT, text_prompt_dict

random.seed(42)
torch.manual_seed(42)

model_code = 'meta-llama/Meta-Llama-3-8B-Instruct'

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="framing")

def get_messages(model_code:str, article:str, task_prompt:str) -> list:
    if 'mistral' in model_code:
        messages = [{"role": "user", "content": SYS_PROMPT + task_prompt + f"Article: {article}\n" + POST_PROMPT}]
    else:
        messages = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": task_prompt + f"Article: {article}\n" + POST_PROMPT}]
    return messages

def annotate_frames(model_code)-> None:
    model_name_short = model_code.split('/')[1].split('-')[0]
    
    data_path = Path("~/projects/frame-align/data/raw/2023-24/")
    news_df = pd.read_csv(data_path/"July-23"/"topic_samples.csv")

    for i, (uuid, row) in enumerate(news_df.iterrows()):
        article_text = row['maintext']
        title = row['title']

        article_annotations = {}
        for task, task_prompt in text_prompt_dict.items():

            messages = get_messages(model_code, article_text, task_prompt)
            completion = client.chat.completions.create(
                model=model_code,
                messages=messages
            )
            output_text = completion.choices[0].message
            try:
                output_json = json.loads(output_text.content)
                article_annotations.update(output_json)

            except Exception as e:
                print(f"Skipped-{i}-{task}: {e}")
                pass
        article_annotations["article_text"] = article_text
        article_annotations["title"] = title
        article_annotations["id"] = i
        article_annotations["uuid"] = uuid 

        with open(f"./data/annotated/topic_sampled_jul23_annotated_{model_name_short}_vllm.jsonl", "a") as f:
            json.dump(article_annotations, f)
            f.write("\n")

def main():
    # for model_code in ['meta-llama/Meta-Llama-3-8B-Instruct', 'meta-llama/Meta-Llama-3.1-8B']:
    # for model_code in ['mistralai/Mistral-7÷B-Instruct-v0.2']:
    annotate_frames(model_code)

if __name__ == "__main__":
    main()