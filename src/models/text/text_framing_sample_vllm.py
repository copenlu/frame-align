import json
import torch
import random
import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from pdb import set_trace
from vllm import LLM, SamplingParams
from text_prompts import SYS_PROMPT, POST_PROMPT, text_prompt_dict

random.seed(42)
torch.manual_seed(42)

def get_messages(model_code:str, article:str, task_prompt:str) -> list:
    if 'mistral' in model_code:
        messages = [{"role": "user", "content": SYS_PROMPT + task_prompt + f"Article: {article}\n" + POST_PROMPT}]
    else:
        messages = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": task_prompt + f"Article: {article}\n" + POST_PROMPT}]
    return messages

def annotate_frames(model_code, data_file)-> None:
    model_name_short = model_code.split('/')[1].split('-')[0]
    
    news_df = pd.read_csv(data_file)

    llm = LLM(model=model_code)
    sampling_params = SamplingParams(max_tokens=6000, temperature=0.2)

    for i, row in news_df.iterrows():
        article_text = row['maintext']
        title = row['title']
        uuid = row['id']

        article_annotations = {}
        for task, task_prompt in text_prompt_dict.items():

            messages = get_messages(model_code, article_text, task_prompt)
            completion = llm.chat(messages,
                   sampling_params=sampling_params,
                   use_tqdm=False)
                
            output_text = completion[0].outputs[0].text
            try:
                output_json = json.loads(output_text)
                article_annotations.update(output_json)

            except Exception as e:
                print(f"Skipped-{i}-uuid-{uuid}-{task}: {e}")
                pass
        article_annotations["article_text"] = article_text
        article_annotations["title"] = title
        article_annotations["id"] = uuid

        with open(f"./data/annotated/text/topic_samples_{model_name_short}.jsonl", "a") as f:
            json.dump(article_annotations, f)
            f.write("\n")

def main():
    parser = argparse.ArgumentParser(description='Annotate text frames using VLLM')
    parser.add_argument('--model_code', type=str, default='mistralai/Mistral-7B-Instruct-v0.2', help='Model code for VLLM')
    parser.add_argument('--data_file', type=str, default='./data/raw/2023-24/topic_samples.csv', help='CSV file to annotate')
    args = parser.parse_args()
    annotate_frames(args.model_code, args.data_file)

if __name__ == "__main__":
    main()