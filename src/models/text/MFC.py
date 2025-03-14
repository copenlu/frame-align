import json
import torch
import random
import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from vllm import LLM, SamplingParams
from text_prompts import SYS_PROMPT, POST_PROMPT, text_prompt_dict

seed = 42

random.seed(seed)
torch.manual_seed(seed)

def get_messages(model_code:str, article:str, task_prompt:str) -> list:
    messages = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": task_prompt + f"Article: {article}\n" + POST_PROMPT}]
    return messages

def annotate_frames(model_code, data_file, output_dir)-> None:
    model_name_short = model_code.split('/')[1].split('-')[0]

    # Month for which data is running
    data_name = Path(data_file).parent.stem

    output_file = Path(f"{output_dir}/{data_name}.jsonl")
    output_fail_file = Path(f"{output_dir}/{data_name}_fail.tsv")
    if output_file.exists():
        print(f"Output file {output_file} already exists. Skipping annotation.")
        return

    news_df = pd.read_csv(data_file).sample(frac=1, random_state=seed).reset_index(drop=True)

    if 'mistral' in model_code:
        llm = LLM(model=model_code, tokenizer_mode="mistral", config_format="mistral", load_format="mistral", max_model_len=8096, dtype='half')
    else:
        llm = LLM(model=model_code)

    sampling_params = SamplingParams(max_tokens=4000, temperature=0.2)

    for i, row in news_df.iterrows():
        article_text = row['clean_text']
        id = i

        article_annotations = {}
        for task, task_prompt in text_prompt_dict.items():
            if task != 'GenericFraming':
                continue

            messages = get_messages(model_code, article_text, task_prompt)
            completion = llm.chat(messages,
                   sampling_params=sampling_params,
                   use_tqdm=False)
                
            output_text = completion[0].outputs[0].text
            try:
                output_json = json.loads(output_text)
                article_annotations.update(output_json)
            except Exception as e:
                try:
                    output_json = json.loads(output_text[output_text.index('{'):output_text.rindex('}')+1])
                    article_annotations.update(output_json)
                except Exception as e:
                    print(f"Skipped-{i}-uuid-{i}-{task}: {e}")
                    with open(output_fail_file, "a") as f:
                        f.write(f"{i}\t{task}\t{output_text}\n")
                    continue
        article_annotations["id"] = i
        article_annotations['label'] = row['multiple_labels_names']

        with open(output_file, "a") as f:
            json.dump(article_annotations, f)
            f.write("\n")

def main():
    parser = argparse.ArgumentParser(description='Annotate text frames using VLLM')
    parser.add_argument('--model_code', type=str, default='mistralai/Mistral-7B-Instruct-v0.3', help='Model code for VLLM')
    parser.add_argument('--data_file', type=str, default='/projects/frame_align/data/raw/2023-2024/topic_samples.csv', help='CSV file to annotate')
    parser.add_argument('--output_dir', type=str, default='./data/annotated/text/', help='Output directory for annotated files')
    args = parser.parse_args()
    annotate_frames(args.model_code, args.data_file, args.output_dir)

if __name__ == "__main__":
    main()