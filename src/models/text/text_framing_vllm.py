import json
import torch
import random
import pickle
import logging
import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from vllm import LLM, SamplingParams
from text_prompts import SYS_PROMPT, POST_PROMPT, text_prompt_dict

logger = logging.getLogger(__name__)

random.seed(42)
torch.manual_seed(42)

def get_messages(article:str, task_prompt:str) -> list:
    messages = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": task_prompt + f"Article: {article}\n" + POST_PROMPT}]
    return messages

def annotate_frames(model_code, uuid_pkl_file, input_dir, output_dir)-> None:
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    uuid_pkl_file = Path(uuid_pkl_file)
    # model_name_short = model_code.split('/')[1].split('-')[0]

    # load pickle file with month name and uuids
    with open(uuid_pkl_file, "rb") as f:
        month_uuids_dict = pickle.load(f)

    month_name = list(month_uuids_dict.keys())[0]
    month_uuids = month_uuids_dict[month_name]

    input_file = input_dir / f"{month_name}.csv"
    logger.info(f"Loading CSV from path: {input_file}")

    output_file = Path(f"{output_dir}/textframes_{uuid_pkl_file.stem}.jsonl")
    output_fail_file = Path(f"{output_dir}/{uuid_pkl_file.stem}_fail.tsv")
    if output_file.exists():
        logger.info(f"Output file {output_file} already exists. Exiting...")
        return

    og_data_df = pd.read_csv(input_file)
    logger.info(f"Original Dataframe shape: {og_data_df.shape}, UUIDs to load: {len(month_uuids)}")
    data_df = og_data_df[og_data_df['id'].isin(month_uuids)]
    logger.info(f"Filtered Dataframe shape: {data_df.shape}, UUIDs to load: {len(month_uuids)}")

    if 'mistral' in model_code:
        llm = LLM(model=model_code, tokenizer_mode="mistral", config_format="mistral", load_format="mistral", max_model_len=8096, dtype='half')
    else:
        llm = LLM(model=model_code)

    sampling_params = SamplingParams(max_tokens=4000, temperature=0.2)

    for i, row in data_df.iterrows():
        article_text = row['maintext']
        uuid = row['id']

        article_annotations = {}
        for task, task_prompt in text_prompt_dict.items():

            messages = get_messages(article_text, task_prompt)
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
                    print(f"Skipped-{i}-uuid-{uuid}-{task}: {e}")
                    with open(output_fail_file, "a") as f:
                        f.write(f"{uuid}\t{task}\t{output_text}\n")
                    continue
        article_annotations["id"] = uuid
        article_annotations['title'] = row['title']   

        with open(output_file, "a") as f:
            json.dump(article_annotations, f)
            f.write("\n")

def main():
    parser = argparse.ArgumentParser(description='Annotate text frames using VLLM')
    parser.add_argument('--model_name', type=str, default='mistralai/Mistral-7B-Instruct-v0.3', help='Model code for VLLM')
    parser.add_argument('--uuid_pkl_file', type=str, default='/projects/frame_align/data/uuid_splits/set_0.pkl', help='Pickle file with month name and uuids')
    parser.add_argument('--input_dir', type=str, default='/projects/frame_align/data/filtered/text/', help='CSV file to annotate')
    parser.add_argument('--output_dir', type=str, default='/projects/frame_align/data/annotated/text/', help='Output directory for annotated files')
    args = parser.parse_args()
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        logger.info(f"Using GPU: {gpu_name} with {gpu_memory / (1024 ** 3):.2f} GB of memory")
    else:
        logger.info("No GPU available, using CPU")
    annotate_frames(args.model_name, args.uuid_pkl_file, args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()