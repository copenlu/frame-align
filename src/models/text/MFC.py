import json
import torch
import random
import numpy as np
import pandas as pd

from pdb import set_trace
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


random.seed(42)
torch.manual_seed(42)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = ['mistralai/Mistral-7B-Instruct-v0.2']#,'meta-llama/Llama-2-7b-chat-hf'] #'databricks/dbrx-instruct'

frames = "{1:Economic - costs, benefits, or other financial implications,\
    2:Capacity and resources - availability of physical, human, or financial resources, and capacity of current systems, \
    3:Morality - religious or ethical implications,\
    4:Fairness and equality - balance or distribution of rights, responsibilities, and resources,\
    5:Legality, constitutionality and jurisprudence - rights, freedoms, and authority of individuals, corporations, and government,\
    6:Policy prescription and evaluation - discussion of specific policies aimed at addressing problems,\
    7:Crime and punishment - effectiveness and implications of laws and their enforcement,\
    8:Security and defense - threats to welfare of the individual, community, or nation,\
    9:Health and safety - health care, sanitation, public safety,\
    10:Quality of life - threats and opportunities for the individual’s wealth, happiness, and well-being,\
    11:Cultural identity - traditions, customs, or values of a social group in relation to a policy issue,\
    12:Public opinion - attitudes and opinions of the general public, including polling and demographics,\
    13:Political - considerations related to politics and politicians, including lobbying, elections, and attempts to sway voters,\
    14:External regulation and reputation - international reputation or foreign policy of the U.S,\
    15:Other - any coherent group of frames not covered by the above categories}"

sys_prompt = f" You are a journalism scholar doing framing analysis of news articles.\
    Framing is defined as selecting and highlighting some facets of events or issues, and making connections among them so as to promote a particular interpretation, evaluation, and/or solution.\
    A dictionary of generic frames with a frame_id, frame_name and its description is: {frames}.\
    Your task is to code articles for the listed frames, output the confidence in the answer 0-100, and provide a reasoning or justification for it. Format your output as a list of json entries with each entry having fields 'frame_id', 'frame_name', 'confidence' and 'reasoning'. Output only the list and no other text."
    
user_prompt = " Output up to three generic frames and the corresponding reasonings in order of your confidence of your answer, for the article below:\n"

def get_messages(model_name:str, article:str) -> list:
    if 'mistral' in model_name:
        messages = [{"role": "user", "content": sys_prompt + user_prompt + article}]
    else:
        messages = [{"role": "system", "content": sys_prompt},
            {"role": "user", "content":  user_prompt + article}]
    return messages

def annotate_frames(model_name)-> None:
    model_name_short = model_name.split('/')[1].split('-')[0]

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/projects/copenlu/data/models/")
    model = AutoModelForCausalLM.from_pretrained(model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir="/projects/copenlu/data/models/")
        
    tokenizer.pad_token = tokenizer.eos_token

    data_df = pd.read_csv("./data/processed/mfc_consolidated.csv")
    num_examples = len(data_df)

    for index, row in data_df.iterrows():
        if index <= 27618:
            continue
        text = row['clean_text']
        messages = get_messages(model_name, text)

        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

        inputs = inputs.to(device)
        input_length = inputs.shape[1]

        outputs = model.generate(inputs, max_new_tokens=5000, pad_token_id=tokenizer.pad_token_id)
        generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        article_json = {}
        try:
            # Extract the json from the generated string
            begin = generated_text.find("[")
            end = generated_text.find("]")
            generated_text = generated_text[begin:end+1]
            article_json['predictions'] = generated_text
        except Exception as e:
            print("Skipped- ", e)
            article_json['predictions'] = None
        finally:
            article_json["text"] = text
            article_json["label"] = row['label']
            article_json["topic"] = row['topic']
            article_json["id"] = index
            with open(f"./data/annotated/mfc_annotated_{model_name_short}.json", "a") as f:
                json.dump(article_json, f)
                f.write("\n")

def main():
    for model_name in models:
        annotate_frames(model_name)

if __name__ == "__main__":
    main()