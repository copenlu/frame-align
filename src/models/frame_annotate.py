import json
import torch
import random
import numpy as np
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


random.seed(42)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

frames = "Economic: costs, benefits, or other financial implications.\
    Capacity and resources: availability of physical, human, or financial resources, and capacity of current systems. \
    Morality: religious or ethical implications\
    Fairness and equality: balance or distribution of rights, responsibilities, and resources\
    Legality, constitutionality and jurisprudence: rights, freedoms, and authority of individuals, corporations, and government\
    Policy prescription and evaluation: discussion of specific policies aimed at addressing problems\
    Crime and punishment: effectiveness and implications of laws and their enforcement\
    Security and defense: threats to welfare of the individual, community, or nation\
    Health and safety: health care, sanitation, public safety\
    Quality of life: threats and opportunities for the individual’s wealth, happiness, and well-being\
    Cultural identity: traditions, customs, or values of a social group in relation to a policy issue\
    Public opinion: attitudes and opinions of the general public, including polling and demographics\
    Political: considerations related to politics and politicians, including lobbying, elections, and attempts to sway voters\
    External regulation and reputation: international reputation or foreign policy of the U.S.\
    Other: any coherent group of frames not covered by the above categories"

def annotate_frames()->None:
    """Annotate frames using Mistral-7B-Instruct"""
    
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",
        load_in_4bit=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir="/projects/copenlu/data/mistral/")

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", padding_side="left", cache_dir="/projects/copenlu/data/mistral/")
    
    tokenizer.pad_token = tokenizer.eos_token

    news_df = pd.read_csv("./data/raw/news_data_100.csv", index_col=[0])
    news_df['text'] = news_df['chunk2'] + ' ' + news_df['chunk3'] + ' ' + \
                news_df['chunk4'] + ' ' + news_df['chunk5'] + ' ' + news_df['chunk6']
    news_df['title'] = news_df['chunk1']

    article_json = {}

    for i, (index, row) in enumerate(news_df.sample(frac=1).iterrows()):
        text = row['text']
        title = row['title']
        
        inputs = tokenizer.encode("You are a journalism scholar doing analysis of news articles. A list of generic frames and their description is: " + frames + "\n Your task is to annotate the article below for one of the listed frames and provide reasoning for it. " + text + "\n Output the generic frame and the reasoning. Format your output in a json format with fields 'frame' and 'reasoning'", return_tensors="pt", padding=True).to(device)

        try:
            outputs = model.generate(inputs, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, max_new_tokens=5000, pad_token_id=tokenizer.pad_token_id)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract the json from the generated string
            generated_text = generated_text.split("Output the generic frame and the reasoning. Format your output in a json format with fields 'frame' and 'reasoning'")[1]
            article_json = json.loads(generated_text)
            article_json["text"] = text
            article_json["title"] = title
            article_json["id"] = index

            with open(f"./data/interim/news_data_100_annotated.json", "a") as f:
                json.dump(article_json, f)
                f.write("\n")

        except Exception as e:
            print(e)
            continue

def main():
    annotate_frames()

if __name__ == "__main__":
    main()