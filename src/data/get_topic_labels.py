import ast
import time
import json
import pickle
import logging
import argparse
import anthropic
import pandas as pd

from pathlib import Path
from pdb import set_trace
from claude_prompts import TOPIC_LABEL_PROMPT, MERGE_TOPICS_PROMPT

logging.basicConfig(filename='app.log', level=logging.INFO)
logger = logging.getLogger(__name__)

client = anthropic.Anthropic()

def get_bertopic_topics(data_dir="./data/raw/2023-24"):
    data_path = Path(data_dir)
    topics = []
    for month_dir in data_path.iterdir():
        df = pd.read_csv(month_dir/"datawithtopics_merged.csv")
        topics.extend(df['auto_topic_label'].unique().tolist())
    topics = list(set(topics))
    pickle.dump(topics, open(data_dir/"bertopic_topics.pkl", "wb"))
    return None

def prompt_claude(system_prompt, user_prompt):
    message = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=8000,
                temperature=0,
                system=system_prompt,
                messages=[{
                        "role": "user",
                        "content": [{
                                "type": "text",
                                "text": str(user_prompt)
                            }]
                    }]
                )
    return message
    
def get_topic_labels(data_dir="./data/raw/2023-24"):
    topics_file = Path(f"{data_dir}/bertopic_topics.pkl")
    if not topics_file.exists():
        logger.info("Getting BERTopic topics")
        get_bertopic_topics(data_dir)
    topics = pickle.load(open(topics_file, "rb"))
    topic_label_file = Path(f"{data_dir}/topic_labels.json")
    if topic_label_file.exists():
        topic_label_dict = json.load(topic_label_file.open())
    else:
        topic_label_dict = {}

    topics_to_process = [topic for topic in topics if topic not in topic_label_dict]
    topics_at_a_time = 200
    for i in range(0, len(topics_to_process), topics_at_a_time):
        try:
            logger.info(f"Processing topics from {i} to {i+topics_at_a_time}")
            message = prompt_claude(TOPIC_LABEL_PROMPT, topics_to_process[i:i+topics_at_a_time])
            # logger.info(message.content[0].text)
            out_dict = ast.literal_eval(message.content[0].text)
            logger.info(f"Adding {len(out_dict)} topics to meta_topic_dict")
            topic_label_dict.update(out_dict)
            time.sleep(5) # to avoid rate limit
        except Exception as e:
            if "Overloaded" in str(e).lower():
                logger.info("Overload error. Sleeping for 1 minute")
                time.sleep(60)
                message = prompt_claude(TOPIC_LABEL_PROMPT, topics_to_process[i:i+topics_at_a_time])
                out_dict = ast.literal_eval(message.content[0].text)
                logger.info(f"Adding {len(out_dict)} topics to topic_label_dict")
                topic_label_dict.update(out_dict)
                continue
            logger.error(e)
            print(e)
            json.dump(topic_label_dict, open(topic_label_file, "w"))
            continue
    json.dump(topic_label_dict, open(topic_label_file, "w"))

def merge_topic_labels(data_dir="./data/raw/2023-24"):
    topic_label_file = Path(f"{data_dir}/topic_labels.json")
    topic_label_dict = json.load(open(topic_label_file))
    topic_label_list = list(set(topic_label_dict.values()))
    logger.info("Merging topics")
    logger.info(f"Number of topics to merge: {len(topic_label_list)}")
    try:
        message = prompt_claude(MERGE_TOPICS_PROMPT, topic_label_list)
        merged_topic_dict = ast.literal_eval(message.content[0].text)
    except Exception as e:
        logger.error(e)
        return
    logger.info(f"Merged from {len(topic_label_list)} topics to {len(merged_topic_dict)} topics")
    topic_label_dict = {k: merged_topic_dict[v] for k, v in topic_label_dict.items()}
    json.dump(topic_label_dict, open(topic_label_file, "w"))

def check_topic_labels(data_dir="./data/raw/2023-24"):
    topic_dict = pickle.load(open(f"{data_dir}/bertopic_topics.pkl", "rb"))
    topic_label_dict = json.load(open(f"{data_dir}/topic_labels.json"))
    try:
        assert len(topic_dict) == len(topic_label_dict)
    except Exception as e:
        logger.error(e)
        get_topic_labels(data_dir)
    return

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_dir", type=str, default=None)
    args = argparser.parse_args()
    get_topic_labels(args.data_dir)
    check_topic_labels(args.data_dir)
    merge_topic_labels(args.data_dir)

if __name__ == "__main__":
    main()