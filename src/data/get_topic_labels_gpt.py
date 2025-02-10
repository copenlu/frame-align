import os
import openai
import ast
import time
import json
import pickle
import logging
import argparse
import pandas as pd
import os
import openai

from pathlib import Path
from pdb import set_trace
from gpt_prompts import TOPIC_LABEL_PROMPT

logging.basicConfig(filename='app.log', level=logging.INFO)
logger = logging.getLogger(__name__)



# Initialize the OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#check if it found the key
# print(os.getenv("OPENAI_API_KEY"))

def prompt_openai(system_prompt, user_prompt):
    """
    Call OpenAI ChatCompletion to get a model response.

    :param system_prompt: The system instruction (string)
    :param user_prompt: The user prompt or data (string or list)
    :return: The model's response as text (string)
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or 'gpt-3.5-turbo'
            temperature=0,
            max_tokens=4000,
            messages=[
                {"role": "system", "content": "You are a journalism scholar doing topic analysis of news"},
                {"role": "user", "content": system_prompt + str(user_prompt)}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        print(f"OpenAI API error: {e}")
        raise e



def get_topic_labels(data_dir="/projects/frame_align/data/annotated"):
    topics_file = Path(f"{data_dir}/unique_topics.pkl")
    if not topics_file.exists():
        logger.info("Topics file not found.")
        exit()
    else:
        logger.info(f"Loading topics from {topics_file}")
        print(f"Loading topics from {topics_file}")
    topics = pickle.load(open(topics_file, "rb"))
    logger.info(f"Number of topics: {len(topics)}")
    print(f"Number of topics: {len(topics)}")
    
    test = False
    # select 5 random topics if test flag is set to True
    if test:
        topics = topics[:5]
    
    topic_label_file = Path(f"{data_dir}/latest_topic_labels.json")
    if topic_label_file.exists():
        topic_label_dict = json.load(topic_label_file.open())
    else:
        topic_label_dict = {}

    logger.info(f"Loading topics from topic_label_dict")
    print(f"Loading topics from topic_label_dict")
    # Convert topic_label_dict keys to a set
    already_labeled_topics = set(topic_label_dict.keys())

    # Now membership checks (topic in already_labeled_topics) are O(1)
    topics_to_process = [topic for topic in topics if topic not in already_labeled_topics]

    logger.info(f"Number of topics loaded: {len(topics_to_process)}")
    print(f"Number of topics loaded: {len(topics_to_process)}")
    print(f" Sample topics: {topics_to_process[:5]}")
    topics_at_a_time = 200

    total_batches = (len(topics_to_process) + topics_at_a_time - 1) // topics_at_a_time
    for batch_idx, i in enumerate(range(0, len(topics_to_process), topics_at_a_time), start=1): #len(topics_to_process)
        try:
            start_time = time.time()
            logger.info(
                f"Processing batch {batch_idx}/{total_batches} topics from {i} to {i+topics_at_a_time}"
            )
            print(
                f"Processing batch {batch_idx}/{total_batches} topics from {i} to {i+topics_at_a_time}"
            )
            message = prompt_openai(TOPIC_LABEL_PROMPT, topics_to_process[i:i+topics_at_a_time])
            # import pdb; pdb.set_trace()
            out_dict = ast.literal_eval(message)
            logger.info(f"Adding {len(out_dict)} topics to topic_label_dict")
            print(f"Adding {len(out_dict)} topics to topic_label_dict")
            topic_label_dict.update(out_dict)
            end_time = time.time()
            if batch_idx == 1:
                logger.info(f"Time taken for batch {batch_idx}: {(end_time - start_time) * (total_batches)/60} minutes")
                print(f"Time taken for batch {batch_idx}: {(end_time - start_time) * (total_batches)/60} minutes")
        except Exception as e:
            print(f"Error: {e}")
            logger.error(f"Error: {e}")
            if "overload" in str(e).lower():
                logger.info("Overload error. Sleeping for 1 minute")
                print("Overload error. Sleeping for 1 minute")
                time.sleep(60)
                message = prompt_openai(TOPIC_LABEL_PROMPT, topics_to_process[i:i+topics_at_a_time])
                out_dict = ast.literal_eval(message)
                logger.info(f"Adding {len(out_dict)} topics to topic_label_dict")
                print(f"Adding {len(out_dict)} topics to topic_label_dict")
                topic_label_dict.update(out_dict)
                continue
            else:
                out_dict = {top: "no_topic" for top in topics_to_process[i:i+topics_at_a_time]} 
                topic_label_dict.update(out_dict)
                with open(f"{data_dir}/latest_topic_label_errors.tsv", "a") as f:
                    f.write(f"Batch_{batch_idx}\t{str(message)}\n")
                    
        if batch_idx % 10 == 0:
            logger.info(f"Saving topic_label_dict to {topic_label_file}")
            print(f"Saving topic_label_dict to {topic_label_file}")
            json.dump(topic_label_dict, open(topic_label_file.stem + f"_{batch_idx}.json", "w"))

    json.dump(topic_label_dict, open(topic_label_file, "w"))



def check_topic_labels(data_dir="/projects/frame_align/data/annotated"):
    topic_dict = pickle.load(open(f"{data_dir}/unique_topics.pkl", "rb"))
    topic_label_dict = json.load(open(f"{data_dir}/latest_topic_labels.json"))
    try:
        assert len(topic_dict) == len(topic_label_dict)
    except Exception as e:
        logger.error(e)
        # get_topic_labels(data_dir)
    return


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_dir", type=str, default="/projects/frame_align/data/annotated")
    args = argparser.parse_args()
    get_topic_labels(args.data_dir)
    check_topic_labels(args.data_dir)

if __name__ == "__main__":
    main()
