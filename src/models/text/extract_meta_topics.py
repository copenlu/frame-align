import ast
import time
import json
import pickle
import anthropic
import pandas as pd

from pathlib import Path
from pdb import set_trace
import logging

logging.basicConfig(filename='app.log', level=logging.INFO)
logger = logging.getLogger(__name__)

client = anthropic.Anthropic()

def get_bertopic_topics(topics_file=None):
    data_path = Path("data/raw/2023-24/")
    topics = []
    for month_dir in data_path.iterdir():
        df = pd.read_csv(month_dir/"datawithtopics_merged.csv")
        topics.extend(df['auto_topic_label'].unique().tolist())
    topics = list(set(topics))
    pickle.dump(topics, open(topics_file, "wb"))
    return None
    
def main():
    topics_file = Path("data/raw/bertopic_topics.pkl")
    if not topics_file.exists():
        get_bertopic_topics(topics_file)
    topics = pickle.load(open("data/raw/bertopic_topics.pkl", "rb"))
    meta_topic_dict = Path("data/raw/meta_topics.json")
    if meta_topic_dict.exists():
        meta_topic_dict = json.load(meta_topic_dict.open())
    else:
        meta_topic_dict = {}
    topics_to_process = [topic for topic in topics if topic not in meta_topic_dict]
    topics_at_a_time = 200
    for i in range(0,len(topics_to_process), topics_at_a_time):
        try:
            message = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=8000,
                temperature=0,
                system="""
                    Below is a list of underscore separated keywords related to a topic output by BERTopic when analyzing a set of US based news articles.
                    Your job is to come up high level categories for each of them like Business, Politics, Sports etc.
                    If the set of keywords is not in English, is gibberish, or you cannot get a broad topic for them, output no_topic.
                    Output a dictionary with the keywords as keys and broad topics for the corresponding keywords as values, in the same order. Output no other tokens, only the dictionary. List:
                    """,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": str(topics_to_process[i:i+topics_at_a_time])
                            }
                        ]
                    }
                ])
            logger.info(message.content[0].text)
            try:
                out_dict = ast.literal_eval(message.content[0].text)
                logger.info(f"Adding {len(out_dict)} topics to meta_topic_dict")
            except Exception as e:
                logger.error(e)
                logger.error(message.content[0].text)
                print(message.content[0].text, "\n")
            meta_topic_dict.update(out_dict)
            time.sleep(5) # to avoid rate limit
        except Exception as e:
            logger.error(e)
            print(e)
            json.dump(meta_topic_dict, open("data/raw/meta_topics.json", "w"))
            continue
    json.dump(meta_topic_dict, open("data/raw/meta_topics.json", "w"))

if __name__ == "__main__":
    main()