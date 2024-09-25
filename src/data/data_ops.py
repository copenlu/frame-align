import json
import pandas as pd

from pathlib import Path

def add_topics_to_data(data_path):
    topic_label_dict = json.load(open(data_path/"topic_labels.json", "r"))
    # merged_topic_label_dict = json.load(open(data_path/"merged_topic_labels.json", "r"))
    for month_dir in data_path.iterdir():
        if not month_dir.is_dir():
            continue
        df = pd.read_csv(month_dir/"datawithtopics_merged.csv")
        df['month'] = month_dir.name
        df['claude_topic_label'] = df['auto_topic_label'].apply(lambda x: topic_label_dict[x])
        df.to_csv(month_dir/"datawithtopiclabels.csv", index=False)

def get_combined_topics(data_path):
    combined_topic_df = []
    for month_dir in data_path.iterdir():
        if not month_dir.is_dir():
            continue
        df = pd.read_csv(month_dir/"datawithtopiclabels.csv")
        combined_topic_df.append(df[['id', 'month', 'topic_label']])
    combined_topic_df = pd.concat(combined_topic_df)
    combined_path = data_path/"combined_topic_data.csv"
    combined_topic_df.to_csv(combined_path, index=False)
    return combined_path

def get_samples_from_combined(combined_path):
    topic_samples = []
    combined_topic_df = pd.read_csv(combined_path)
    for _, group in combined_topic_df.groupby('topic_label'):
        topic_samples.append(group.sample(10))
    topic_sample_df = pd.concat(topic_samples)
    print(f"Number of unique topics: {len(topic_sample_df['topic_label'].unique())}")
    print(f"Number of samples: {len(topic_sample_df)}")
    topic_sample_data_df = []
    for month_dir in data_path.iterdir():
        if not month_dir.is_dir():
            continue
        df = pd.read_csv(month_dir/"datawithtopiclabels.csv")
        df = df[df['id'].isin(topic_sample_df['id'])]
        topic_sample_data_df.append(df)
    topic_sample_data_df = pd.concat(topic_sample_data_df)
    print(f"Number of samples with data: {len(topic_sample_data_df)}")
    topic_sample_data_df.to_csv(data_path/"topic_samples.csv", index=False)


if __name__ == '__main__':
    data_path = Path("/projects/copenlu/data/arnav/frame-align/raw/2023-24")
    add_topics_to_data(data_path)
    combined_path = get_combined_topics(data_path)
    get_samples_from_combined(combined_path)