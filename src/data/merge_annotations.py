
import json
import pickle
import pandas as pd

from pathlib import Path

month_set_dict = pickle.load(open("/projects/frame_align/data/text_month_set_dict.pkl", "rb"))

frame_short_dict = {'economic': 'economic',
 'capacity and resources': 'cap&res',
 'morality': 'morality',
 'fairness and equality': 'fairness',
 'legality, constitutionality and jurisprudence': 'legality',
 'policy prescription and evaluation': 'policy',
 'crime and punishment': 'crime',
 'security and defense': 'security',
 'health and safety': 'health',
 'quality of life': 'quality_life',
 'cultural identity': 'culture',
 'public opinion': 'public_op',
 'political': 'political',
 'external regulation and reputation': 'regulation',
 'other': 'other'}


frame_fix_dict = {'legality, constitutionality and jurispudence': 'legality, constitutionality and jurisprudence', 'safety and health':'health and safety'}


orig_col_names = ['topic', 'topic_justification', 'entity-name', 'entity-gender',
       'sentiment', 'sentiment-reason', 'frames-list',
       'frames-list-justification', 'issue_frame', 'issue_frame_justification',
       'uuid', 'title', 'vision_frames_frames-list', 'vision_frames_reason',
       'entity_entity-name', 'entity_entity-gender', 'entity_sentiment',
       'entity_sentiment-reason', 'image_url']
clean_col_names = ['text-topic', 'text-topic-exp', 'text-entity-name', 'text-entity-gender','text-entity-sentiment', 'text-entity-sentiment-exp', 'text-generic-frame',
       'text-generic-frame-exp', 'text-issue-frame', 'text-issue-frame-exp',
       'uuid', 'title', 'img-generic-frame', 'img-frame-exp',
       'img-entity-name', 'img-entity-gender', 'img-entity-sentiment',
       'img-entity-sentiment-exp', 'image-url']


raw_path = Path('/projects/frame_align/data/annotated/merged/raw')

nan_count_text = 0
nan_count_vision = 0
for month_file in raw_path.iterdir():
    month_df = pd.read_json(month_file, orient='records', lines=True)
    month_df = month_df[orig_col_names]
    month_df.columns = clean_col_names
    nan_count_text += month_df['text-generic-frame'].isna().sum()
    nan_count_vision += month_df['img-generic-frame'].isna().sum()
print("Text nan count: ", nan_count_text)
print("Vision nan count: ", nan_count_vision)

# ### Adding Topics and Political Leaning

topic_mapping = json.load(open("/projects/frame_align/data/annotated/topics/topic_labels.json", "r"))


political_leaning = {"left" : ['alternet.org', 'editor.cnn.com', 'democracynow.org', 'dailybeast.com', 'huffpost.com', 'theintercept.com','jacobin.com', 'motherjones.com', 'newyorker.com', 'slate.com',   'msnbc.com', 'vox.com'],
'left_lean' : ['abcnews.com','apnews.com', 'theatlantic.com', 'bloomberg.com', 'cbsnews.com', 'insider.com', 'nbcnews.com', 'thenytimes.com', 'npr.com', 'politico.com', 'propublica.org', 'time.com', 'washingtonpost.com', 'yahoonews.com','usatoday.com', 'theguardian.com'],
"center" : ['axios.com', 'bbc.com', 'forbes.com', 'newsweek.com', 'reuters.com', 'realclearpolitics.com', 'thehill.com'],
"right_lean" : ['thedispatch.com', 'theepochtimes.com', 'foxbusiness.com', 'ijr.com', 'nypost.com', 'thepostmillennial.com', 'washingtonexaminer.com', 'washingtontimes.com'],
"right" : ['theamericanconservative.com', 'theamericanspectator.com', 'breitbart.com', 'dailycaller.com', 'dailywire.com', 'dailymail.com', 'foxnews.com', 'newsmax.com', 'oann.com', 'thefederalist.com']}
all_hosts = [values for k, v in political_leaning.items() for values in v]
host_mapping = {host: k for k, v in political_leaning.items() for host in v}


orig_columns_to_keep = ['id', 'authors', 'date_publish', 'description','language', 'maintext', 'source_domain','url']


for month_file in raw_path.iterdir():
    month_df = pd.read_json(month_file, orient='records', lines=True)
    month_name = month_file.stem[12:]
    print(f"{month_name}, OG length: ", len(month_df), end=",")
    # Clean column names
    month_df = month_df[orig_col_names]
    month_df.columns = clean_col_names
    # Add GPT topics
    month_df['gpt-topic'] = month_df['text-topic'].apply(lambda x: topic_mapping[x] if x in topic_mapping else None)
    orig_df = pd.read_csv(f"/projects/frame_align/data/raw/text/{month_name}/datawithtopics_merged.csv")
    merged_df = month_df.merge(orig_df[orig_columns_to_keep], left_on='uuid', right_on='id', how='left')
    merged_df.drop(columns=['id'], inplace=True)
    # Only keeping english articles
    merged_df = merged_df[merged_df['language'] == 'en']
    print(f" Post non-english: {len(merged_df)}", end=",")
    # Remove bbc and dailymail
    merged_df = merged_df[~merged_df['source_domain'].isin(['www.bbc.com', 'www.dailymail.com'])]
    print(f" Post bbc filter: {len(merged_df)}")
    # Add political leaning
    political_leaning = []
    for row_no, row in merged_df.iterrows():
        for host in all_hosts:
            if host in row['source_domain']:
                political_leaning.append(host_mapping[host])
                break
        else:
            political_leaning.append(None)
    assert len(political_leaning) == len(merged_df)
    merged_df['political_leaning'] = political_leaning
    merged_df.to_json(f"/projects/frame_align/data/annotated/merged/processed/json/{month_file.stem}.jsonl", orient="records", lines=True)
    merged_df.to_csv(f"/projects/frame_align/data/annotated/merged/processed/csv/{month_file.stem}.csv", index=False)


# Total number of annotations
total_annotations = 0
combined_annotations = []
for month in month_set_dict.keys():
    month_df = pd.read_json(f"/projects/frame_align/data/annotated/merged/processed/json/merged_anno_{month}.jsonl", lines=True)
    combined_annotations.append(month_df)
    total_annotations += len(month_df)
print("Total processed annotations: ", total_annotations)
combined_annotations = pd.concat(combined_annotations)
combined_annotations.to_json("/projects/frame_align/data/annotated/merged/processed/combined_annotations.jsonl", orient="records", lines=True)
combined_annotations.to_csv("/projects/frame_align/data/annotated/merged/processed/combined_annotations.csv", index=False)


processed_path = Path("/projects/frame_align/data/annotated/merged/processed/json")
for month_file in processed_path.iterdir():
    month_df = pd.read_json(month_file, orient='records', lines=True)
    month_name = month_file.stem[12:]
    print(f"{month_name}, OG length: ", len(month_df), end=",")    
    # Drop NaN framing values
    month_df.dropna(subset=['text-generic-frame', 'img-generic-frame'], inplace=True)
    print(f" Post NaN removal: {len(month_df)}", end=",")
    # Fix frame names
    month_df['text-generic-frame'] = month_df['text-generic-frame'].apply(lambda frame_list: [frame_fix_dict[frame.lower()] if frame.lower() in frame_fix_dict else frame for frame in frame_list])
    # Shorten frame names
    clean_frame_preds_text = month_df['text-generic-frame'].apply(lambda frame_list: [frame_short_dict[frame.lower()] for frame in frame_list if frame.lower() in frame_short_dict])
    clean_frame_preds_text_len = clean_frame_preds_text.apply(len)
    clean_frame_preds_img = month_df['img-generic-frame'].apply(lambda frame_list: [frame_short_dict[frame.lower()] for frame in frame_list if frame.lower() in frame_short_dict])
    clean_frame_preds_img_len = clean_frame_preds_img.apply(len)
    # Filter out zero length frames
    month_df['text-generic-frame'] = clean_frame_preds_text
    month_df['img-generic-frame'] = clean_frame_preds_img
    month_df['img-generic-frame-len'] = clean_frame_preds_img_len
    month_df['text-generic-frame-len'] = clean_frame_preds_text_len
    month_df = month_df[(month_df['img-generic-frame-len'] > 0) & (month_df['text-generic-frame-len'] > 0) & (month_df['text-generic-frame-len'] <= 5)]
    print(f" Invalid removal: {len(month_df)}", end=",")
    month_df = month_df[month_df['gpt-topic'] != "Sports"]
    print(f" Post sports: {len(month_df)}")
    month_df.to_json(f"/projects/frame_align/data/annotated/merged/analysis/json/{month_file.stem}.jsonl", orient="records", lines=True)
    month_df.to_csv(f"/projects/frame_align/data/annotated/merged/analysis/csv/{month_file.stem}.csv", index=False)


# Total number of annotations
total_annotations = 0
combined_annotations = []
for month in month_set_dict.keys():
    month_annotations = pd.read_json(f"/projects/frame_align/data/annotated/merged/analysis/json/merged_anno_{month}.jsonl", lines=True)
    combined_annotations.append(month_annotations)
    total_annotations += len(month_annotations)
print("Total analysis annotations: ", total_annotations)
combined_annotations = pd.concat(combined_annotations)
combined_annotations.to_json("/projects/frame_align/data/annotated/merged/analysis/combined_annotations.jsonl", orient="records", lines=True)
combined_annotations.to_csv("/projects/frame_align/data/annotated/merged/analysis/combined_annotations.csv", index=False)


processed_path = Path("/projects/frame_align/data/annotated/merged/processed/json")
for month_file in processed_path.iterdir():
    month_df = pd.read_json(month_file, orient='records', lines=True)
    month_name = month_file.stem[12:]
    month_df.dropna(subset=['text-generic-frame', 'img-generic-frame'], inplace=True)
    # Shorten frame names
    clean_frame_preds_text = month_df['text-generic-frame'].apply(lambda frame_list: [frame_short_dict[frame.lower()] for frame in frame_list if frame.lower() in frame_short_dict])
    clean_frame_preds_text_len = clean_frame_preds_text.apply(len)
    clean_frame_preds_img = month_df['img-generic-frame'].apply(lambda frame_list: [frame_short_dict[frame.lower()] for frame in frame_list if frame.lower() in frame_short_dict])
    clean_frame_preds_img_len = clean_frame_preds_img.apply(len)
    # Filter out zero length frames, save to a separate file
    bad_preds = month_df[(clean_frame_preds_text_len == 0) | (clean_frame_preds_img_len == 0) | (clean_frame_preds_text_len > 5)]
    bad_preds.to_json(f"/projects/frame_align/data/annotated/merged/bad_preds/{month_name}_badpreds.jsonl", orient="records", lines=True)





