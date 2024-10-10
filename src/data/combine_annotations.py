import os
import json
import shutil
import pandas as pd

from pathlib import Path

# Copy all annotations from different folders into a single analysis folder
def copy_annotations(collection_path: Path):
    text_collection_path = collection_path / 'text'
    vision_collection_path = collection_path / 'vision'
    if not text_collection_path.exists():
        text_collection_path.mkdir(parents=True)
    if not vision_collection_path.exists():
        vision_collection_path.mkdir(parents=True)

    # Vision annotations
    vision_dir_set1 = "/projects/frame_align/data/annotated/vision"
    vision_files_set1 = [os.path.join(vision_dir_set1, f) for f in os.listdir(vision_dir_set1) if f.endswith('.jsonl')]

    vision_dir_set2 = "/projects/frame_align/data/img_data/"
    vision_dir_set2list = [f for f in os.listdir(vision_dir_set2) if os.path.isdir(os.path.join(vision_dir_set2, f))]

    vision_files_set2 = []
    for directory in vision_dir_set2list:
        file_dir = os.path.join(vision_dir_set2, directory, "annotated/vision")
        vision_files_set2.extend([os.path.join(file_dir, f) for f in os.listdir(file_dir) if f.endswith(f'{directory}_llava.jsonl')])
    vision_files_set2.sort()

    vision_files = vision_files_set1 + vision_files_set2

    month_count = {}
    for file in vision_files:
        file_path = Path(file)
        month = file_path.stem.split('_')[0]
        if month not in month_count:
            month_count[month] = 0
        new_file_path = vision_collection_path/f"{month}_{month_count[month]}.jsonl"
        month_count[month] += 1
        shutil.copy(file, new_file_path)

    # Text annotations
    text_dir_set1 = "/projects/frame_align/data/annotated/text"
    text_dir_set2 = "/projects/frame_align/data/annotated/fixed/part1/"
    text_dir_set3 = "/projects/frame_align/data/annotated/fixed/part2/"
    text_dir_set4 = "/projects/frame_align/data/arnav_part1/"
    text_files_set1 = [os.path.join(text_dir_set1, f) for f in os.listdir(text_dir_set1) if f.endswith('.jsonl')]
    text_files_set2 = [os.path.join(text_dir_set2, f) for f in os.listdir(text_dir_set2) if f.endswith('.jsonl')]
    text_files_set3 = [os.path.join(text_dir_set3, f) for f in os.listdir(text_dir_set3) if f.endswith('.jsonl')]
    text_files_set4 = [os.path.join(text_dir_set4, f) for f in os.listdir(text_dir_set4) if f.endswith('.jsonl')]

    text_files = text_files_set1 + text_files_set2 + text_files_set3 + text_files_set4

    # Copy and save text files
    month_count = {}
    for file in text_files:
        file_path = Path(file)
        month = file_path.stem.split('_')[0]
        if month not in month_count:
            month_count[month] = 0
        new_file_path = text_collection_path/f"{month}_{month_count[month]}.jsonl"
        month_count[month] += 1
        shutil.copy(file, new_file_path)
    
# Read all text and vision annotations, merge common rows into a single CSV file
def combine_annotations(annotation_path: Path):
    text_analysis_path = annotation_path / 'text'
    vision_analysis_path = annotation_path / 'vision'

    all_text_data = []
    all_vision_data = []

    for file in text_analysis_path.iterdir():
        if file.suffix != '.jsonl':
            continue
        try:
            df = pd.read_json(file, lines=True)
        except:
            file_content = []
            for line in open(file, 'r'):
                try:
                    file_content.append(json.loads(line))
                except:
                    pass
            df = pd.DataFrame(file_content)
        all_text_data.append(df)
    all_text_data = pd.concat(all_text_data)

    for file in vision_analysis_path.iterdir():
        if file.suffix != '.jsonl':
            continue
        try:
            df = pd.read_json(file, lines=True)
        except:
            file_content = []
            for line in open(file, 'r'):
                try:
                    file_content.append(json.loads(line))
                except:
                    pass
            df = pd.DataFrame(file_content)
        all_vision_data.append(df)
    all_vision_data = pd.concat(all_vision_data)

    print(f"Raw concatenated. Text data shape: {all_text_data.shape}, Vision data shape: {all_vision_data.shape}")

    text_columns = ['topic', 'topic_justification', 'summary', 'entity_name',
        'justification_entity_sentiment', 'entity_sentiment',
        'frame_justification', 'frame_id', 'frame_name', 'tone',
        'justification_tone', 'issue_frame', 'issue_frame_justification', 'id']

    vision_columns = ['caption', 'main-actor', 'sentiment', 'sentiment-justification',
        'facial-expression', 'facial-expression-justification', 'perceivable-gender',
        'perceivable-gender-justification', 'symbolic-object', 'symbolic-meaning',
        'symbolic-meaning-explanation', 'frame-id', 'frame-name', 'frame-justification',
        'image_url', 'title', 'uuid']

    all_text_data.drop_duplicates(subset=['id'], inplace=True)
    all_vision_data.drop_duplicates(subset=['uuid'], inplace=True)

    all_text_data = all_text_data[text_columns]
    all_text_data.columns = ['text_'+c for c in all_text_data.columns]
    all_vision_data = all_vision_data[vision_columns]
    all_vision_data.columns = ['vision_'+c for c in all_vision_data.columns]

    # After deduplication and removing columns
    print(f"Post deduplication and removing cols. Text data shape: {all_text_data.shape}, Vision data shape: {all_vision_data.shape}")

    combined_data = all_text_data.merge(all_vision_data, left_on='text_id', right_on='vision_uuid', how='inner')

    print("Combined data shape:", combined_data.shape)
    combined_data.to_csv(annotation_path / 'combined_annotations.csv', index=False)

def main():
    collection_path = Path("/projects/frame_align/data/annotated/consolidated/")
    copy_annotations(collection_path)
    combine_annotations(collection_path)

if __name__ == "__main__":
    main()