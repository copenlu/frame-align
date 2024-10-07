import json
import pandas as pd

from pathlib import Path

analysis_path = Path('/projects/frame_align/data/annotated/analysis')
text_analysis_path = analysis_path / 'text'
vision_analysis_path = analysis_path / 'vision'

all_text_data = []
all_vision_data = []

for file in text_analysis_path.iterdir():
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

print(f"Text data shape: {all_text_data.shape}, Vision data shape: {all_vision_data.shape}")

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
print(f"Text data shape: {all_text_data.shape}, Vision data shape: {all_vision_data.shape}")

combined_data = all_text_data.merge(all_vision_data, left_on='text_id', right_on='vision_uuid', how='inner')

print("Combined data shape:", combined_data.shape)
combined_data.to_csv(analysis_path / 'combined_annotations.csv', index=False)



