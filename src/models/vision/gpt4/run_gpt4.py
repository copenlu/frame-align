# Use gpt 4 api to answer the question about set of images (in loop for each image) and save the results to a CSV file.
from openai import OpenAI
import os, json, csv
import base64
import requests
from tqdm import tqdm as tdqm 

# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")

base_dir = os.getcwd() + "/gpt4/"
os.chdir(base_dir)

prompt = """
What is the frame of the image given that frames are defined as follows:
Economic: costs, benefits, or other financial implications.
Capacity and resources: availability of physical, human or financial resources, and capacity of current systems.
Morality: religious or ethical implications.
Fairness and equality: balance or distribution of rights, responsibilities, and resources.
Legality, constitutionality and jurisprudence: rights, freedoms, and authority of individuals, corporations, and government.
Policy prescription and evaluation: discussion of specific policies aimed at addressing problems.
Crime and punishment: effectiveness and implications of laws and their enforcement.
Security and defense: threats to welfare of the individual, community, or nation.
Health and safety: health care, sanitation, public safety
Quality of life: threats and opportunities for the individual's wealth, happiness, and well-being.
Cultural identity: traditions, customs, or values of a social group in relation to a policy issue.
Public opinion: attitudes and opinions of the general public, including polling and demographics.
Political: considerations related to politics and politicians, including lobbying, elections, and attempts to sway voters.
External regulation and reputation: international reputation or foreign policy of a country or
Other: any coherent group of frames not covered by the above categories.

Given these definitions, output your response in the format

'{"frame": "<answer>", "Reasoning": "<reasoning>"}'
"""

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


# read file names from img folder.
image_files = os.listdir("../hf_scripts/img")
image_files = [os.path.join("../hf_scripts/img", file) for file in image_files]

# Prepare the list to collect data
data = []

for image_path in tdqm(image_files):

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    payload = {
    "model": "gpt-4-turbo",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": prompt
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    # print(response.json())
    result_dict = response.json()

    if 'choices' in result_dict:
        # Extract the message content, which is a JSON string
        message_content = result_dict['choices'][0]['message']['content']

        # Parse the JSON string to get an actual dictionary
        content_dict = json.loads(message_content)

        # Extract 'frame' and 'Reasoning' from the parsed dictionary
        frame = content_dict['frame']
        reasoning = content_dict['Reasoning']
        image_id = image_path.split('/')[-1].split('.')[0]
        img_url = "https://raw.githubusercontent.com/copperwiring/news-image-cleanup/main/1107_images/" + image_id + ".jpg"

        data.append({'id': image_id, 'img_url': img_url, 'gpt4_frame': frame, 'gpt4_reasoning': reasoning})

    else:
        print("The key 'choices' is not present in the response.")
        print(result_dict)
        continue

# # save data to json file
# with open('gpt4_output.json', 'w') as f:
#     json.dump(data, f)
    
# Save this data to a CSV file
csv_file_path = 'gpt4_output.csv'

# Delete the file if it exists
if os.path.exists(csv_file_path):
    os.remove(csv_file_path)

# Write data to a CSV file
with open(csv_file_path, mode='w', newline='') as file:
    fieldnames = ['id', 'img_url', 'gpt4_frame', 'gpt4_reasoning']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    
    # Write the header
    writer.writeheader()
    
    # Write data rows
    writer.writerows(data)


