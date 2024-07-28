"""
This script is used to run LLAVA(Vilcuna) on a set of images and save the results to a CSV file.
"""

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import os, json, csv

base_dir = os.getcwd() + "/hf_scripts/"
os.chdir(base_dir)
print(os.getcwd())

# read file names from img folder. 
image_files = os.listdir("img")
image_files = [os.path.join("img", file) for file in image_files]

# Ask user if they want to process all images. Default is Yes else ask for number of images to process
process_all_images = input("Do you want to process all images? (Default is Yes). If no, we will ask for number of images to process. ") or "Yes"
if process_all_images.lower() == "no":
    num_images = int(input(" Total images available: " + str(len(image_files)) + ". How many images do you want to process? (Default is 10) ") or "10")
    image_files = image_files[:num_images]

else:
    print(f"Processing all {len(image_files)} images.")


# Prepare the list to collect data
data = []

model_path = "liuhaotian/llava-v1.5-7b"
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

for img_file in image_files:

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": img_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()

    result = eval_model(args)

    # Parse the JSON result into a dictionary
    result_dict = json.loads(result)

    # Collect the id (from img_file name or other identifier) and frame result
    img_url = "https://raw.githubusercontent.com/copperwiring/news-image-cleanup/main/1107_images/" + img_file.split('/')[-1]
    data.append({'id': img_file.split('/')[-1].split('.')[0], 'img_url': img_url, 'llava_frame': result_dict['frame'], 'llava_reasoning': result_dict['Reasoning']})

# Define the path to save your CSV file. 
csv_file_path = 'llava_output.csv'

# Delete the file if it exists
if os.path.exists(csv_file_path):
    os.remove(csv_file_path)

# Write data to a CSV file
with open(csv_file_path, mode='w', newline='') as file:
    fieldnames = ['id', 'img_url', 'llava_frame', 'llava_reasoning']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    
    # Write the header
    writer.writeheader()
    
    # Write data rows
    writer.writerows(data)

