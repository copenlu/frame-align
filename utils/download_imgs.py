import requests
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm

class ImageDownloader:
    def __init__(self, text_dir="data/text", img_dir="data/images"):
        self.textdata_dir = text_dir
        self.imgdata_dir = img_dir
        self.input_fname = "0"
        self.input_txtfile = os.path.join(self.textdata_dir, f"{self.input_fname}.csv")

    def download_images(self, file_id):
        """
        Download images from a GitHub repository.

        Args:
        file_id (str): The ID of the image to download.
        """
        url = f"https://raw.githubusercontent.com/copperwiring/news-image-cleanup/main/1107_images/{file_id}.jpg"
        response = requests.get(url)
        if response.status_code == 200:
            file_path = os.path.join(self.imgdata_dir, f"{file_id}.jpg")
            with open(file_path, 'wb') as f:
                f.write(response.content)

    def delete_files(self):
        """
        Delete all files in the images directory.
        """
        path = Path(self.imgdata_dir)
        for file in path.glob('*.jpg'):
            if file.is_file():
                file.unlink()

    def get_ids(self):
        """
        Read IDs from a CSV file.

        Returns:
        List of image IDs (list)
        """
        df = pd.read_csv(self.input_txtfile)
        return df['uuid'].tolist()

    def run(self):
        # Ensure the image directory exists
        if not os.path.exists(self.imgdata_dir):
            os.makedirs(self.imgdata_dir)

        # Read IDs from CSV
        ids = self.get_ids()

        # Ask for user input on the number of images to download
        num_images = int(input(f"Total images available: {len(ids)}. How many images do you want to download?(Default is 10)") or "10")
        ids = ids[:num_images]

        # Ask if user wants to delete the existing images folder
        if input("Do you want to delete the images folder? (Default is No). Yes/No ").lower() == "yes":
            self.delete_files()

        # Download images
        for file_id in tqdm(ids):
            self.download_images(file_id)

# To run the downloader:
if __name__ == "__main__":
    downloader = ImageDownloader()
    downloader.run()
