import pandas as pd
import os

class CSVFileMerger:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        self.output_model = input("Please enter the model name of the file to be merged. Option: mistral/llava/mistral_llava/gpt4: ")
        self.input_model = input("Please enter the model name of the input file. Option: mistral/llava/mistral_llava: ")
        self.input_file = os.path.join(self.output_dir, f"{self.input_model}_100.csv")
        self.output_file = os.path.join(self.output_dir, f"{self.output_model}_100.csv")
        self.merged_csv_file = os.path.join(self.output_dir, f"{self.input_model}_{self.output_model}_100.csv")
        
    def read_csv_files(self):
        """
        Reads two CSV files into pandas DataFrames.
        """
        self.df1 = pd.read_csv(self.input_file)
        self.df2 = pd.read_csv(self.output_file)

    def find_missing_ids(self):
        """
        Finds IDs present in the first DataFrame but missing in the second.
        
        Returns:
        list of missing IDs.
        """
        f1_ids = set(self.df1['id'].tolist())
        f2_ids = set(self.df2['id'].tolist())
        missing_ids = list(f1_ids - f2_ids)
        print(f" f1 has {len(f1_ids)} ids and f2 has {len(f2_ids)} ids but {len(missing_ids)} ids from f1 are missing in f2")
        print(f"Missing ids: {missing_ids}")
        return missing_ids

    def merge_files(self, missing_ids):
        """
        Merges two DataFrames based on 'id' column if there are no missing IDs.
        """
        if len(missing_ids) == 0:
            self.df1.set_index('id', inplace=True)
            self.df2.set_index('id', inplace=True)
            merged_df = self.df2.join(self.df1, how='right').reset_index()
            if 'Unnamed: 0' in merged_df.columns:
                merged_df = merged_df.loc[:, ~merged_df.columns.str.contains('^Unnamed')]
            merged_df.to_csv(self.merged_csv_file, index=False)
            print(f"Output file: {self.merged_csv_file} created successfully")
        else:
            print("Merge was not performed due to missing IDs. Check the following IDs:")
            print(missing_ids)

    def run(self):
        """
        Executes the merging process.
        """
        # Check and create output directory if needed
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self.read_csv_files()
        missing_ids = self.find_missing_ids()

        # Delete the output file if it already exists
        if os.path.exists(self.merged_csv_file):
            os.remove(self.merged_csv_file)
        
        self.merge_files(missing_ids)

# To use the class:
if __name__ == "__main__":
    merger = CSVFileMerger()
    merger.run()
