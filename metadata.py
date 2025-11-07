import os
import csv
import pandas as pd

class ArxivMetadataCombiner:
    def __init__(self, save_dir: str = "arxiv_data", output_filename: str = "master_metadata.csv"):
        """
        :param save_dir: Root directory containing topic subfolders each with metadata.csv
        :param output_filename: Name of the combined metadata CSV file to write.
        """
        self.save_dir = save_dir
        self.output_path = os.path.join(save_dir, output_filename)

    def run(self):
        all_dfs = []
        # iterate over subfolders (topics)
        for topic_folder in os.listdir(self.save_dir):
            topic_path = os.path.join(self.save_dir, topic_folder)
            if not os.path.isdir(topic_path):
                continue
            meta_file = os.path.join(topic_path, "metadata.csv")
            if os.path.exists(meta_file):
                try:
                    df = pd.read_csv(meta_file, encoding="utf-8")
                    # add topic column (convert folder name back to dot form if needed)
                    topic_name = topic_folder.replace("_", ".")
                    df["topic"] = topic_name
                    all_dfs.append(df)
                    print(f"Loaded metadata for topic: {topic_name}, {len(df)} rows")
                except Exception as e:
                    print(f"⚠️ Could not read {meta_file}: {e}")
            else:
                print(f"⚠️ No metadata.csv found in {topic_path}")

        if not all_dfs:
            print("❌ No metadata files found. Nothing to combine.")
            return

        # concatenate all dataframes
        combined = pd.concat(all_dfs, ignore_index=True, sort=False)
        # write to single CSV
        combined.to_csv(self.output_path, index=False, encoding="utf-8")
        print(f"✅ Combined metadata saved to {self.output_path}, total rows = {len(combined)}")

if __name__ == "__main__":
    combiner = ArxivMetadataCombiner(save_dir="arxiv_data", output_filename="master_metadata.csv")
    combiner.run()
