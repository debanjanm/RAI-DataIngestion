import os
import csv
from chonkie import RecursiveChunker  # Swap in desired chunker: SentenceChunker, SemanticChunker, etc.

class ArxivUnifiedChunker:
    def __init__(self, save_dir="arxiv_data", chunker=None):
        """
        :param save_dir: Root directory containing topic subfolders with markdown files.
        :param chunker: Chonkie chunker instance; defaults to RecursiveChunker.
        """
        self.save_dir = save_dir
        self.chunker = chunker or RecursiveChunker()
        self.output_file = os.path.join(self.save_dir, "all_chunks.csv")

    def run(self):
        fieldnames = ["topic", "pdf_name", "chunk_id", "chunk_text", "token_count"]
        with open(self.output_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for topic_dir in os.listdir(self.save_dir):
                topic_folder = os.path.join(self.save_dir, topic_dir)
                md_dir = os.path.join(topic_folder, "markdown")
                if not os.path.isdir(md_dir):
                    continue

                topic = topic_dir.replace("_", ".")

                for md_filename in os.listdir(md_dir):
                    if not md_filename.endswith(".md"):
                        continue

                    pdf_name = md_filename[:-3]
                    md_path = os.path.join(md_dir, md_filename)
                    with open(md_path, "r", encoding="utf-8") as f:
                        text = f.read()

                    chunks = self.chunker(text)
                    for idx, chunk in enumerate(chunks, start=1):
                        writer.writerow({
                            "topic": topic,
                            "pdf_name": pdf_name,
                            "chunk_id": f"{pdf_name}_chunk_{idx}",
                            "chunk_text": chunk.text,
                            "token_count": getattr(chunk, "token_count", None),
                        })

        print(f"All chunks saved to {self.output_file}")

if __name__ == "__main__":
    chunker = ArxivUnifiedChunker(save_dir="arxiv_data")
    chunker.run()
