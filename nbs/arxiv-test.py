import arxiv
import os
import csv
from datetime import datetime

class ArxivPDFDownloader:
    def __init__(self, query: str, max_results: int = 10, save_dir: str = "downloads"):
        """
        Initialize the downloader.

        :param query: Search query or category (e.g., "cs.LG" or "machine learning").
        :param max_results: Maximum number of papers to download.
        :param save_dir: Directory to save downloaded PDFs and metadata.
        """
        self.query = query
        self.max_results = max_results
        self.save_dir = save_dir
        self.pdf_dir = os.path.join(save_dir, "pdfs")
        self.meta_file = os.path.join(save_dir, "metadata.csv")

        os.makedirs(self.pdf_dir, exist_ok=True)

        # If metadata file doesn't exist, create it with headers
        if not os.path.exists(self.meta_file):
            with open(self.meta_file, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "arxiv_id", "title", "authors", "published",
                    "updated", "categories", "abstract", "pdf_path"
                ])

    def search(self):
        """
        Search arXiv for the query.
        """
        return arxiv.Search(
            query=self.query,
            max_results=self.max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        ).results()

    def download_pdfs(self):
        """
        Download PDFs and save metadata to CSV.
        """
        with open(self.meta_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            for result in self.search():
                arxiv_id = result.get_short_id()
                filename = f"{arxiv_id}.pdf"
                filepath = os.path.join(self.pdf_dir, filename)

                # Download PDF if not already present
                if not os.path.exists(filepath):
                    print(f"Downloading {result.title} -> {filename}")
                    result.download_pdf(filename=filepath)
                else:
                    print(f"Already exists: {filename}")

                # Collect metadata
                metadata = [
                    arxiv_id,
                    result.title.strip().replace("\n", " "),
                    "; ".join([a.name for a in result.authors]),
                    result.published.strftime("%Y-%m-%d"),
                    result.updated.strftime("%Y-%m-%d"),
                    " ".join(result.categories),
                    result.summary.strip().replace("\n", " "),
                    filepath
                ]

                writer.writerow(metadata)

        print(f"\n✅ Download complete. PDFs in {self.pdf_dir}, metadata in {self.meta_file}")

if __name__ == "__main__":
    downloader = ArxivPDFDownloader(query="cs.LG", max_results=5, save_dir="arxiv_data")
    downloader.download_pdfs()
