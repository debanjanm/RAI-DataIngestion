import arxiv
import os
import csv

class ArxivPDFDownloader:
    def __init__(self, topics: list, papers_per_topic: int = 5, save_dir: str = "downloads"):
        """
        Initialize the downloader.

        :param topics: List of arXiv categories or queries (e.g., ["cs.LG", "astro-ph.CO"]).
        :param papers_per_topic: Number of papers per topic.
        :param save_dir: Root directory to save PDFs and metadata.
        """
        self.topics = topics
        self.papers_per_topic = papers_per_topic
        self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)

    def search(self, query, max_results):
        """Search arXiv for a given query."""
        return arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        ).results()

    def download_pdfs(self):
        """
        Download PDFs topic-wise and save metadata.
        """
        for topic in self.topics:
            topic_dir = os.path.join(self.save_dir, topic.replace(".", "_"))
            pdf_dir = os.path.join(topic_dir, "pdfs")
            os.makedirs(pdf_dir, exist_ok=True)

            meta_file = os.path.join(topic_dir, "metadata.csv")

            # If metadata file doesn't exist, create it with headers
            if not os.path.exists(meta_file):
                with open(meta_file, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "arxiv_id", "title", "authors", "published",
                        "updated", "categories", "abstract", "pdf_path"
                    ])

            print(f"\n🔍 Searching {self.papers_per_topic} papers for topic: {topic}")

            with open(meta_file, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                for result in self.search(topic, self.papers_per_topic):
                    arxiv_id = result.get_short_id()
                    filename = f"{arxiv_id}.pdf"
                    filepath = os.path.join(pdf_dir, filename)

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

            print(f"✅ Completed topic: {topic}. PDFs in {pdf_dir}, metadata in {meta_file}")


if __name__ == "__main__":
    # Choose 20 topics, 5 papers each = 100 papers total
    topics = [
        "cs.LG", "cs.CV", "cs.CL", "cs.AI", "stat.ML",
        # "astro-ph.CO", "hep-th", "quant-ph", "cond-mat.mtrl-sci", "gr-qc",
        # "math.OC", "math.PR", "eess.SP", "eess.SY", "cs.SY",
        "q-bio.BM", "q-bio.NC", "q-fin.PR", "econ.EM", "physics.soc-ph"
    ]

    downloader = ArxivPDFDownloader(topics=topics, papers_per_topic=5, save_dir="arxiv_data")
    downloader.download_pdfs()
