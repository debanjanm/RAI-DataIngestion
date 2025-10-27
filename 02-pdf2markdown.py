# import arxiv
# import os
# import csv
# from docling.document_converter import DocumentConverter
# from docling.datamodel.pipeline_options import PdfPipelineOptions
# from docling.datamodel.base_models import InputFormat
# from docling.document_converter import PdfFormatOption


# class ArxivMarkdownGenerator:
#     def __init__(self, save_dir: str = "downloads"):
#         self.save_dir = save_dir

#         pipeline_options = PdfPipelineOptions(
#             do_ocr=False,
#             generate_page_images=False,
#             generate_picture_images=False,
#             do_picture_classification=False,
#             do_picture_description=False,
#         )

#         self.converter = DocumentConverter(
#             format_options={
#                 InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
#             }
#         )

#     def save_markdown(self, content, filename):
#         with open(filename, "w", encoding="utf-8") as f:
#             f.write(content)

#     def generate_markdown(self, topic):
#         topic_dir = os.path.join(self.save_dir, topic.replace(".", "_"))
#         pdf_dir = os.path.join(topic_dir, "pdfs")
#         md_dir = os.path.join(topic_dir, "markdown")
#         os.makedirs(md_dir, exist_ok=True)

#         meta_file = os.path.join(topic_dir, "metadata.csv")
#         if not os.path.exists(meta_file):
#             print(f"⚠️ No metadata found for {topic}. Run PDF downloader first.")
#             return

#         updated_rows = []
#         with open(meta_file, mode="r", encoding="utf-8") as f:
#             reader = csv.reader(f)
#             headers = next(reader)
#             if "md_path" not in headers:
#                 headers.append("md_path")
#             for row in reader:
#                 updated_rows.append(row)

#         for i, row in enumerate(updated_rows):
#             arxiv_id = row[0]
#             pdf_path = row[-1]
#             md_filename = f"{arxiv_id}.md"
#             md_path = os.path.join(md_dir, md_filename)

#             if not os.path.exists(md_path):
#                 try:
#                     conversion = self.converter.convert(pdf_path)
#                     output_md = conversion.document.export_to_markdown()
#                     self.save_markdown(output_md, md_path)
#                     print(f"Markdown snippet saved -> {md_filename}")
#                 except Exception as e:
#                     print(f"⚠️ Conversion failed for {pdf_path}: {e}")
#             else:
#                 print(f"Already exists: {md_filename}")

#             if len(row) < len(headers):
#                 row.append(md_path)
#             else:
#                 row[-1] = md_path

#         with open(meta_file, mode="w", newline="", encoding="utf-8") as f:
#             writer = csv.writer(f)
#             writer.writerow(headers)
#             writer.writerows(updated_rows)

#         print(f"✅ Completed Markdown generation for topic: {topic}")


# if __name__ == "__main__":
#     topics = [
#         "cs.LG", "cs.CV", "cs.CL", "cs.AI", "stat.ML",
#         "astro-ph.CO", "hep-th", "quant-ph", "cond-mat.mtrl-sci", "gr-qc",
#         "math.OC", "math.PR", "eess.SP", "eess.SY", "cs.SY",
#         "q-bio.BM", "q-bio.NC", "q-fin.PR", "econ.EM", "physics.soc-ph"
#     ]

#     generator = ArxivMarkdownGenerator(save_dir="arxiv_data")
#     for topic in topics:
#         generator.generate_markdown(topic)


import arxiv
import os
import csv
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption


class ArxivRecoveryGenerator:
    def __init__(self, save_dir: str = "downloads"):
        self.save_dir = save_dir

        pipeline_options = PdfPipelineOptions(
            do_ocr=False,
            generate_page_images=False,
            generate_picture_images=False,
            do_picture_classification=False,
            do_picture_description=False,
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def save_markdown(self, content, filename):
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)

    def recover_from_pdfs(self, topic):
        topic_dir = os.path.join(self.save_dir, topic.replace(".", "_"))
        pdf_dir = os.path.join(topic_dir, "pdfs")
        md_dir = os.path.join(topic_dir, "markdown")
        os.makedirs(md_dir, exist_ok=True)

        meta_file = os.path.join(topic_dir, "metadata.csv")
        headers = [
            "arxiv_id", "title", "authors", "published",
            "updated", "categories", "abstract", "pdf_path", "md_path"
        ]

        rows = []
        for pdf_file in os.listdir(pdf_dir):
            if pdf_file.endswith(".pdf"):
                arxiv_id = os.path.splitext(pdf_file)[0]
                pdf_path = os.path.join(pdf_dir, pdf_file)
                md_filename = f"{arxiv_id}.md"
                md_path = os.path.join(md_dir, md_filename)

                if not os.path.exists(md_path):
                    try:
                        conversion = self.converter.convert(pdf_path)
                        output_md = conversion.document.export_to_markdown()
                        self.save_markdown(output_md, md_path)
                        print(f"Markdown snippet saved -> {md_filename}")
                    except Exception as e:
                        print(f"⚠️ Conversion failed for {pdf_file}: {e}")
                else:
                    print(f"Already exists: {md_filename}")

                rows.append([
                    arxiv_id,
                    "",  # title unknown
                    "",  # authors unknown
                    "",  # published unknown
                    "",  # updated unknown
                    "",  # categories unknown
                    "",  # abstract unknown
                    pdf_path,
                    md_path
                ])

        with open(meta_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

        print(f"✅ Recovery complete for topic: {topic}. Metadata rebuilt with available PDFs.")


if __name__ == "__main__":
    topics = [
        "cs.LG", "cs.CV", "cs.CL", "cs.AI", "stat.ML",
        "astro-ph.CO", "hep-th", "quant-ph", "cond-mat.mtrl-sci", "gr-qc",
        "math.OC", "math.PR", "eess.SP", "eess.SY", "cs.SY",
        "q-bio.BM", "q-bio.NC", "q-fin.PR", "econ.EM", "physics.soc-ph"
    ]

    # Example usage of recovery class
    recovery = ArxivRecoveryGenerator(save_dir="arxiv_data")
    for topic in topics:
        recovery.recover_from_pdfs(topic)