# from docling.document_converter import DocumentConverter

# source = "https://arxiv.org/pdf/2408.09869"  # document per local path or URL
# converter = DocumentConverter()
# result = converter.convert(source)

# markdown_text = result.document.export_to_markdown()

# print(markdown_text)

# save_markdown(markdown_text, "research-paper.md")

def save_markdown(content, filename="output.md"):
    """
    Save text content to a Markdown (.md) file.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)


from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption

# Configure pipeline to ignore all image-based processing
pipeline_options = PdfPipelineOptions(
    do_ocr=False,
    generate_page_images=False,
    generate_picture_images=False,
    do_picture_classification=False,
    do_picture_description=False,
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

result = converter.convert("https://arxiv.org/pdf/2408.09869")
output_md = result.document.export_to_markdown()
print(output_md)

save_markdown(output_md, "research-paper.md")