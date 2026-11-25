
from pypdf import PdfReader

def load_pdf(pdf_path):

    reader = PdfReader(pdf_path)

    document_text = ""

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if text:
            document_text += text + "\n"

        print(f"Page {page_number} extracted successfully")

    return document_text

