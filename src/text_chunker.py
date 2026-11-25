
from document_loader import load_pdf

def create_chunks(pdf_path):

    # Load PDF content
    document_text = load_pdf(pdf_path)

    chunk_size = 500
    chunk_overlap = 50

    chunks = []
    start = 0

    while start < len(document_text):

        end = start + chunk_size

        chunk = document_text[start:end]

        chunks.append(chunk)

        start = end - chunk_overlap

    return chunks
print("Hello, World!")