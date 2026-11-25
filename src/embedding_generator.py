
from sentence_transformers import SentenceTransformer
from text_chunker import create_chunks

# Step 1: Load text chunks
pdf_path = "data/Testdata.pdf"

chunks = create_chunks(pdf_path)

print("\nTotal chunks:", len(chunks))

# Step 2: Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Step 3: Convert chunks into embeddings
embeddings = model.encode(
    chunks,
    convert_to_numpy=True,
    show_progress_bar=True
)

# Step 4: Display embedding information
print("\nEmbedding generation completed!")

print("Total embeddings:", len(embeddings))
print("Vector dimensions:", embeddings.shape)

# Display first embedding
print("\nFirst chunk embedding:")
print(embeddings[0])
