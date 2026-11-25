
import chromadb
from sentence_transformers import SentenceTransformer
from text_chunker import create_chunks

# Step 1: Load PDF chunks
pdf_path = "data/Testdata.pdf"

chunks = create_chunks(pdf_path)

# Step 2: Load embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Step 3: Generate embeddings
embeddings = model.encode(chunks).tolist()

# Step 4: Create persistent ChromaDB client
client = chromadb.PersistentClient(
    path="chroma_db"
)

# Step 5: Create collection
collection = client.get_or_create_collection(
    name="playwright_documents"
)

# Step 6: Store chunks and embeddings
collection.upsert(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings
)

print("Documents stored successfully in ChromaDB!")

print("Total stored chunks:", collection.count())
