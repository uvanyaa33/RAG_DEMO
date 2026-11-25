
import chromadb
from sentence_transformers import SentenceTransformer

# Step 1: Load embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Step 2: Connect to existing ChromaDB
client = chromadb.PersistentClient(
    path="chroma_db"
)

# Step 3: Get existing collection
collection = client.get_collection(
    name="playwright_documents"
)

# Step 4: Get user question
user_question = input("\nAsk your question: ")

# Step 5: Convert question into embedding
question_embedding = model.encode(
    user_question
).tolist()

# Step 6: Search ChromaDB
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)

# Step 7: Display retrieved chunks
print("\n========== RETRIEVED INFORMATION ==========\n")

for index, document in enumerate(results["documents"][0], start=1):

    print(f"\nResult {index}:")
    print(document)

print("\nRetrieval completed!")
