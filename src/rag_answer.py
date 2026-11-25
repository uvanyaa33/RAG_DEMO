
import os
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Step 1: Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
groq_model = os.getenv("GROQ_MODEL")

if not groq_api_key or not groq_model:
    raise ValueError("Groq API key or model is missing in .env")

# Step 2: Connect to Groq
client = OpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1"
)

# Step 3: Connect to ChromaDB
db_client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = db_client.get_collection(
    name="playwright_documents"
)

# Step 4: Load the same embedding model
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Step 5: Get user question
question = input("\nAsk a question about your PDF: ")

# Step 6: Convert question into embedding
question_embedding = embedding_model.encode(
    question
).tolist()

# Step 7: Retrieve relevant chunks
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)

retrieved_chunks = results["documents"][0]

# Step 8: Combine retrieved chunks
context = "\n\n".join(retrieved_chunks)

print("\nRetrieved document context:")
print(context)

# Step 9: Send context and question to Groq
response = client.chat.completions.create(
    model=groq_model,
    messages=[
        {
            "role": "system",
            "content": (
                "You are a helpful document-based assistant. "
                "Answer using only the provided context. "
                "If the answer is not available in the context, "
                "say that the information was not found in the document. "
                "Do not invent information."
            )
        },
        {
            "role": "user",
            "content": f"""
            Context:
            {context}

            Question:
            {question}

            Answer:
            """
        }
    ],
    temperature=0.2
)

# Step 10: Display final answer
print("\n========== AI GENERATED ANSWER ==========\n")
print(response.choices[0].message.content)
