import os
import pymupdf 
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def load_pdf(file_path):
    try:
        print("Loading the PDF file...")
        all_text = ""
        doc = pymupdf.open(file_path)
        
        for page_number in range(len(doc)):
            page = doc[page_number]
            text = page.get_text()
            all_text += text + "\n"
        
        print(f"Loaded {len(doc)} pages")
        return all_text
    
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return ""
    
def split(text, chunk_size=500, overlap=50):
    print("Splitting text into chunks...")
    words = text.split()
    chunks = []
    
    i = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunk = " ".join(chunk_words)
        if len(chunk_words) > 20:
            chunks.append(chunk)
        i += chunk_size - overlap
    
    print(f"Created {len(chunks)} chunks!")
    return chunks

def create_embeddings(chunks):
    print("Creating embeddings... this might take a minute")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    
    print("Embeddings created successfully")
    return embeddings, model

def create_vector_store(embeddings):
    print("Creating FAISS vector store...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"Stored {index.ntotal} embeddings in the vector database.")
    return index

def retrieve_context(query, model, index, chunks, top_k=5):
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    return retrieved_chunks

def generate_answer(query, context_chunks):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set. Please set it before running.")

    client = Groq(api_key=api_key)
    
    context = "\n\n---\n\n".join(context_chunks)

    system_prompt = """You are a strictly context-bound AI assistant. Your ONLY job is to answer the user's question based on the provided text.

                    CRITICAL RULES:
                        1. You must read the text provided inside the <context> tags.
                        2. You must formulate your answer relying COMPLETELY and EXCLUSIVELY on that context.
                        3. DO NOT use any outside knowledge, pre-trained facts, or assumptions.
                        4. If the exact answer to the user's question is not contained within the <context> tags, you MUST reply exactly with: "I cannot answer this based on the provided document." Do not try to guess.
                        5. Do not start your response with phrases like "Based on the context..." Just provide the direct answer."""

    user_prompt = f"""<context>\n{context}\n</context>\n\nQuestion: {query}"""

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model="llama-3.1-8b-instant",
        temperature=0.0,
    )
    return response.choices[0].message.content

def main():
    file_path = "context.pdf"
    if os.path.exists(file_path):
        pdf_text = load_pdf(file_path)
        chunks = split(pdf_text)
        embeddings, model = create_embeddings(chunks)
        index = create_vector_store(embeddings)

        os.system('cls' if os.name == 'nt' else 'clear')

        while True:
            query = input("Enter your question: ")
            if query.lower() in ["exit", "quit"]:
                print("Exiting the application.")
                break
            
            context_chunks = retrieve_context(query, model, index, chunks)
            answer = generate_answer(query, context_chunks)
            print(f"\nAnswer: {answer}")
            print("\nRetrieved Context:")
            for i, chunk in enumerate(context_chunks):
                print(f"\nChunk {i+1}:\n{chunk}")
                
    else:
        print(f"File {file_path} does not exist.")

if __name__ == "__main__":
    main()