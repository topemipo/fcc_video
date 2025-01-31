# Dependecies
import os
from dotenv import load_dotenv
import argparse
import chromadb
from  rag_functions import (
    load_text_files, chunk_casedocs, add_to_chroma_collection,
    augment_query_generated, query_chroma_collection,
    get_file_contents, summarize_text_with_map_reduce,
    generate_response
)

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")


def main(query):
    """Runs the full RAG pipeline given a user query."""
    folder_path = os.getenv("20casedocs")
    if not folder_path or not os.path.exists(folder_path):
        raise FileNotFoundError("Case document folder path is missing or incorrect.")

    print("🔹 Loading case documents...")
    all_texts = load_text_files(os.path.join(folder_path, "*.txt"))
    
    print("🔹 Splitting documents into chunks...")
    chunked_texts = chunk_casedocs(all_texts)
    
    print("🔹 Adding documents to ChromaDB collection...")
    add_to_chroma_collection(chunked_texts, openai_key)
    
    print("🔹 Expanding query with a hypothetical response...")
    hypothetical_answer = augment_query_generated(query)
    joint_query = f"{query} {hypothetical_answer}"
    
    print("🔹 Retrieving relevant case document...")
    retrieved_metadata = query_chroma_collection(chroma_collection, joint_query, n_results=1)
    if not retrieved_metadata:
        print("⚠️ No relevant case found. Try rephrasing your query.")
        return
    
    case_doc = retrieved_metadata[0]['source']
    print(f"📄 Most relevant case: {case_doc}")
    
    print("🔹 Fetching case document content...")
    file_content = get_file_contents(os.path.join(folder_path, "*.txt"), case_doc)
    
    print("🔹 Summarizing case document...")
    summary = summarize_text_with_map_reduce(file_content, max_length=2000)
    
    print("🔹 Generating legal response...")
    final_response = generate_response(query, summary)
    
    print("\n✅ **Legal Response:**\n")
    print(final_response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG system for legal queries.")
    parser.add_argument("--query", type=str, required=True, help="User's legal question.")
    args = parser.parse_args()
    
    main(args.query)
