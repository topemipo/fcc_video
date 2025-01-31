import os
import glob
import tiktoken
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from openai import OpenAI

import re

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)




# load text files
def load_text_files(folder_path):
    """Loads all text files from a given directory."""
    all_text = []
    for filepath in glob.glob(folder_path):
        if os.path.isfile(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    all_text.append({"text": text, "source": os.path.basename(filepath)})

    return all_text



# chunking
def chunk_casedocs(all_texts, chunk_size=8100, chunk_overlap=500):
    """
    Splits a list of text entries into chunks using a token-aware text splitter.

    Args:
        all_texts (list of dict): List of dictionaries containing "text" and "source" keys.
        chunk_size (int): Maximum size of each chunk in tokens. Default is 8100.
        chunk_overlap (int): Overlap between chunks in tokens. Default is 500.

    Returns:
        list of dict: List of chunked texts with metadata, including source and token count.
    """
    # Initialize OpenAI tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")  # Correct for text-embedding-ada-002

    # Configure the text splitter
    optimized_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],  # Prioritize document structure
        chunk_size=chunk_size,  # Adjustable chunk size
        chunk_overlap=chunk_overlap,  # Adjustable overlap
        length_function=lambda text: len(tokenizer.encode(text)),  # Exact token count
        is_separator_regex=False
    )

    chunked_texts = []
    for doc_index, entry in enumerate(all_texts):
        # Split the text into chunks
        chunks = optimized_splitter.split_text(entry["text"])
        
        # Add chunks with metadata
        for chunk_index, chunk in enumerate(chunks):
            chunked_texts.append({
                "id": f"doc_{doc_index}_chunk_{chunk_index}",
                "text": chunk,
                "metadata": {
                    "source": entry["source"],
                    "token_count": len(tokenizer.encode(chunk))  # Optional but useful
                }
            })
    
    return chunked_texts


# embedding
def add_to_chroma_collection(chunked_texts, openai_key, collection_name="case-docs-collection"):
    """
    Adds chunked texts to a Chroma collection with OpenAI embeddings.

    Args:
        chunked_texts (list of dict): List of dictionaries containing "id", "text", and "metadata".
        openai_key (str): OpenAI API key for the embedding function.
        collection_name (str): Name of the Chroma collection. Default is "case-docs-collection".

    Returns:
        chromadb.Collection: The Chroma collection with the added documents.
    """
    # Initialize OpenAI embedding function
    embedding_function = OpenAIEmbeddingFunction(
        api_key=openai_key,
        model_name="text-embedding-ada-002"
    )

    # Initialize Chroma client and create/load collection
    chroma_client = chromadb.Client()
    chroma_collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=embedding_function  # Attach the OpenAI embedder
    )

    # Prepare data for Chroma
    ids = [item["id"] for item in chunked_texts]
    documents = [item["text"] for item in chunked_texts]
    metadatas = [item["metadata"] for item in chunked_texts]

    # Add to Chroma collection
    chroma_collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    return chroma_collection


# generate hypothetical answer
def augment_query_generated(user_query, model="gpt-3.5-turbo"):
    system_prompt = """You are a helpful expert research assistant. 
    Provide a plausible example answer to the user's query as if you found it in a case document."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content



# retrieval
def query_chroma_collection(chroma_collection, query_text, n_results=1):
    """
    Queries a Chroma collection and returns metadata with relevance scores.

    Args:
        chroma_collection (chromadb.Collection): The Chroma collection to query.
        query_text (str): The query text (original query + hypothetical answer) to search for.
        n_results (int): Number of results to return. Default is 1.

    Returns:
        list of dict: Retrieved metadata with relevance scores included.
    """
    # Query the Chroma collection
    results = chroma_collection.query(
        query_texts=[query_text],
        n_results=n_results,
        include=["documents", "embeddings", "metadatas", "distances"]  # Include distances explicitly
    )

    # Extract metadata and distances from the results
    retrieved_metadata = results["metadatas"][0]
    retrieved_distances = results["distances"][0]

    # Compute relevance scores from distances
    relevance_scores = [1 / (1 + distance) for distance in retrieved_distances]

    # Add relevance scores to the metadata
    for metadata, relevance_score in zip(retrieved_metadata, relevance_scores):
        metadata["relevance_score"] = relevance_score

    return retrieved_metadata


# get most relevant casedoc from folder
def get_file_contents(pattern, target_filename):
    """
    pattern: A glob pattern, e.g. '/path/to/*.txt'
    target_filename: A filename we want to match in those results
    """
    # Expand the pattern into all matching file paths
    matching_files = glob.glob(pattern)  # returns a list of matching paths

    for fpath in matching_files:
        # Get the base name of the file (e.g., "C2.txt") and compare 
        if os.path.basename(fpath) == target_filename:
            with open(fpath, 'r', encoding='utf-8') as file:
                return file.read()

    # If we exit the loop, the file wasn't found
    raise FileNotFoundError(f"File '{target_filename}' not found in the pattern '{pattern}'.")


# summarise most relevant case document
def summarize_text_with_map_reduce(file_content: str, max_length: int = 100) -> str:
    """
    Summarize long text using the Map-Reduce method, preserving specific details before a list starts.

    Args:
        file_content (str): The long text to summarize.
        max_length (int): The maximum length of the summary in tokens or words (approximation).

    Returns:
        str: The summarized text, preserving details before the first list starts.
    """
    # Step 1: Extract the section to preserve (everything before the first list starts)
    match = re.search(r"^(.*?)(\n\s*1\.)", file_content, re.DOTALL)
    if match:
        preserved_section = match.group(1).strip()
        remaining_content = file_content[len(preserved_section):].strip()
    else:
        # If no list is found, treat the whole text as the preserved section
        preserved_section = file_content.strip()
        remaining_content = ""

    # Step 2: Split the remaining text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Adjust chunk size as needed
        chunk_overlap=50  # Adjust overlap to preserve context
    )
    chunks = text_splitter.split_text(remaining_content)

    # Step 3: Convert chunks into a list of Document objects
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Step 4: Initialize the LLM
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    # Step 5: Load the Map-Reduce summarization chain
    chain = load_summarize_chain(llm, chain_type="map_reduce")

    # Step 6: Invoke the chain with the documents and adjust the summary length
    result = chain.invoke({"input_documents": documents, "length": max_length})

    # Step 7: Combine the preserved section with the generated summary
    summary = preserved_section + "\n\n" + result["output_text"]

    return summary



# generate response
def generate_response(question, context_data):
    """
    Generate a response based on the provided legal context (as a string).
    
    Args:
        question (str): The user's legal question.
        context_data (str): The full legal text to be used as context.

    Returns:
        str: A generated response summarizing relevant legal cases.
    """

    # Construct the prompt using the full context_data string
    prompt = f"""You are a legal assistant designed to help users understand their legal situations by retrieving and summarizing relevant cases. Follow these steps STRICTLY:
    
    1. **Sympathize with the user** (1-2 sentences):
       - Acknowledge their situation with empathy (e.g., "I’m sorry to hear...", "This sounds difficult...").
    
    2. **Retrieve and summarize a case** from the knowledge base below:
    {context_data}
       - Format:
         **Case Name**: [Exact case title]
         **Introduction**: [1-2 sentence overview: who was involved and the core issue]
         **Details**: [Key facts/events in chronological order]
         **Verdict**: [Court decision + outcomes like damages or policy changes]

    3. **Next Steps** (3-4 bullet points):
       - Practical actions tied to the case (e.g., "Save emails from [date range]")
       - Resources (e.g., "Contact [Agency Name] within [timeframe]")
    
    Tone Rules:
    - Professional but compassionate
    - Zero legal jargon (avoid terms like "plaintiff" or "motion")
    - If no matching case: 
      * Apologize briefly
      * Provide 2-3 general steps
      * Add: "Every case is unique – consulting a lawyer is recommended"

    Example structure to mimic:
    "I’m sorry to hear about your situation. Let me share a similar case:
    **Case Name**: Smith v. ABC Corp
    **Introduction**: A warehouse worker fired after reporting safety issues.
    **Details**: The employee reported violations in March 2022, was terminated April 2022 with no warning. The employer claimed budget cuts.
    **Verdict**: Court ruled wrongful termination – $150k awarded due to retaliation evidence.
    Next steps:
    - Document all safety reports you filed
    - Contact OSHA within 30 days
    - Consult an employment lawyer"""

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.3,
        max_tokens=1500
    )

    return response.choices[0].message.content