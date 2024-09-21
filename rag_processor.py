import logging
import fitz  # PyMuPDF for PDF extraction
import openai
import faiss
import numpy as np
import json

# Configure logging to write to a file called RAG_logging.log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("RAG_logging.log"),  # Log to file
        logging.StreamHandler()  # Optional: Log to console as well
    ]
)

# Load configuration file to get the OpenAI API key and other credentials
def load_config(file_path):
    """Loads the configuration from a JSON file."""
    logging.info(f"Loading configuration from {file_path}")
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

# PDF Text Extraction
def extract_text_from_pdf(pdf_file_path):
    """Extracts text from a PDF file."""
    logging.info(f"Extracting text from PDF: {pdf_file_path}")
    doc = fitz.open(pdf_file_path)
    text = ""
    for page_num, page in enumerate(doc, start=1):
        text += page.get_text()
        logging.debug(f"Extracted text from page {page_num}")
    return text

# Text Chunking for LLM token limit
def chunk_text(text, chunk_size=2000):
    """Splits the text into chunks within the token limit of LLM."""
    logging.info("Chunking text into smaller pieces")
    words = text.split(" ")
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    logging.info(f"Text has been split into {len(chunks)} chunks")
    return chunks

# Generate embeddings using OpenAI's embedding model
def generate_embeddings(text_chunks, api_key):
    """Generate embeddings for text chunks using OpenAI API."""
    logging.info("Generating embeddings for text chunks")
    openai.api_key = api_key
    embeddings = []
    for i, chunk in enumerate(text_chunks, start=1):
        response = openai.Embedding.create(
            input=chunk,
            model="text-embedding-ada-002"  # Low cost OpenAI embedding model
        )
        embeddings.append(response['data'][0]['embedding'])
        logging.debug(f"Generated embedding for chunk {i}/{len(text_chunks)}")
    logging.info("Generated embeddings for all text chunks")
    return embeddings

# Store embeddings in FAISS vector database
def create_faiss_index(embeddings):
    """Creates a FAISS index for embedding vectors."""
    logging.info("Creating FAISS index for embeddings")
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)  # L2 distance
    index.add(np.array(embeddings, dtype=np.float32))
    logging.info(f"FAISS index created with {len(embeddings)} embeddings")
    return index

# Retrieve relevant chunks based on question
def retrieve_relevant_chunks(question, index, text_chunks, api_key):
    """Retrieve the most relevant text chunks using FAISS and embeddings."""
    logging.info(f"Retrieving relevant chunks for question: {question}")
    openai.api_key = api_key
    response = openai.Embedding.create(
        input=question,
        model="text-embedding-ada-002"
    )
    question_embedding = response['data'][0]['embedding']

    # Search for the most relevant chunks
    D, I = index.search(np.array([question_embedding], dtype=np.float32), k=3)  # k is the number of top results
    logging.info(f"Retrieved top {len(I[0])} relevant chunks for question")
    return [text_chunks[i] for i in I[0]]

# LLM-based Answer Generation using OpenAI GPT-4o-mini
def get_answer_from_llm(retrieved_chunks, question, model="gpt-4o-mini"):
    """Generate an answer using LLM and retrieved chunks."""
    logging.info(f"Generating answer for question: {question}")
    context = " ".join(retrieved_chunks)
    prompt = f"Based on the following context, answer the question: {question}\n\nContext: {context}"
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    logging.info("Generated answer using LLM")
    return response.choices[0].text.strip()

# # LLM-based Answer Generation using OpenAI GPT-3.5-turbo (free/low-cost option)
# def get_answer_from_llm(retrieved_chunks, question, model="gpt-3.5-turbo"):
#     """Generate an answer using LLM and retrieved chunks."""
#     logging.info(f"Generating answer for question: {question}")
#     context = " ".join(retrieved_chunks)
#     prompt = f"Based on the following context, answer the question: {question}\n\nContext: {context}"

#     # Using gpt-3.5-turbo for completion (cost-effective)
#     response = openai.ChatCompletion.create(
#         model=model,
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": prompt}
#         ],
#         max_tokens=150,
#         temperature=0.7,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0
#     )

#     logging.info("Generated answer using GPT-3.5-turbo")
#     return response['choices'][0]['message']['content'].strip()


# Main function that processes the PDF and answers the questions
def process_pdf_and_questions(pdf_path, questions, config_file):
    """Processes the PDF, chunks the text, generates embeddings, and answers questions."""
    logging.info(f"Processing PDF: {pdf_path}")
    
    # Load the configuration (including OpenAI API key) from the config file
    config = load_config(config_file)
    openai_api_key = config["openai"]["api_key"]  # Extract OpenAI API key from the loaded config

    # Step 1: Extract text from the PDF
    pdf_text = extract_text_from_pdf(pdf_path)

    # Step 2: Split the text into chunks
    text_chunks = chunk_text(pdf_text)

    # Step 3: Generate embeddings for each chunk
    embeddings = generate_embeddings(text_chunks, openai_api_key)

    # Step 4: Create FAISS index for chunk embeddings
    faiss_index = create_faiss_index(embeddings)

    # Step 5: Prepare answers for each question
    answers = []
    for question in questions:
        logging.info(f"Processing question: {question}")
        # Retrieve relevant chunks for each question
        relevant_chunks = retrieve_relevant_chunks(question, faiss_index, text_chunks, openai_api_key)
        
        # Generate an answer from the relevant chunks
        answer = get_answer_from_llm(relevant_chunks, question)
        
        # If the answer is too short, assume low confidence and return "Data Not Available"
        if len(answer) < 20:
            logging.warning(f"Answer too short for question: {question}. Returning 'Data Not Available'")
            answer = "Data Not Available"

        answers.append(answer)

    logging.info("Finished processing all questions")
    # Return the generated answers
    return answers
