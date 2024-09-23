import logging
import fitz  # PyMuPDF for PDF extraction
from langchain_community.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import spacy
import json
import openai
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/RAG_logging.log"),  # Log to file
        logging.StreamHandler()  # Optional: Log to console as well
    ]
)

# Load spaCy NER model
nlp = spacy.load("en_core_web_sm")

# Function to load configuration
def load_config(file_path):
    """Loads the configuration from a JSON file."""
    logging.info(f"Loading configuration from {file_path}")
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

# Function to extract text from PDF (with optional page limit)
def extract_text_from_pdf(pdf_file_path, page_limit=None):
    """Extracts text from a PDF file."""
    logging.info(f"Extracting text from PDF: {pdf_file_path}")
    doc = fitz.open(pdf_file_path)
    text = ""
    for page_num, page in enumerate(doc, start=1):
        if page_limit and page_num > page_limit:
            break
        text += page.get_text()
        logging.debug(f"Extracted text from page {page_num}")
    return text

# Function to extract named entities using spaCy
def extract_named_entities(text):
    """Uses spaCy to extract named entities from the text."""
    logging.info("Extracting named entities from the text")
    doc = nlp(text)
    entities = {ent.label_: ent.text for ent in doc.ents}
    logging.info(f"Named entities found: {entities}")
    return entities

# Function to chunk the text for LLM token limits
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

# Function to create FAISS index
def create_faiss_index(text_chunks, api_key):
    """Creates FAISS index from text chunks using OpenAI embeddings."""
    logging.info("Creating FAISS index for embeddings")

    # Initialize OpenAIEmbeddings with the API key
    embedding_model = OpenAIEmbeddings(openai_api_key=api_key)

    # Generate embeddings for the text chunks
    embeddings = [embedding_model.embed_query(chunk) for chunk in text_chunks]
    
    # Create FAISS index using the generated embeddings
    faiss_index = FAISS.from_texts(text_chunks, embedding_model)

    logging.info(f"FAISS index created with {len(text_chunks)} chunks")
    return faiss_index

# Function to prioritize chunks based on index
def use_index_to_guide_retrieval(text_chunks, index_text):
    """Processes the index from the first few pages and returns prioritized text chunks."""
    logging.info("Processing index to guide retrieval.")
    
    prioritized_chunks = []
    for chunk in text_chunks:
        if any(keyword.lower() in chunk.lower() for keyword in index_text.split()):
            prioritized_chunks.append(chunk)
    
    logging.info(f"Prioritized {len(prioritized_chunks)} chunks based on index.")
    
    if len(prioritized_chunks) == 0:
        logging.warning("No prioritized chunks found, returning original chunks.")
        return text_chunks
    
    return prioritized_chunks

# # Function to generate a custom answer
# def generate_custom_answer(question, retrieved_chunks, openai_api_key, llm_model="gpt-3.5-turbo"):
#     """Generates an answer by passing the retrieved chunks and question to the OpenAI model."""
#     logging.info(f"Generating answer for question: {question}")
    
#     # Extract the text from each retrieved chunk (Document objects)
#     context = " ".join([doc.page_content for doc in retrieved_chunks])  # Extracting 'page_content' from Document objects

#     # Create the prompt with the extracted text as context
#     prompt = f"Based on the following context, answer the question: {question}\n\nContext: {context}"

#     # Make the call to OpenAI's completion API
#     openai.api_key = openai_api_key
#     response = openai.ChatCompletion.create(
#         model=llm_model,
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
    
#     answer = response['choices'][0]['message']['content'].strip()
#     logging.info(f"Generated answer: {answer}")
    
#     return answer

# Function to generate a custom answer using gpt-4o-mini
def generate_custom_answer(question, retrieved_chunks, openai_api_key, llm_model="gpt-4o-mini"):
    """Generates an answer by passing the retrieved chunks and question to the OpenAI model."""
    logging.info(f"Generating answer for question: {question}")

    # Extract the text from each retrieved chunk (Document objects)
    context = " ".join([doc.page_content for doc in retrieved_chunks])  # Extracting 'page_content' from Document objects

    # Create the prompt with the extracted text as context
    prompt = f"Based on the following context, answer the question: {question}\n\nContext: {context}"

    # Make the call to OpenAI's completion API with gpt-4o-mini model
    openai.api_key = openai_api_key
    response = openai.ChatCompletion.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates concise answers."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,  # Adjust token limit if required by the model
        temperature=0.5,  # Lower temperature to ensure more focused responses
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    # Extract and return the generated answer
    answer = response['choices'][0]['message']['content'].strip()
    logging.info(f"Generated answer: {answer}")
    
    return answer

# Function to extract valid questions using regex
def extract_questions(text):
    """Extracts complete questions using a regex pattern."""
    logging.info("Extracting questions from the input text.")
    
    # Regex pattern to match questions starting with a number followed by a period
    pattern = r"\d+\.\s.+?(?=\d+\.|$)"
    
    # Find all matches in the input text
    questions = re.findall(pattern, text, re.DOTALL)
    
    logging.info(f"Extracted {len(questions)} questions.")
    return questions

def format_questions_and_answers(questions, answers):
    """
    Formats questions and answers as a single string where each question
    is followed by its corresponding answer.
    """
    formatted_output = []
    
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_output.append(f"{question}\nAnswer: {answer}\n")
    
    return "\n".join(formatted_output)

# Main function to process PDF and answer questions
def process_pdf_and_questions(pdf_path, questions_text, config_file):
    """
    Processes the PDF, extracts named entities, chunks the text,
    generates embeddings, and answers each question separately.
    """
    logging.info(f"Processing PDF: {pdf_path}")
    
    # Load configuration (including OpenAI API key)
    config = load_config(config_file)
    openai_api_key = config["openai"]["api_key"]
    
    if not openai_api_key:
        raise ValueError("OpenAI API key not found in the configuration file.")
    
    # Extract index from the first two pages of the PDF
    index_text = extract_text_from_pdf(pdf_path, page_limit=2)
    
    # Extract full text from the PDF
    pdf_text = extract_text_from_pdf(pdf_path)
    
    # Extract named entities
    named_entities = extract_named_entities(pdf_text)

    # Split the text into chunks
    text_chunks = chunk_text(pdf_text)

    # Prioritize text chunks using the index
    prioritized_chunks = use_index_to_guide_retrieval(text_chunks, index_text)
    
    # Create FAISS index
    vectorstore = create_faiss_index(prioritized_chunks, openai_api_key)

    # Extract complete questions using regex
    questions = extract_questions(questions_text)

    # Answer each question
    answers = []
    for question in questions:
        logging.info(f"Processing question: {question}")
        
        # Retrieve the most relevant chunks
        retrieved_chunks = vectorstore.similarity_search(question, k=3)

        # Generate an answer from the retrieved chunks
        answer = generate_custom_answer(question, retrieved_chunks, openai_api_key=openai_api_key)
        
        # If the answer is too short, assume low confidence and return "Data Not Available"
        if len(answer) < 2:
            logging.warning(f"Answer too short for question: {question}. Returning 'Data Not Available'")
            answer = "Data Not Available"
        
        answers.append(answer)

    # Format questions and answers into a single string
    formatted_output = format_questions_and_answers(questions, answers)
    
    logging.info("Finished processing all questions")
    return formatted_output
