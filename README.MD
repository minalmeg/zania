# RAG Processor

This project processes PDFs using Retrieval-Augmented Generation (RAG) with OpenAI's models and FAISS for efficient text retrieval.

## Setup

### RAG Code setup

1. **Clone the repository**:
   ```
   git clone https://github.com/your-username/rag-processor.git
   cd rag-processor
   ```

2. **Install dependencies**:
    ```
    pip install -r requirements.txt
    ```
3. **Download spaCy model**:
    ```
    python -m spacy download en_core_web_sm
    ```

4. **Set up OpenAI API key**:
    Create a config.json file with your OpenAI API key:
    ```
    {
        "openai": {
            "api_key": "your-openai-api-key"
        }
    }
    ```



