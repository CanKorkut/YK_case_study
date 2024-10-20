# QA Retrieval System with Sentence Transformers (RAG) and T5

This project demonstrates a Question Answering (QA) system using Sentence Transformers to retrieve relevant documents from a dataset and a T5 model to generate natural language answers based on those retrieved documents.

## Project Overview

The system works by:

1. **Embedding the dataset**: Using the `SentenceTransformers` model, we embed context data (e.g., answers) into vector space.
2. **Building a FAISS Index**: We use FAISS to index these embeddings, enabling efficient similarity searches.
3. **Retrieving Relevant Documents**: Given a user query, the system retrieves the top 3 most relevant documents from the indexed dataset.
4. **Generating a Response**: The relevant documents are combined, and a `T5` model generates a natural language response based on the combined text.

### Directory Structure

```plaintext

├── create_dataset.py     # Creates embeddings and builds FAISS index
├── init_model.py         # Loads the T5 model and tokenizer
├── model_inference.py    # Handles document retrieval and response generation
├── main.py                   # Main script to run the system
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
