# Research Paper Question Answering System (RAG)

## Overview

This project is a simple Research Paper Question Answering System built using the RAG (Retrieval-Augmented Generation) approach. It allows users to ask questions related to a research paper and generates answers based only on the information available in that paper.

The project loads the research paper, splits it into smaller chunks, creates embeddings, stores them in a Chroma vector database, and retrieves the most relevant information when a question is asked. The retrieved content is then passed to the Llama 3.1 model using the Groq API to generate the final answer.

## Project Structure

CEI-DS-WEEK7-ASSIGNMENT-NITIN-PRAJAPAT_PCE

├── data
│   └── research.pdf

├── notebook
│   └── week7.ipynb

├── app.py

├── chatbot.py

├── vectorstore.py

├── requirements.txt

├── README.md

└── LICENSE

## Technologies Used

- Python
- LangChain
- ChromaDB
- HuggingFace Embeddings
- Groq API
- Llama 3.1
- PyPDF

## How It Works

1. Load the research paper.
2. Split the document into smaller chunks.
3. Create embeddings for each chunk.
4. Store the embeddings in ChromaDB.
5. Accept a question from the user.
6. Retrieve the most relevant chunks from the database.
7. Generate the final answer using the Llama 3.1 model.

## Features

- Loads a research paper in PDF format.
- Splits the document into smaller chunks.
- Creates vector embeddings.
- Stores embeddings in ChromaDB.
- Retrieves relevant information based on the user's question.
- Generates answers using a Large Language Model.

## Running the Project

Install all required libraries using:

pip install -r requirements.txt

Run the project using:

python app.py

## Sample Question

What are the main applications of Artificial Intelligence discussed in this research paper?

## Author

Nitin Prajapat

B.Tech Computer Science and Engineering