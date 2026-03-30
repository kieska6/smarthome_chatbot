SmartHome Pro Assistant

A technical support chatbot based on Retrieval-Augmented Generation (RAG) for the SmartHome Pro ecosystem (smart plugs, cameras, and hubs). Its main goal is to provide instant, automated technical assistance and product information.1

Key Features
Automated Technical Support: Reduces human intervention for basic customer service tasks.1
Retrieval-Augmented Generation (RAG): Uses a Large Language Model (LLM) to synthesize answers from a verified FAQ database, avoiding "hallucinations" and ensuring grounded responses.1
Semantic Search: Utilizes Sentence Embeddings to convert text into mathematical vectors, allowing the system to find answers based on the query's meaning rather than simple keywords.1
Application Architecture

The chatbot follows a Vector-Store Retrieval pipeline:1
Loading: The FAQ dataset is ingested via Pandas.1
Embedding: Each FAQ pair is converted into a vector using an OpenAI embedding model.1
Storage: These vectors are stored and indexed within ChromaDB for fast retrieval.1
Querying: The user's question is converted into a vector to perform a similarity search in the database.1
Generation: The user's query and the retrieved FAQ context are passed to the LLM (GPT-5.4) to craft a final, context-aware response.1
Technologies and Dependencies

This project uses the following tools and libraries:1
LangChain: The central orchestration framework used to "chain" the data retrieval process with the LLM prompt.1
OpenAI API (GPT-5.4): The Large Language Model used to generate human-like responses.1
ChromaDB: A vector database used to store and index FAQ embeddings for high-speed semantic retrieval.1
Pandas: Used to load, clean, and structure the initial FAQ CSV data.1
Installation and Configuration

Follow the steps below to get the SmartHome Pro Assistant running on your local machine.

1. Prerequisites
Python (version 3.x recommended)
An OpenAI API key
2. Environment Setup

Create and activate a virtual environment:
python -m venv venv
# On Linux/Mac
source venv/bin/activate
# On Windows
.\venv\Scripts\activate
3. Install Dependencies

Install all the required libraries:
pip install langchain langchain-openai langchain-community chromadb pandas
4. API Key Configuration

Set your OpenAI API key as an environment variable:
# For Linux/Mac
export OPENAI_API_KEY='your-key-here'
# For Windows
set OPENAI_API_KEY='your-key-here'
Usage

1. Running the Chatbot

Execute the main script in your terminal:
python smarthome_chatbot.py
2. Interaction

The chatbot will launch, and you can interact directly within the terminal prompt.
