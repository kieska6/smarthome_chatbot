# SmartHome Pro Assistant

A technical support chatbot based on Retrieval-Augmented Generation (RAG) for the SmartHome Pro ecosystem (smart plugs, cameras, and hubs). Its main goal is to provide instant, automated technical assistance and product information.

## Key Features
* **Automated Technical Support:** Reduces human intervention for basic customer service tasks.
* **Retrieval-Augmented Generation (RAG):** Uses a Large Language Model (LLM) to synthesize answers from a verified FAQ database, avoiding "hallucinations" and ensuring grounded responses.
* **Semantic Search:** Utilizes Sentence Embeddings to convert text into mathematical vectors, allowing the system to find answers based on the query's meaning rather than simple keywords.

## Application Architecture
The chatbot follows a Vector-Store Retrieval pipeline:
1. **Loading:** The FAQ dataset is ingested via Pandas.
2. **Embedding:** Each FAQ pair is converted into a vector using an OpenAI embedding model.
3. **Storage:** These vectors are stored and indexed within ChromaDB for fast retrieval.
4. **Querying:** The user's question is converted into a vector to perform a similarity search in the database.
5. **Generation:** The user's query and the retrieved FAQ context are passed to the LLM to craft a final, context-aware response.

## Technologies and Dependencies
This project uses the following tools and libraries:
* **LangChain:** The central orchestration framework used to "chain" the data retrieval process with the LLM prompt.
* **OpenAI API:** The Large Language Model used to generate human-like responses.
* **ChromaDB:** A vector database used to store and index FAQ embeddings for high-speed semantic retrieval.
* **Pandas:** Used to load, clean, and structure the initial FAQ CSV data.

## Installation and Configuration

### 1. Prerequisites
* Python (version 3.x recommended)
* An OpenAI API key

### 2. Environment Setup
Create and activate a virtual environment:
```bash
python -m venv venv

# On Linux/Mac
source venv/bin/activate
# On Windows
.\venv\Scripts\activate

### 3. Install Dependencies
Install all the required libraries:
pip install langchain langchain-openai langchain-community chromadb pandas tiktoken

### 4. Setup Data
Ensure that the faq_data.csv file is located in the same root directory as your main script.

### 5. API Key Configuration
Set your OpenAI API key as an environment variable.

# For Linux/Mac:

Bash
export OPENAI_API_KEY='your-key-here'
# For Windows:

DOS
set OPENAI_API_KEY='your-key-here'
Usage
1. Running the Chatbot
Execute the main script in your terminal:

Bash
python smarthome_chatbot.py
2. Interaction
The chatbot will launch, and you can interact directly within the terminal prompt. Type exit or quit to safely close the session.
