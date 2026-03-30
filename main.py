'''

Welcome to GDB Online.
GDB online is an online compiler and debugger tool for C, C++, Python, Java, PHP, Ruby, Perl,
C#, OCaml, VB, Swift, Pascal, Fortran, Haskell, Objective-C, Assembly, HTML, CSS, JS, SQLite, Prolog.
Code, Compile, Run and Debug online from anywhere in world.

'''

import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pandas as pd

# 1. Setup API Key (Ensure this is set in your environment)
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# 2. Prepare the Data (Simulated FAQ Dataset)
faq_data = {
    "question": [
        "How do I reset my SmartPlug?",
        "Does the camera work at night?",
        "What is the warranty period?",
        "How do I connect to Wi-Fi?"
    ],
    "answer": [
        "Hold the power button for 10 seconds until the light blinks red.",
        "Yes, the camera has infrared sensors for night vision up to 30 feet.",
        "All SmartHome Pro products come with a 2-year limited warranty.",
        "Use the SmartHome app and scan the QR code on the back of the device."
    ]
}
df = pd.DataFrame(faq_data)
text_data = df.apply(lambda row: f"Question: {row['question']} Answer: {row['answer']}", axis=1).tolist()

# 3. Create Vector Store (ChromaDB)
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(texts=text_data, embedding=embeddings)

# 4. Define the Prompt Template
template = """You are a helpful customer support assistant for SmartHome Pro. 
Use the following pieces of context to answer the user's question. 
If you don't know the answer, just say you don't know, don't try to make up an answer.

Context: {context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# 5. Build the QA Chain
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# 6. Run the Chatbot
print("--- SmartHome Pro Assistant is Online ---")
user_query = "What happens if my device breaks after a year?"
response = qa_chain.invoke(user_query)

print(f"User: {user_query}")
print(f"AI: {response['result']}")