import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# Load environment variables
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize LLM and embeddings
llm = ChatGroq(api_key=GROQ_API_KEY, model="Llama3-8b-8192")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Specify the directory where embeddings are stored
persist_directory = "./db"

# Load the vector store from the persisted directory
vectorstore = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
retriever = vectorstore.as_retriever()

# Define the system prompt for the assistant
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise.\n\n{context}"
)

# Create the chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create the RAG chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Set up the Streamlit app layout
st.set_page_config(page_title="Datanetiix RAG Chat", layout="centered")
st.title("Datanetiix RAG Chatbot")

# User input for the question
user_input = st.text_input("Ask a question about Datanetiix:")

# Process the input and get the response
if user_input:
    with st.spinner("Fetching the answer..."):
        response = rag_chain.invoke({"input": user_input})
        st.success("Answer retrieved!")
        st.write(f"**Assistant**: {response['answer']}")

# Display footer for additional info
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Renga Rajan K")
st.sidebar.markdown("Powered by Streamlit")
