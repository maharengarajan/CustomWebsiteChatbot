from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os

load_dotenv()
HUGGINGFACE_TOKEN=os.getenv("HUGGINGFACE_TOKEN")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

llm=ChatGroq(api_key=GROQ_API_KEY,model="Llama3-8b-8192")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

loader = RecursiveUrlLoader("https://www.datanetiix.com/",)
docs = loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
splits=text_splitter.split_documents(docs)
vectorstore=Chroma.from_documents(documents=splits,embedding=embeddings,persist_directory="./db")
retriever=vectorstore.as_retriever()
retriever

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain=create_stuff_documents_chain(llm,prompt)
rag_chain=create_retrieval_chain(retriever,question_answer_chain)

response=rag_chain.invoke({"input":"What is Datanetiix"})
response['answer']