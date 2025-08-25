import pypdf
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import  GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory



load_dotenv()
API_KEY = os.getenv("API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=1.5, api_key=API_KEY)

def load_documents_from_directory(directory_path):
    loaders=[
        DirectoryLoader(directory_path, glob="**/*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(directory_path, glob="**/*.txt", loader_cls=TextLoader),
        DirectoryLoader(directory_path, glob="**/*.docx", loader_cls=UnstructuredWordDocumentLoader)
    ]
    docs=[]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    for loader in loaders:
        text=text_splitter.split_documents(loader.load())        
        docs.extend(text)
    return docs

    
def create_vector_database(docs, persist_directory="chroma_db"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vector_db.persist()
    return vector_db


def load_or_create_vector_db(persist_directory="chroma_db"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


    if os.path.exists(persist_directory):
        print("Loading existing Chroma database...")
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    # Otherwise create it once
    print("Creating new Chroma database...")
    docs = load_documents_from_directory("user_docs")
    return create_vector_database(docs, persist_directory=persist_directory)


# Create the database once at startup
vector_db = load_or_create_vector_db("chroma_db")



def create_qa_chain_with_memory(vector_db):
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    memory = ConversationBufferMemory(
        memory_key="chat_history",  # where memory will be stored
        return_messages=True,
        output_key="answer" # stores conversation as messages instead of plain text
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
    return qa_chain


qa_chain = create_qa_chain_with_memory(vector_db)


def run_query(query):
    llm_response = qa_chain(query)
    
    result = llm_response["answer"]


    return result