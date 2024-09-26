import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import Ollama
from langchain.chains import VectorDBQA


file_paths = ['multiply.py', 'test.py'] 

documents = []
for path in file_paths:
    loader = TextLoader(path)
    documents.extend(loader.load())  

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = FAISS.from_documents(docs, embeddings)

print(vector_store.similarity_search("Write a test case for multiply_three as well"))


llm = Ollama(model="llama2")  

class LoggingLLMWrapper:
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, *args, **kwargs):
        print("Input to LLM:", args, kwargs)
        return self.llm(*args, **kwargs)


llm_with_logging = LoggingLLMWrapper(llm)


qa_chain = VectorDBQA.from_chain_type(llm, chain_type="stuff", vectorstore=vector_store)

question = input("ENTER YOUR QUESTION")
response = qa_chain.run(question)

print("Answer:", response)
