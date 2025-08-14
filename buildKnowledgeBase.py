from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama.llms import OllamaLLM
from langchain_community.docstore.in_memory import InMemoryDocstore
from config import *
import logging
import json
import os
import numpy as np
import faiss
import pickle
import uuid

# Configure logging for detailed debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# System prompt for concise, relevant responses
SYSTEM_PROMPT = """You are a precise document analysis assistant. Provide accurate, concise, and relevant answers based solely on the provided documents. Focus on key information and avoid extraneous details."""

class DocumentSearchSystem:
    def __init__(self):
        self.ollm = None
        self.embed_model = None
        self.index = None
        self.docs = None
        self.index_to_docstore_id = None
        self.vectorstore = None  # New: Store FAISS vectorstore for reuse

    def init_llm(self):
        """Initialize LLM and embedding model with config validation."""
        try:
            required_configs = ['LLM_MODEL', 'OLLAMA_URL', 'EMBED_MODEL', 'FOLDER_PATH', 'FAISS_INDEX_NAME', 'INDEX_STORAGE_PATH']
            for config in required_configs:
                if not globals().get(config):
                    raise ValueError(f"Missing configuration: {config}")
            self.ollm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_URL)
            self.embed_model = OllamaEmbeddings(base_url=OLLAMA_URL, model=EMBED_MODEL)
            logging.info("LLM and embedding model initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize LLM or embedding model: {str(e)}")
            raise
        
    def load_index(self):
        """Load and index documents from folder, creating FAISS vectorstore."""
        path = FOLDER_PATH
        if not os.path.exists(path):
            raise ValueError(f"Folder path {path} does not exist.")
        
        logging.info("Loading docs from %s", path)
        all_docs = []
        
        for entry in os.listdir(path):
            full_path = os.path.join(path, entry)
            logging.info("Loading %s", full_path)
            
            file_extension = os.path.splitext(entry)[1].lower()
            try:
                if file_extension == '.pdf':
                    loader = PyPDFLoader(full_path)
                elif file_extension == '.txt':
                    loader = TextLoader(full_path, encoding='utf-8')
                else:
                    logging.warning(f"Unsupported file type: {file_extension}. Skipping {entry}")
                    continue
                
                documents = loader.load()
                text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator="\n")
                docs = text_splitter.split_documents(documents=documents)
                all_docs.extend(docs)
            except Exception as e:
                logging.error(f"Error loading {entry}: {str(e)}")
                continue
        
        if not all_docs:
            raise ValueError("No documents were successfully loaded.")
        
        # Create FAISS vectorstore and save it
        try:
            logging.info("Creating FAISS vectorstore...")
            self.vectorstore = FAISS.from_documents(all_docs, self.embed_model)
            
            # Save index to disk
            os.makedirs(INDEX_STORAGE_PATH, exist_ok=True)
            self.vectorstore.save_local(INDEX_STORAGE_PATH, FAISS_INDEX_NAME)
            logging.info(f"Index saved at {INDEX_STORAGE_PATH}/{FAISS_INDEX_NAME}.faiss")
            return self.vectorstore, all_docs
        except Exception as e:
            logging.error(f"Error creating FAISS vectorstore: {str(e)}")
            raise
        
    # def load_index1(self):
    #     """Load and index documents from folder, creating FAISS vectorstore."""
    #     path = FOLDER_PATH
    #     if not os.path.exists(path):
    #         raise ValueError(f"Folder path {path} does not exist.")
        
    #     logging.info("Loading docs from %s", path)
    #     all_docs = []
        
    #     for entry in os.listdir(path):
    #         full_path = os.path.join(path, entry)
    #         logging.info("Loading %s", full_path)
            
    #         file_extension = os.path.splitext(entry)[1].lower()
    #         try:
    #             if file_extension == '.pdf':
    #                 loader = PyPDFLoader(full_path)
    #             elif file_extension == '.txt':
    #                 loader = TextLoader(full_path, encoding='utf-8')
    #             else:
    #                 logging.warning(f"Unsupported file type: {file_extension}. Skipping {entry}")
    #                 continue
                
    #             documents = loader.load()
    #             text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator="\n")
    #             docs = text_splitter.split_documents(documents=documents)
    #             all_docs.extend(docs)
    #         except Exception as e:
    #             logging.error(f"Error loading {entry}: {str(e)}")
    #             continue
        
    #     if not all_docs:
    #         raise ValueError("No documents were successfully loaded.")
        
    #     # Create FAISS vectorstore directly from documents
    #     try:
    #         logging.info("Creating FAISS vectorstore...")
    #         self.vectorstore = FAISS.from_documents(all_docs, self.embed_model)
    #         self.docs = all_docs
    #         self.index = self.vectorstore.index
    #         self.index_to_docstore_id = self.vectorstore.index_to_docstore_id
            
    #         # Save index, documents, and mapping
    #         os.makedirs(INDEX_STORAGE_PATH, exist_ok=True)
    #         index_path = os.path.join(INDEX_STORAGE_PATH, f"{FAISS_INDEX_NAME}")
    #         faiss.write_index(self.index, index_path)
            
    #         docs_path = os.path.join(INDEX_STORAGE_PATH, f"{FAISS_INDEX_NAME}")
    #         with open(docs_path, 'wb') as f:
    #             pickle.dump(self.docs, f)
            
    #         mapping_path = os.path.join(INDEX_STORAGE_PATH, f"{FAISS_INDEX_NAME}")
    #         with open(mapping_path, 'wb') as f:
    #             pickle.dump(self.index_to_docstore_id, f)
            
    #         logging.info(f"Index saved at {index_path}, mapping at {mapping_path}")
    #         return self.index, self.docs
    #     except Exception as e:
    #         logging.error(f"Error creating FAISS vectorstore: {str(e)}")
    #         raise

    def query_pdf(self, query):
        """Query the indexed documents with RetrievalQA."""
        try:
            self.vectorstore = FAISS.load_local(f"{INDEX_STORAGE_PATH}", self.embed_model, f"{FAISS_INDEX_NAME}", allow_dangerous_deserialization=True)
            if not self.vectorstore:
                raise ValueError("Vectorstore not initialized. Run load_index first.")
            
            
              # Load document using PyPDFLoader document loader
   
            # Load from local storage
            
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2, "score_threshold": 0.7})
            qa = RetrievalQA.from_chain_type(
                llm=self.ollm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            result = qa.invoke(query)
            return {
                "answer": result["result"].strip(),
                "sources": [doc.metadata.get('source', 'Unknown') for doc in result["source_documents"]]
            }
        except Exception as e:
            logging.error(f"Error in query_pdf: {str(e)}")
            return {"answer": f"Error processing query: {str(e)}", "sources": []}

    
def main():
    try:
        search_system = DocumentSearchSystem()
        search_system.init_llm()
        index, docs = search_system.load_index()
        # logging.info(f"Created index with {len(docs)} documents")
        
        print("\nDocument Search System")
        print("=====================")
        # print(f"Indexed {len(docs)} documents")
        print("\nEnter your query (type 'exit' to quit):")
        
        while True:
            query = input("\nQuery: ").strip()
            if not query:
                print("Please enter a valid query.")
                continue
            if query.lower() == "exit":
                break
            qa_result = search_system.query_pdf(query)
            print("\nQA Response")
            print("===========")
            print(f"Answer: {qa_result['answer']}")
            print(f"Sources: {', '.join(qa_result['sources'])}")
    
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()