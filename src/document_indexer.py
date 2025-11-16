from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class Indexation :
    def __init__(self,path,chunk_size,chunk_overlap):
        self.path = path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def data_load(self):
        loader = PyPDFLoader(self.path)
        return loader.load()
        
    def splitting(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
        )
        all_splits = text_splitter.split_documents(self.data_load())
        return all_splits

    def embedding(self):
        #j'ai utilisé le modele huggingFace recommander dans la documentation de langchain
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")  
        return embedding_model

    def vector_store(self):
        vector_store = Chroma.from_documents(
            documents=self.splitting(),
            embedding=self.embedding(),
            persist_directory="./chroma_langchain_db"
        )
        return vector_store






