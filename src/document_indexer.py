from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class Indexation:
    def __init__(self, path, chunk_size, chunk_overlap):
        self.path = path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.documents = None
        self.chunks = None
        self.embedding_model = None
        self.vectorstore = None

    def data_load(self):
        loader = PyPDFLoader(self.path)
        self.documents = loader.load()
        return self.documents
    def splitting(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
            separators=["\n## ","\n### ","\n\n","\n"," ",""]
        )
        all_splits = text_splitter.split_documents(self.documents)
        self.chunks = all_splits
        return all_splits

    def embedding(self):
        if self.embedding_model is None:
            self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        return self.embedding_model

    def vector_store(self):
        # optimisation du modele comme ca on a pas besoin de tous recalculer si c'est deja remplie
        if self.vectorstore is None:  
            self.vectorstore = Chroma.from_documents(
                documents=self.chunks,
                embedding=self.embedding(),
                persist_directory="./chroma_langchain_db"
            )
        return self.vectorstore

    def index(self):
        self.data_load()
        self.splitting()
        self.embedding()
        return self.vector_store()



