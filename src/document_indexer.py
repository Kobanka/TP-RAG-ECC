from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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


