from langchain_community.document_loaders import PyPDFLoader

class Indexation :
    def __init__(self,path,chunk_size,chunk_overlap):
        self.path = path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def data_load(self):
        loader = PyPDFLoader(self.path)
        return loader.load()
