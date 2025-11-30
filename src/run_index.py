#fichier test pour run l'indexer

from src.document_indexer import Indexation
import glob
import os


DATA_DIR = os.getenv("DATA_DIR", "./data")
pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))


for pdf in pdf_files:
    print("Indexation de :", pdf)
    idx = Indexation(pdf, chunk_size=500, chunk_overlap=50)
    idx.index()

print("Indexation complète !")
