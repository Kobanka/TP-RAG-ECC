from src.document_indexer import Indexation
import glob

pdf_files = glob.glob("data/*.pdf")

for pdf in pdf_files:
    print("Indexation de :", pdf)
    idx = Indexation(pdf, chunk_size=500, chunk_overlap=50)
    idx.index()

print("Indexation complète !")
