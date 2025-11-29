import sys
import os
from pathlib import Path
import glob
from dotenv import load_dotenv
from src.document_indexer import Indexation
from src.retriever import Recherche
from src.chatbot import RAGQuestionAnswering
from langchain_openai import ChatOpenAI

# Ajouter le répertoire parent au PATH
sys.path.insert(0, str(Path(__file__).parent.parent))
# Charger .env
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "file.env")
load_dotenv(dotenv_path)
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

print("=== INDEXATION ===")
pdf_files = glob.glob( "../data/*.pdf")
for pdf in pdf_files:
    print("Indexation de :", pdf)
    idx = Indexation(pdf, chunk_size=500, chunk_overlap=50)
    idx.index()
print("Indexation complète !\n")

print("=== CHATBOT ===")
llm = ChatOpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
    model="mistralai/mistral-7b-instruct:free",
    temperature=0.1
)

retriever = Recherche()
chatbot = RAGQuestionAnswering(retriever=retriever, llm=llm, max_history=5)
chatbot.interactive_session()

