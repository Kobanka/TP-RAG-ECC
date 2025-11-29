# test_qa.py
from src.retriever import Recherche
from src.rag_qa import RAGQuestionAnswering
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

#charger la clé API
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "file.env")
load_dotenv(dotenv_path)
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

llm = ChatOpenAI(
    openai_api_key=openrouter_api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    model="mistralai/mistral-7b-instruct:free",
    temperature=0.1
)

#charger le retriever et initialiser le RAG
retriever = Recherche()
rag = RAGQuestionAnswering(retriever=retriever, llm=llm)

#tester par une question
question = "Why is AI trade issue given the documents, explain?"
answer = rag.answer(question, k=4)

print("Question :", question)
print("Réponse :", answer)
