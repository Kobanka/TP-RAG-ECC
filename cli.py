#python cli.py index
#python cli.py query --q "ma question"
#python cli.py chat
#python cli.py evaluate

import argparse
import sys
import os
from pathlib import Path
import glob

from dotenv import load_dotenv

load_dotenv("file.env")

from src.document_indexer import Indexation
from src.retriever import Recherche
from src.chatbot import RAGQuestionAnswering
from src.evaluator import Evaluator
from langchain_openai import ChatOpenAI


def load_llm():
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        print("Erreur : OPENROUTER_API_KEY non trouvée")
        sys.exit(1)

    llm = ChatOpenAI(
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        model="mistralai/mistral-7b-instruct:free",
        temperature=0.1
    )
    return llm


def cmd_index(args):
    data_dir = Path("./data")
    pdf_files = glob.glob(str(data_dir / "*.pdf"))

    print(f"Indexation de {len(pdf_files)} fichier(s)...")

    for pdf in pdf_files:
        indexer = Indexation(path=pdf, chunk_size=500, chunk_overlap=50)
        indexer.index()

    print("Indexation complète !")


def cmd_query(args):
    retriever = Recherche(persist_dir="./chroma_langchain_db")
    llm = load_llm()
    rag = RAGQuestionAnswering(retriever=retriever, llm=llm, max_history=5)

    response = rag.answer(args.question, k=4)
    print(f"\nRéponse :\n{response}\n")


def cmd_chat(args):
    retriever = Recherche(persist_dir="./chroma_langchain_db")
    llm = load_llm()
    chatbot = RAGQuestionAnswering(retriever=retriever, llm=llm, max_history=5)
    chatbot.interactive_session()


def cmd_evaluate(args):
    evaluator = Evaluator()
    metrics = evaluator.evaluate_pair(args.reference, args.prediction)

    print(f"\nExact Match : {metrics['exact_match']}")
    print(f"F1 Score : {metrics['f1']:.4f}")
    print(f"Similarité : {metrics['semantic_similarity']:.4f}\n")


def main():
    parser = argparse.ArgumentParser(description="CLI RAG")
    subparsers = parser.add_subparsers(dest='command')

    index_parser = subparsers.add_parser('index')
    index_parser.set_defaults(func=cmd_index)

    query_parser = subparsers.add_parser('query')
    query_parser.add_argument('-q', '--question', type=str, required=True)
    query_parser.set_defaults(func=cmd_query)

    chat_parser = subparsers.add_parser('chat')
    chat_parser.set_defaults(func=cmd_chat)

    eval_parser = subparsers.add_parser('evaluate')
    eval_parser.add_argument('--reference', type=str, required=True)
    eval_parser.add_argument('--prediction', type=str, required=True)
    eval_parser.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
