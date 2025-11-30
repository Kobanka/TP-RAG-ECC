# python cli.py index
# python cli.py query --q "ma question"
# python cli.py chat
# python cli.py evaluate

import argparse,yaml,sys,os,glob
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("file.env")

from src.document_indexer import Indexation
from src.retriever import Recherche
from src.RAG_ChatBot import RAGQuestionAnswering
from src.evaluator import Evaluator
from langchain_openai import ChatOpenAI


def load_config(config_path="config.yaml"):
    """Charge la configuration depuis le fichier YAML"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_llm(config):
    api_key_env = config['llm']['api_key_env']
    openrouter_api_key = os.getenv(api_key_env)

    if not openrouter_api_key:
        print(f"Erreur : {api_key_env} non trouvée")
        sys.exit(1)

    llm = ChatOpenAI(
        api_key=openrouter_api_key,
        base_url=config['llm']['base_url'],
        model=config['llm']['model'],
        temperature=config['llm']['temperature']
    )
    return llm


def cmd_index(args, config):
    data_dir = Path(config['indexation']['data_dir'])
    pdf_files = glob.glob(str(data_dir / "*.pdf"))

    print(f"Indexation de {len(pdf_files)} fichier(s)...")

    chunk_size = config['indexation']['chunk_size']
    chunk_overlap = config['indexation']['chunk_overlap']

    for pdf in pdf_files:
        indexer = Indexation(
            path=pdf,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        indexer.index()

    print("Indexation complète !")


def cmd_query(args, config):
    persist_dir = config['indexation']['persist_dir']
    retriever = Recherche(persist_dir=persist_dir)

    llm = load_llm(config)

    max_history = config['rag']['max_history']
    k_retrieve = config['rag']['k_retrieve']

    rag = RAGQuestionAnswering(
        retriever=retriever,
        llm=llm,
        max_history=max_history
    )

    response = rag.answer(args.question, k=k_retrieve)
    print(f"\nRéponse :\n{response}\n")


def cmd_chat(args, config):
    persist_dir = config['indexation']['persist_dir']
    retriever = Recherche(persist_dir=persist_dir)

    llm = load_llm(config)

    max_history = config['rag']['max_history']

    chatbot = RAGQuestionAnswering(
        retriever=retriever,
        llm=llm,
        max_history=max_history
    )
    chatbot.interactive_session()


def cmd_evaluate(args, config):
    evaluator = Evaluator()
    metrics = evaluator.evaluate_pair(args.reference, args.prediction)

    print(f"\nExact Match : {metrics['exact_match']}")
    print(f"F1 Score : {metrics['f1']:.4f}")
    print(f"Similarité : {metrics['semantic_similarity']:.4f}\n")


def main():
    parser = argparse.ArgumentParser(description="CLI RAG")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Chemin vers le fichier de configuration YAML'
    )

    subparsers = parser.add_subparsers(dest='command')

    # Commande index
    index_parser = subparsers.add_parser('index', help='Indexer les documents PDF')
    index_parser.set_defaults(func=cmd_index)

    # Commande query
    query_parser = subparsers.add_parser('query', help='Poser une question')
    query_parser.add_argument('-q', '--question', type=str, required=True)
    query_parser.set_defaults(func=cmd_query)

    # Commande chat
    chat_parser = subparsers.add_parser('chat', help='Session de chat interactive')
    chat_parser.set_defaults(func=cmd_chat)

    # Commande evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Évaluer une prédiction')
    eval_parser.add_argument('--reference', type=str, required=True)
    eval_parser.add_argument('--prediction', type=str, required=True)
    eval_parser.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Charger la configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Erreur : fichier de configuration '{args.config}' introuvable")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Erreur lors de la lecture du fichier de configuration : {e}")
        sys.exit(1)

    # Exécuter la commande avec la config
    args.func(args, config)


if __name__ == "__main__":
    main()
