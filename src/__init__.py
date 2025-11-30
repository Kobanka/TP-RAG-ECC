import sys,os,glob,yaml
from pathlib import Path
from dotenv import load_dotenv
from src.document_indexer import Indexation
from src.retriever import Recherche
from src.RAG_ChatBot import RAGQuestionAnswering
from langchain_openai import ChatOpenAI

# Load config
with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print(config['paths']['data'])
# Ajouter le répertoire parent au PATH
sys.path.insert(0, str(Path(__file__).parent.parent))

# Charger file.env
dotenv_path = config['paths']['env_file']
load_dotenv(dotenv_path)
openrouter_api_key = os.getenv(config['llm']['api_key_env'])

print("=== INDEXATION ===")
pdf_files = glob.glob( "../data/*.pdf")
for pdf in pdf_files:
    print("Indexation de :", pdf)
    idx = Indexation(pdf)
    idx.index()
print("Indexation complète !\n")

print("=== CHATBOT ===")
llm = ChatOpenAI(
    api_key=openrouter_api_key,
    base_url=config['llm']['base_url'],
    model=config['llm']['model'],
    temperature=config['llm']['temperature']
)

retriever = Recherche()
chatbot = RAGQuestionAnswering(retriever=retriever, llm=llm, max_history=config['rag']['max_history'])
chatbot.interactive_session()



