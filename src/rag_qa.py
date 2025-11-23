from src.retriever import Recherche
from langchain_core.language_models import BaseLanguageModel

class RAGQuestionAnswering:
    def __init__(self, retriever: Recherche, llm: BaseLanguageModel):
        self.retriever = retriever
        self.llm = llm

    def answer(self, question: str, k: int = 4) -> str:
        #récupérer les chunks pertinents depuis la base vectorielle
        retrieved_docs = self.retriever.query(question, k=k)

        #Extraire et concaténer le contenu des chunks
        context = "\n\n".join([doc["content"] for doc in retrieved_docs])

        #construire un prompt simple
        prompt = (
            f"Use the following pieces of context to answer the question at the end. "
            f"If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )

        #appeler le llm
        response = self.llm.invoke(prompt)
        return response.content.strip()
