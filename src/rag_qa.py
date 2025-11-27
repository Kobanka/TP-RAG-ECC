from src.retriever import Recherche
from langchain_core.language_models import BaseLanguageModel
from src.synthesis import ContextSynthesizer
from src.prompts import get_rag_prompt_template

class RAGQuestionAnswering:
    def __init__(self, retriever: Recherche, llm: BaseLanguageModel):
        self.retriever = retriever
        self.llm = llm
        self.synthesizer = ContextSynthesizer()
        self.prompt_template = get_rag_prompt_template()

    def answer(self, question: str, k: int = 4) -> str:
        #récupérer les chunks pertinents depuis la base vectorielle
        retrieved_docs = self.retriever.query(question, k=k)

        #synthetiser les chunks en un contexte clair et structuré
        synthesized_context = self.synthesizer.synthesize(retrieved_docs)

        #appliquer le promptTemplate
        prompt = self.prompt_template.format(context=synthesized_context, question=question)

        #appeler le llm
        response = self.llm.invoke(prompt)
        return response.content.strip()
