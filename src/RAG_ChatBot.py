from src.retriever import Recherche
from langchain_core.language_models import BaseLanguageModel
from src.synthesis import ContextSynthesizer
from src.prompts import get_rag_prompt_template
from typing import List, Dict
import json,yaml

class RAGQuestionAnswering:
    
    def __init__(self, retriever: Recherche, llm: BaseLanguageModel, max_history: int = None):
        with open('./config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        self.conversation_history: List[Dict[str, str]] = []
        self.max_history = max_history if max_history is not None else config['rag']['max_history']
        self.k_retrieve = config['rag']['k_retrieve']
        self.retriever = retriever
        self.llm = llm
        self.synthesizer = ContextSynthesizer()
        self.prompt_template = get_rag_prompt_template()
        
    def _format_history(self) -> str:

        if not self.conversation_history:
            return "No previous conversation."

        formatted = []
        for i, turn in enumerate(self.conversation_history, 1):
            formatted.append(f"Turn {i}:")
            formatted.append(f"  User: {turn['user']}")
            formatted.append(f"  Assistant: {turn['assistant']}\n")

        return "\n".join(formatted)

    def _add_to_history(self, user_message: str, assistant_message: str):

        self.conversation_history.append({
            "user": user_message,
            "assistant": assistant_message
        })

        # Limiter la taille de l'historique
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)

    def _enhance_context_with_history(self, context: str) -> str:

        history = self._format_history()

        enhanced_context = f"""CONVERSATION HISTORY:
        {history}
        
        DOCUMENT CONTEXT:
        {context}"""

        return enhanced_context

    def chat(self, question: str, k: int = 4) -> str:
        # Récupérer les chunks pertinents depuis la base vectorielle
        retrieved_docs = self.retriever.query(question, k=k)

        # Synthétiser les chunks en un contexte clair et structuré
        synthesized_context = self.synthesizer.synthesize(retrieved_docs)

        # Enrichir le contexte avec l'historique
        enhanced_context = self._enhance_context_with_history(synthesized_context)

        # Appliquer le prompt template (celui de prompts.py)
        prompt = self.prompt_template.format(
            context=enhanced_context,
            question=question
        )
        response = self.llm.invoke(prompt)
        assistant_message = response.content.strip()
        self._add_to_history(question, assistant_message)

        return assistant_message
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
        
    def reset_conversation(self):
        self.conversation_history = []
        print("Conversation history has been reset.")

    def get_history(self) -> List[Dict[str, str]]:
        return self.conversation_history

    def save_conversation(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
        print(f"Conversation saved to {filepath}")

    def load_conversation(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            self.conversation_history = json.load(f)
        print(f"Conversation loaded from {filepath}")

    def interactive_session(self):
        print("=" * 70)
        print("RAG CHATBOT - Interactive Session")
        print("=" * 70)
        print("Commands:")
        print("  - Type your question to chat")
        print("  - 'reset' to clear conversation history")
        print("  - 'history' to view conversation history")
        print("  - 'save <filename>' to save conversation")
        print("  - 'quit' or 'exit' to end session")
        print("=" * 70)
        print()

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                # Commandes spéciales
                if user_input.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break

                elif user_input.lower() == 'reset':
                    self.reset_conversation()
                    continue

                elif user_input.lower() == 'history':
                    print("\n" + self._format_history())
                    continue

                elif user_input.lower().startswith('save '):
                    filename = user_input[5:].strip()
                    if filename:
                        self.save_conversation(filename)
                    else:
                        print("Please provide a filename.")
                    continue

                print("\nAssistant: ", end="")
                response = self.chat(user_input)
                print(response)
                print()

            except KeyboardInterrupt:
                print("\n\nSession interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again.\n")
