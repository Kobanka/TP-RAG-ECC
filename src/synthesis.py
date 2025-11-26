# Mission: Exploiter les paragraphes récupérés pour synthétiser l'information
from typing import List, Dict, Any

class ContextSynthesizer:
    """
    Classe dédiée UNIQUEMENT à l'exploitation et la synthèse des paragraphes récupérés
    """
    
    @staticmethod
    def synthesize(retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Exploite les paragraphes récupérés de la base vectorielle
        pour produire un contexte synthétique, clair et bien structuré
        prêt à être envoyé au LLM.
        
        Args:
            retrieved_chunks: Liste de dicts avec clés 'content', 'source', 'score', etc.
            
        Returns:
            Chaîne de texte optimisée pour la génération (synthèse + sources)
        """
        if not retrieved_chunks:
            return "Aucun document pertinent n'a été trouvé."

        synthesized_parts = []

        for i, chunk in enumerate(retrieved_chunks, 1):
            content = chunk.get("content", "").strip()
            source = chunk.get("source", "document inconnu")
            page = chunk.get("page", "page inconnue")
            score = chunk.get("score", 0.0)

            # Synthèse intelligente : on enlève les doublons, on garde l'essentiel
            if len(content) > 500:
                content = content[:500] + " [...]"

            synthesized_parts.append(
                f"[Document {i}] Source : {source} | Page : {page} | Pertinence : {score:.3f}\n"
                f"{content}\n"
            )

        intro = "Voici les extraits les plus pertinents trouvés dans les documents :\n\n"
        return intro + "\n".join(synthesized_parts)
