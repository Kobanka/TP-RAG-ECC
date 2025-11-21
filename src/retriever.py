# src/retriever.py
from typing import List, Dict, Any
from pathlib import Path
import yaml

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


class DocumentRetriever:
    """
    Interroge le vector store Chroma et agrège les chunks par document (file).
    Renvoie une liste de documents les plus pertinents avec score et snippet.
    """

    def __init__(self, config_path: str = "config.yaml"):
        cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
        self.persist_directory = cfg["persist_directory"]
        self.embedding_model_name = cfg["embedding_model"]
        self.top_k = int(cfg.get("top_k", 5))
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

        # charge la DB Chroma déjà persistée
        self.vectordb = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)

    def query(self, query_text: str, k: int = None) -> List[Dict[str, Any]]:
        """
        Interroge la DB et renvoie des résultats agrégés par fichier source.
        Format de sortie: [
            {"file": "ia_article1.pdf", "score": 0.87, "snippet": "best chunk text..."},
            ...
        ]
        score ici = max similarity among chunks for that file (simple et efficace).
        """
        if k is None:
            k = self.top_k

        # Chroma return: list of documents and distances/scores
        results = self.vectordb.similarity_search_with_score(query_text, k=k, include_metadata=True)

        # results: list of (Document, score) where Document.page_content and metadata available
        # Aggregate by metadata['source'] using max score and keep best snippet
        agg = {}
        for doc, score in results:
            src = doc.metadata.get("source", "unknown")
            # Chroma returns distance; in LangChain usually smaller = better (distance).
            # We'll convert to similarity-like score for readability: sim = 1 / (1 + distance)
            sim_score = 1.0 / (1.0 + float(score))
            if src not in agg or sim_score > agg[src]["score"]:
                agg[src] = {
                    "file": src,
                    "score": sim_score,
                    "snippet": doc.page_content[:800]  # truncated snippet
                }

        # sort by descending score
        sorted_results = sorted(agg.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results
