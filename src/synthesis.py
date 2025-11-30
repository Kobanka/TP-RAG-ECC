import yaml
from typing import List, Dict, Any

class ContextSynthesizer:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)["synthesizer"]

        self.max_chunk = cfg["max_chunk_size"]
        self.intro = cfg["intro_message"]
        self.default_source = cfg["default_source"]
        self.default_page = cfg["default_page"]
        self.score_precision = cfg["score_precision"]

    def synthesize(self, chunks: List[Dict[str, Any]]) -> str:

        if not chunks:
            return "Aucun document pertinent n'a été trouvé."

        parts = []

        for i, chunk in enumerate(chunks, 1):
            content = chunk.get("content", "").strip()
            source = chunk.get("source", self.default_source)
            page = chunk.get("page", self.default_page)
            score = chunk.get("score", 0.0)

            # Réduction configurable
            if len(content) > self.max_chunk:
                content = content[:self.max_chunk] + " [...]"

            parts.append(
                f"[Document {i}] Source: {source} | Page: {page} | Score: {score:.{self.score_precision}f}\n"
                f"{content}\n"
            )

        return self.intro + "\n\n" + "\n".join(parts)
