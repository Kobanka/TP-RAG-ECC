from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class Recherche:
    def __init__(self, persist_dir="./chroma_langchain_db"):
        self.persist_dir = persist_dir
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embedding_model
        )

    def query(self, question, k=5):
        """
        Retourne les chunks + scores de similarité
        """
        results = self.vectorstore.similarity_search_with_score(question, k=k)

        # formatter clairement les résultats
        output = []
        for doc, score in results:
            output.append({
                "source": doc.metadata.get("source", "unknown"),
                "score": float(score),
                "content": doc.page_content[:300]  # snippet
            })
        return output
