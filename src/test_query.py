from src.retriever import Recherche

retr = Recherche()

query = "Quelles sont les principales idées du document ?"

results = retr.query(query, k=5)

for r in results:
    print("----")
    print("Document :", r['source'])
    print("Score :", r['score'])
    print("Snippet :", r['content'])
