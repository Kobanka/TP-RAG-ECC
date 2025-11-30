# 🧠 TP ECC - Retrieval Augmented Generation (RAG)

## 📘 Description du projet

Ce projet constitue un **travail pratique** dans le cadre du module d’**Intelligence Artificielle Générative** à l’École Centrale Casablanca.
L’objectif principal est de concevoir un **système complet de Retrieval-Augmented Generation (RAG)** permettant à un modèle de langage (LLM) d’accéder à des **informations externes** stockées dans une base vectorielle afin de générer des réponses pertinentes, contextualisées et explicables.

Le projet se fonde sur les principes du **pipeline RAG**, qui combine :

1. **Indexation et vectorisation** de documents textuels (PDF, Markdown, etc.)
2. **Recherche sémantique** dans une base de vecteurs à partir d’une requête utilisateur
3. **Génération de réponses** par un **LLM** (Large Language Model) guidé par le contexte récupéré
4. **Évaluation de la pertinence** des réponses produites
5. **Discuter de façon interactive** via un chatbot avec historique.

---

## 🎯 Objectifs pédagogiques

À travers ce TP, les étudiants développent :

* Une compréhension approfondie des systèmes RAG et de leur architecture.
* Des compétences pratiques dans l’utilisation de **LangChain**, **ChromaDB**, et **Hugging Face**.
* Une maîtrise de la **programmation orientée objet (POO)** en Python appliquée à un système d’IA.
* La capacité à **structurer un projet IA complet**, reproductible et exécutable en ligne de commande.

---

## ⚙️ Fonctionnalités principales

### 🔹 Q1 : Indexation des documents
- Chargement des PDF avec `PyPDFium2Loader` (meilleure extraction que `PyPDFLoader`)
- Découpage intelligent avec `RecursiveCharacterTextSplitter` (séparateurs Markdown)
- Embeddings avec `sentence-transformers/all-mpnet-base-v2`
- Stockage persistant dans **ChromaDB**
- 
### 🔹 Q2 : Recherche documentaire
- Récupération des `k` chunks les plus similaires
- Renvoi du contenu complet + métadonnées (`source`, `page`, `score`)

### 🔹 Q3 : Système de question-réponse (RAG complet)
- Synthèse du contexte via `ContextSynthesizer`
- Prompt template dédié (`prompts.py`) avec rôle expert et consignes strictes
- LLM open-source via OpenRouter (`mistralai/mistral-7b-instruct:free`)

### 🔹 Q4 : Évaluation du système
- Exact Match (normalisé)
- F1 token-level (style SQuAD)
- Similarité cosinus avec Sentence-BERT

### 🔹 Q5 (Bonus) : Chatbot conversationnel
- Historique des échanges (limité à `max_history`)
- Commandes interactives (`reset`, `history`, `save`)
- Contexte enrichi avec l’historique de conversation

---

## 🧩 Architecture du projet

```
TP-RAG-ECC/
│
├── data/                       # Fichiers PDF ou Markdown à indexer
├── src/                        # Code source principal
│   ├── document_indexer.py     # Classe pour l’indexation des documents
│   ├── retriever.py            # Module de recherche vectorielle
│   ├── synthesis.py            # ContextSynthesizer
│   ├── prompts.py              # Prompt template structuré
│   ├── RAG_ChatBot.py          # RAGQuestionAnswering + historique
│   ├── evaluator.py            # Evaluation quantitative
│   └── run_index.py            # Script utilitaire pour l’indexation
│
├── config.yaml                 # Fichier de configuration (modèles, paramètres, chemins)
├── cli.py                      # Interface en ligne de commande (index, query, chat, evaluate)
├── requirements.txt            # Dépendances Python
├── .env.example                # Exemple de fichier pour la clé API
└── README.md                   # Documentation (ce fichier)

```

---

## 🧠 Technologies et bibliothèques
- **Framework** : LangChain
- **Vector store** : ChromaDB
- **Embeddings** : Hugging Face (`sentence-transformers/all-mpnet-base-v2`)
- **LLM** : Mistral-7B-Instruct (via OpenRouter)
- **PDF Loader** : PyPDFium2
- **Evaluation** : Sentence-BERT, implémentation custom de EM/F1
- **CLI** : `argparse`, `python-dotenv`

## 💾 Installation

1. Cloner le dépôt :  
   ```bash
   git clone https://github.com/Kobanka/TP-RAG-ECC.git
   cd TP-RAG-ECC
2. Installer les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

3. Configurer la clé API OpenRouter :
    ```bash
    cp .env.example .env
    ```
    Puis éditer le fichier `.env` et insérer votre clé :
    ```env
    OPENROUTER_API_KEY=votre_clé_ici
    ```
    
## 🧑‍💻 Utilisation
Le système peut être utilisé via la CLI avec les commandes suivantes :

- **Indexer les documents** :  
  ```bash
  python cli.py index
  ```

- **Poser une question** :  
  ```bash
  python cli.py query -q "What is the difference between traditional banking and trading from the documents, explain ?"
  ```

- **Lancer le chatbot interactif** :  
  ```bash
  python cli.py chat
  ```

- **Évaluer une réponse** :  
  ```bash
  python cli.py evaluate --reference "..." --prediction "..."
  ```

**Exemples de questions pertinentes :**  
- "Quels sont les points principaux discutés dans ces documents ?"  
- "Quelle méthode est utilisée pour l'analyse ?"  

---

## 🧪 Évaluation

Le système inclut un module d’évaluation (`evaluator.py`) qui calcule trois métriques principales :  
- **Exact Match** (comparaison textuelle normalisée)  
- **F1 score** (intersection des tokens entre référence et prédiction)  
- **Similarité sémantique** (cosinus entre embeddings `all-mpnet-base-v2`)  

Ces métriques permettent d’évaluer à la fois la **fidélité factuelle** et la **qualité de reformulation** des réponses générées.


## 🚀 Améliorations possibles
 
- Filtrage dynamique des chunks par seuil de similarité  
- Interface web interactive (Gradio ou Streamlit)  
- Fine-tuning du LLM sur le domaine spécifique de Trading

---

## 👥 Auteurs

Projet réalisé dans le cadre du TP “Retrieval-Augmented Generation” à l’**Ecole Centrale Casablanca**, sous la supervision de **Mr. Imad LAKIM**.

**Membres de l’équipe :**

* DERBANI Salwa - [@sader04](https://github.com/sader04)
* KANOHA ELENGA Jihane - [@KANOHA242](https://github.com/KANOHA242)
* KOBANKA Anicet - [@Kobanka](https://github.com/Kobanka)
* KOUDIA Selma - [@selmakoudia03](https://github.com/selmakoudia03)
* LARAISSE Hamza - [@laraisse](https://github.com/laraisse)
* TACHRIFT Imane  - [@Imantsh1](https://github.com/Imantsh1)

---
