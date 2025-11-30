#!/usr/bin/env python3
"""
Automated RAG Evaluation Script

Evaluates the RAG system against questions from sample_questions.json
and generates a detailed report with metrics.

Usage:
    python evaluate_rag.py --num 5        # Evaluate first 5 questions
    python evaluate_rag.py --all          # Evaluate all questions
    python evaluate_rag.py --random 10    # Evaluate 10 random questions
"""

import argparse
import json
import yaml
import os
import sys
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv(".env")

from src.retriever import Recherche
from src.RAG_ChatBot import RAGQuestionAnswering
from src.evaluator import Evaluator
from langchain_openai import ChatOpenAI


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_llm(config):
    """Initialize the LLM"""
    api_key_env = config['llm']['api_key_env']
    openrouter_api_key = os.getenv(api_key_env)

    if not openrouter_api_key:
        print(f"Error: {api_key_env} not found")
        sys.exit(1)

    llm = ChatOpenAI(
        api_key=openrouter_api_key,
        base_url=config['llm']['base_url'],
        model=config['llm']['model'],
        temperature=config['llm']['temperature']
    )
    return llm


def load_questions(questions_file="data/sample_questions.json"):
    """Load questions from JSON file"""
    with open(questions_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def select_questions(questions: List[Dict], mode: str, num: int = None) -> List[Dict]:
    """Select questions based on mode"""
    if mode == "all":
        return questions
    elif mode == "num":
        return questions[:num]
    elif mode == "random":
        return random.sample(questions, min(num, len(questions)))
    else:
        return questions


def evaluate_rag_system(questions: List[Dict], rag: RAGQuestionAnswering, 
                       evaluator: Evaluator, k_retrieve: int) -> List[Dict]:
    """Evaluate RAG system on selected questions"""
    results = []
    
    print(f"\nEvaluating {len(questions)} questions...")
    print("=" * 70)
    
    for i, qa_pair in enumerate(questions, 1):
        question = qa_pair["question"]
        reference = qa_pair["answer"]
        
        print(f"\n[{i}/{len(questions)}] Question: {question[:80]}...")
        
        # Get RAG prediction
        try:
            prediction = rag.answer(question, k=k_retrieve)
        except Exception as e:
            print(f"  ERROR: Failed to get prediction: {e}")
            prediction = ""
        
        # Compute metrics
        try:
            metrics = evaluator.evaluate_pair(reference, prediction)
        except Exception as e:
            print(f"  ERROR: Failed to compute metrics: {e}")
            metrics = {"exact_match": 0, "f1": 0.0, "semantic_similarity": 0.0}
        
        # Store result
        result = {
            "question": question,
            "reference": reference,
            "prediction": prediction,
            "metrics": metrics
        }
        results.append(result)
        
        # Print metrics
        print(f"  Exact Match: {metrics['exact_match']}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  Semantic Similarity: {metrics['semantic_similarity']:.4f}")
    
    return results


def compute_summary_metrics(results: List[Dict]) -> Dict[str, Any]:
    """Compute average metrics across all results"""
    if not results:
        return {}
    
    total_exact_match = sum(r["metrics"]["exact_match"] for r in results)
    total_f1 = sum(r["metrics"]["f1"] for r in results)
    total_similarity = sum(r["metrics"]["semantic_similarity"] for r in results)
    
    n = len(results)
    
    return {
        "num_questions": n,
        "avg_exact_match": total_exact_match / n,
        "avg_f1": total_f1 / n,
        "avg_semantic_similarity": total_similarity / n
    }


def find_lowest_scoring(results: List[Dict], top_n: int = 5) -> List[Dict]:
    """Find questions with lowest scores"""
    sorted_results = sorted(results, key=lambda x: x["metrics"]["semantic_similarity"])
    return sorted_results[:top_n]


def save_results(results: List[Dict], summary: Dict[str, Any], 
                eval_params: Dict[str, Any], output_file: str = "evaluation_results.json"):
    """Save evaluation results to JSON file"""
    output = {
        "timestamp": datetime.now().isoformat(),
        "evaluation_parameters": eval_params,
        "summary_metrics": summary,
        "detailed_results": results,
        "lowest_scoring_questions": find_lowest_scoring(results)
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to {output_file}")


def print_summary(summary: Dict[str, Any], lowest_scoring: List[Dict]):
    """Print summary to console"""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total Questions Evaluated: {summary['num_questions']}")
    print(f"Average Exact Match: {summary['avg_exact_match']:.2%}")
    print(f"Average F1 Score: {summary['avg_f1']:.4f}")
    print(f"Average Semantic Similarity: {summary['avg_semantic_similarity']:.4f}")
    
    print("\n" + "-" * 70)
    print("LOWEST SCORING QUESTIONS (for improvement)")
    print("-" * 70)
    for i, result in enumerate(lowest_scoring, 1):
        print(f"\n{i}. Question: {result['question'][:80]}...")
        print(f"   Semantic Similarity: {result['metrics']['semantic_similarity']:.4f}")
        print(f"   F1 Score: {result['metrics']['f1']:.4f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Automated RAG Evaluation")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--num', type=int, help='Evaluate first N questions')
    group.add_argument('--all', action='store_true', help='Evaluate all questions')
    group.add_argument('--random', type=int, help='Evaluate N random questions')
    
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Path to config file')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Determine evaluation mode
    if args.all:
        mode = "all"
        num = None
    elif args.num:
        mode = "num"
        num = args.num
    elif args.random:
        mode = "random"
        num = args.random
    else:
        # Default: evaluate all
        mode = "all"
        num = None
    
    print("=" * 70)
    print("RAG SYSTEM AUTOMATED EVALUATION")
    print("=" * 70)
    
    # Load configuration
    config = load_config(args.config)
    
    # Load questions
    questions = load_questions()
    selected_questions = select_questions(questions, mode, num)
    
    print(f"Mode: {mode}")
    print(f"Total questions available: {len(questions)}")
    print(f"Questions to evaluate: {len(selected_questions)}")
    
    # Initialize components
    print("\nInitializing RAG system...")
    try:
        persist_dir = config['indexation']['persist_dir']
        retriever = Recherche(persist_dir=persist_dir)
        print("✓ Retriever initialized")
    except Exception as e:
        print(f"✗ ERROR initializing retriever: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    try:
        llm = load_llm(config)
        print("✓ LLM initialized")
    except Exception as e:
        print(f"✗ ERROR initializing LLM: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    max_history = config['rag']['max_history']
    k_retrieve = config['rag']['k_retrieve']
    
    try:
        rag = RAGQuestionAnswering(
            retriever=retriever,
            llm=llm,
            max_history=max_history
        )
        print("✓ RAG initialized")
    except Exception as e:
        print(f"✗ ERROR initializing RAG: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    try:
        evaluator = Evaluator()
        print("✓ Evaluator initialized")
    except Exception as e:
        print(f"✗ ERROR initializing Evaluator: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Run evaluation
    start_time = datetime.now()
    results = evaluate_rag_system(selected_questions, rag, evaluator, k_retrieve)
    end_time = datetime.now()
    
    # Compute summary
    summary = compute_summary_metrics(results)
    lowest_scoring = find_lowest_scoring(results)
    
    # Evaluation parameters
    eval_params = {
        "mode": mode,
        "num_questions_evaluated": len(selected_questions),
        "k_retrieve": k_retrieve,
        "execution_time_seconds": (end_time - start_time).total_seconds()
    }
    
    # Save and print results
    save_results(results, summary, eval_params, args.output)
    print_summary(summary, lowest_scoring)
    
    print(f"\nExecution time: {eval_params['execution_time_seconds']:.2f} seconds")


if __name__ == "__main__":
    main()
