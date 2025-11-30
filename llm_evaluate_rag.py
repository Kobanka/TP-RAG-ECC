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
from src.llm_judge import LLMJudgeEvaluator
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
                       evaluator: LLMJudgeEvaluator, retriever: Recherche, k_retrieve: int) -> List[Dict]:
    """Evaluate RAG system on selected questions"""
    results = []
    
    print(f"\nEvaluating {len(questions)} questions...")
    print("=" * 70)
    
    for i, qa_pair in enumerate(questions, 1):
        question = qa_pair["question"]
        reference = qa_pair["answer"]
        
        print(f"\n[{i}/{len(questions)}] Question: {question[:80]}...")
        
        # Get retrieved context
        try:
            retrieved_docs = retriever.query(question, k=k_retrieve)
            context = "\n\n".join([f"[Doc {j+1}] {doc['content'][:300]}..." 
                                   for j, doc in enumerate(retrieved_docs)])
        except Exception as e:
            print(f"  ERROR: Failed to retrieve context: {e}")
            context = ""
        
        # Get RAG prediction
        try:
            prediction = rag.answer(question, k=k_retrieve)
        except Exception as e:
            print(f"  ERROR: Failed to get prediction: {e}")
            prediction = ""
        
        # Evaluate with LLM judge
        try:
            metrics = evaluator.evaluate(
                question=question,
                answer=prediction,
                context=context,
                expected_answer=reference
            )
        except Exception as e:
            print(f"  ERROR: Failed to evaluate: {e}")
            metrics = {
                "correctness": 0.0,
                "groundedness": 0.0,
                "context_relevance": 0.0,
                "reasoning": f"Evaluation failed: {str(e)}"
            }
        
        # Store result
        result = {
            "question": question,
            "reference": reference,
            "prediction": prediction,
            "context_preview": context[:500] + "..." if len(context) > 500 else context,
            "metrics": metrics
        }
        results.append(result)
        
        # Print metrics
        print(f"  Correctness: {metrics['correctness']:.3f}")
        print(f"  Groundedness: {metrics['groundedness']:.3f}")
        print(f"  Context Relevance: {metrics['context_relevance']:.3f}")
        print(f"  Reasoning: {metrics.get('reasoning', 'N/A')[:100]}...")
    
    return results


def compute_summary_metrics(results: List[Dict]) -> Dict[str, Any]:
    """Compute average metrics across all results"""
    if not results:
        return {}
    
    total_correctness = sum(r["metrics"]["correctness"] for r in results)
    total_groundedness = sum(r["metrics"]["groundedness"] for r in results)
    total_context_relevance = sum(r["metrics"]["context_relevance"] for r in results)
    
    n = len(results)
    
    return {
        "num_questions": n,
        "avg_correctness": total_correctness / n,
        "avg_groundedness": total_groundedness / n,
        "avg_context_relevance": total_context_relevance / n
    }


def find_lowest_scoring(results: List[Dict], top_n: int = 5) -> List[Dict]:
    """Find questions with lowest correctness scores"""
    sorted_results = sorted(results, key=lambda x: x["metrics"]["correctness"])
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
    print("EVALUATION SUMMARY (LLM-as-Judge)")
    print("=" * 70)
    print(f"Total Questions Evaluated: {summary['num_questions']}")
    print(f"Average Correctness: {summary['avg_correctness']:.3f}")
    print(f"Average Groundedness: {summary['avg_groundedness']:.3f}")
    print(f"Average Context Relevance: {summary['avg_context_relevance']:.3f}")
    
    print("\n" + "-" * 70)
    print("LOWEST SCORING QUESTIONS (by correctness, for improvement)")
    print("-" * 70)
    for i, result in enumerate(lowest_scoring, 1):
        print(f"\n{i}. Question: {result['question'][:80]}...")
        print(f"   Correctness: {result['metrics']['correctness']:.3f}")
        print(f"   Groundedness: {result['metrics']['groundedness']:.3f}")
        print(f"   Context Relevance: {result['metrics']['context_relevance']:.3f}")
        reasoning = result['metrics'].get('reasoning', 'N/A')
        print(f"   Reasoning: {reasoning[:150]}...")
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
    
    # Initialize LLM Judge Evaluator
    try:
        evaluator = LLMJudgeEvaluator(model=config['llm']['model'])
        print("✓ LLM Judge Evaluator initialized")
    except Exception as e:
        print(f"✗ ERROR initializing LLM Judge: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Run evaluation
    start_time = datetime.now()
    results = evaluate_rag_system(selected_questions, rag, evaluator, retriever, k_retrieve)
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
