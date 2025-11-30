"""LLM-as-Judge evaluator for RAG systems."""

import json
import logging
from typing import Optional

from src.invoke_ai import invoke_ai
from src.extract_xml import extract_xml_tag


logger = logging.getLogger(__name__)


JSON_SYSTEM_PROMPT = """
You are an evaluator for Retrieval-Augmented Generation (RAG) systems.

You will receive the following fields in the user message: question, answer, context, expected_answer.
Your job is to score three metrics in the range [0.0, 1.0] and provide a short reasoning string.

Return output STRICTLY as a single JSON object with the following schema:
{
  "correctness": float,         # 0.0 to 1.0 (agreement with expected_answer)
  "groundedness": float,       # 0.0 to 1.0 (does the answer only use facts in context?)
  "context_relevance": float,   # 0.0 to 1.0 (is the provided context sufficient/necessary?)
  "reasoning": str              # short human-readable justification (<= 200 words)
}

Scoring guidance (rubric):
- Correctness: 1.0 = answer fully matches expected_answer facts; 0.5 = partially correct; 0.0 = incorrect.
- Groundedness: 1.0 = every factual claim in the answer is present in the provided context; 0.0 = answer invents facts not present.
- Context Relevance: 1.0 = the retrieved context contains the specific evidence needed to answer; 0.0 = context is unrelated.

Always produce valid JSON. If you cannot determine a score, use 0.0.
Keep the reasoning concise.
"""


class LLMJudgeEvaluator:
    """Evaluate model outputs with an LLM judge.

    This implementation prefers JSON outputs from the LLM. If parsing fails, it
    falls back to the simple XML tag extractor available in `src.extract_xml`.
    """

    def __init__(self, system_prompt: Optional[str] = None, model: Optional[str] = None):
        self.system_prompt = system_prompt or JSON_SYSTEM_PROMPT
        self.model = model

    @staticmethod
    def _clamp_score(x: Optional[float]) -> float:
        try:
            v = float(x)
        except Exception:
            return 0.0
        if v != v:  # NaN
            return 0.0
        return max(0.0, min(1.0, v))

    def _parse_json_or_fallback(self, llm_output: str):
        # Try JSON first
        try:
            obj = json.loads(llm_output.strip())
            correctness = self._clamp_score(obj.get("correctness"))
            groundedness = self._clamp_score(obj.get("groundedness"))
            context_relevance = self._clamp_score(obj.get("context_relevance"))
            reasoning = obj.get("reasoning", "").strip()
            return correctness, groundedness, context_relevance, reasoning
        except Exception:
            # Fallback: try XML-style tags using the existing helper
            try:
                correctness = self._clamp_score(extract_xml_tag(llm_output, "correctness"))
                groundedness = self._clamp_score(extract_xml_tag(llm_output, "groundedness"))
                context_relevance = self._clamp_score(extract_xml_tag(llm_output, "context_relevance"))
                reasoning = extract_xml_tag(llm_output, "reasoning") or ""
                return correctness, groundedness, context_relevance, reasoning
            except Exception:
                logger.exception("Failed to parse LLM output")
                return 0.0, 0.0, 0.0, ""

    def evaluate(self, question: str, answer: str, context: str, expected_answer: str) -> dict:
        """Evaluate a RAG response using an LLM judge.
        
        Args:
            question: The user's question
            answer: The RAG system's answer
            context: The retrieved context used to generate the answer
            expected_answer: The reference/expected answer
        
        Returns:
            Dictionary with correctness, groundedness, context_relevance scores and reasoning
        """
        user_message = json.dumps({
            "question": question,
            "answer": answer,
            "context": context,
            "expected_answer": expected_answer,
        }, ensure_ascii=False)

        try:
            llm_output = invoke_ai(
                system_message=self.system_prompt, 
                user_message=user_message,
                model=self.model,
                temperature=0.0,
                max_tokens=1024
            )
            
            # Debug: print raw output
            print(f"    [DEBUG] LLM raw output length: {len(llm_output)} chars")
            if not llm_output or llm_output.strip() == "":
                print(f"    [WARNING] LLM returned empty response!")
                llm_output = '{"correctness": 0.0, "groundedness": 0.0, "context_relevance": 0.0, "reasoning": "LLM returned empty response"}'
                
        except Exception as e:
            print(f"    [ERROR] LLM invocation failed: {e}")
            llm_output = f'{{"correctness": 0.0, "groundedness": 0.0, "context_relevance": 0.0, "reasoning": "Error: {str(e)}"}}'

        correctness, groundedness, context_relevance, reasoning = self._parse_json_or_fallback(llm_output)

        return {
            "correctness": correctness,
            "groundedness": groundedness,
            "context_relevance": context_relevance,
            "reasoning": reasoning,
            "raw_llm_output": llm_output,
        }
