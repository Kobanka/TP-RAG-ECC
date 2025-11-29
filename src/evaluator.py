import re
import string
from typing import List, Dict, Any, Union, Optional

import numpy as np
from sentence_transformers import SentenceTransformer, util


def _normalize_text(s: str) -> str:
	"""Lowercase, remove punctuation and extra whitespace."""
	if s is None:
		return ""
	s = s.lower()
	# remove punctuation
	s = s.translate(str.maketrans("", "", string.punctuation))
	# collapse whitespace
	s = re.sub(r"\s+", " ", s).strip()
	return s


def _tokens(s: str) -> List[str]:
	s = _normalize_text(s)
	if s == "":
		return []
	return s.split()


def _f1_score_single(reference: str, prediction: str) -> float:
	"""Token-level F1 (SQuAD-style) between a single ref and a single pred."""
	ref_tokens = _tokens(reference)
	pred_tokens = _tokens(prediction)
	if not ref_tokens and not pred_tokens:
		return 1.0
	if not ref_tokens or not pred_tokens:
		return 0.0

	# count overlap
	from collections import Counter

	ref_counts = Counter(ref_tokens)
	pred_counts = Counter(pred_tokens)
	common = sum(min(ref_counts[t], pred_counts[t]) for t in ref_counts)
	if common == 0:
		return 0.0

	precision = common / sum(pred_counts.values())
	recall = common / sum(ref_counts.values())
	if precision + recall == 0:
		return 0.0
	return 2 * precision * recall / (precision + recall)


class Evaluator:
	"""Evaluator for QA responses.

	Provides Exact Match (EM), token-level F1 and embedding-based
	semantic similarity. Accepts multiple references per example and
	aggregates scores over a batch.
	"""

	def __init__(self, embed_model_name: str = "all-mpnet-base-v2"):
		self.embed_model_name = embed_model_name
		# lazy-load the model to keep startup cheap if not used
		self._embed_model: Optional[SentenceTransformer] = None

	@property
	def embed_model(self) -> SentenceTransformer:
		if self._embed_model is None:
			self._embed_model = SentenceTransformer(self.embed_model_name)
		return self._embed_model

	def exact_match(self, references: Union[str, List[str]], prediction: str) -> int:
		"""Return 1 if prediction exactly matches any reference after normalization.

		Args:
			references: single reference or list of references
			prediction: predicted string
		Returns:
			1 or 0
		"""
		if isinstance(references, str):
			refs = [references]
		else:
			refs = references
		pred_norm = _normalize_text(prediction)
		for r in refs:
			if pred_norm == _normalize_text(r):
				return 1
		return 0

	def f1(self, references: Union[str, List[str]], prediction: str) -> float:
		"""Return the maximum token-level F1 over provided references."""
		if isinstance(references, str):
			refs = [references]
		else:
			refs = references
		scores = [_f1_score_single(r, prediction) for r in refs]
		return max(scores) if scores else 0.0

	def semantic_similarity(self, references: Union[str, List[str]], prediction: str) -> float:
		"""Compute max cosine similarity between prediction and references using embeddings.

		Returns a float in [-1, 1]. If empty strings are provided, falls back to 0.0.
		"""
		if isinstance(references, str):
			refs = [references]
		else:
			refs = references
		refs = [r if r is not None else "" for r in refs]
		prediction = prediction if prediction is not None else ""

		# handle empty edge cases
		if all(_normalize_text(r) == "" for r in refs) and _normalize_text(prediction) == "":
			return 1.0
		if _normalize_text(prediction) == "":
			return 0.0

		# encode
		ref_embs = self.embed_model.encode(refs, convert_to_tensor=True)
		pred_emb = self.embed_model.encode(prediction, convert_to_tensor=True)
		sims = util.cos_sim(pred_emb, ref_embs).cpu().numpy()[0]
		# return the maximum similarity to any reference
		return float(np.max(sims))

	def evaluate_pair(self, reference: Union[str, List[str]], prediction: str) -> Dict[str, Any]:
		"""Evaluate a single (reference(s), prediction) pair and return metrics."""
		em = self.exact_match(reference, prediction)
		f1 = self.f1(reference, prediction)
		sem = self.semantic_similarity(reference, prediction)
		return {"exact_match": em, "f1": f1, "semantic_similarity": sem}

	def evaluate_batch(self, pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""Evaluate a batch of examples.

		Each element in `pairs` should be a dict with keys:
		  - 'reference' : str or list[str]
		  - 'prediction' : str

		Returns:
			dict with per-example metrics list and aggregated averages.
		"""
		per_example = []
		ems = []
		f1s = []
		sems = []

		# For efficiency, we can batch-encode all predictions and references later
		for ex in pairs:
			ref = ex.get("reference")
			pred = ex.get("prediction", "")
			metrics = self.evaluate_pair(ref, pred)
			per_example.append({"reference": ref, "prediction": pred, **metrics})
			ems.append(metrics["exact_match"])
			f1s.append(metrics["f1"])
			sems.append(metrics["semantic_similarity"])

		n = len(per_example)
		agg = {
			"examples": per_example,
			"aggregate": {
				"count": n,
				"exact_match": float(np.mean(ems)) if n else 0.0,
				"f1": float(np.mean(f1s)) if n else 0.0,
				"semantic_similarity": float(np.mean(sems)) if n else 0.0,
			},
		}
		return agg


if __name__ == "__main__":
	# tiny usage example
	ev = Evaluator()
	pairs = [
		{"reference": "The cat sat on the mat.", "prediction": "The cat is on the mat."},
		{"reference": ["Paris is the capital of France.", "Paris is France's capital."], "prediction": "Paris is the capital of France."},
	]
	res = ev.evaluate_batch(pairs)
	print(res)

