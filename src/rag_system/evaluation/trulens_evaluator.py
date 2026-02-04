"""Simulated TruLens-style evaluation for self-hosted demo."""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class SimulatedEvaluator:
    """Simulated TruLens-style evaluation for self-hosted demo.

    This class provides heuristic-based evaluation metrics that simulate
    TruLens evaluation without requiring external API calls or complex
    instrumentation. It implements the RAG triad metrics:
    - Faithfulness: How well the answer is supported by retrieved sources
    - Context Precision: Quality of retrieved sources
    - Context Recall: Coverage of retrieved sources
    - Answer Relevance: How well the answer addresses the query
    """

    async def evaluate_query(self, query: str, response: dict) -> dict:
        """Simulate evaluation scores based on heuristics.

        Args:
            query: The user's query string
            response: RAG engine response containing 'answer' and 'sources'

        Returns:
            Dictionary with evaluation metrics:
            - faithfulness: Answer-source overlap score (0-1)
            - context_precision: Average source score (0-1)
            - context_recall: Retrieved count / ideal count (0-1)
            - answer_relevance: Query-answer overlap score (0-1)
            - overall_score: Weighted average of all metrics (0-1)
        """
        sources = response.get("sources", [])

        # Heuristic: faithfulness based on answer overlap with sources
        answer = response.get("answer", "")
        sources_text = " ".join([s["text"] for s in sources])

        # Simple word overlap as proxy
        answer_words = set(answer.lower().split())
        sources_words = set(sources_text.lower().split())
        overlap = len(answer_words & sources_words)
        faithfulness = min(overlap / max(len(answer_words), 1), 1.0)

        # Heuristic: context precision based on source scores
        if sources:
            context_precision = sum([s["score"] for s in sources]) / len(sources)
        else:
            context_precision = 0.0

        # Heuristic: context recall based on retrieved count
        # Assuming 5 is ideal number of sources
        context_recall = min(len(sources) / 5, 1.0)

        # Heuristic: answer relevance based on query-answer overlap
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        relevance = len(query_words & answer_words) / max(len(query_words), 1)

        # Calculate overall weighted score
        overall_score = (
            faithfulness * 0.3 + relevance * 0.3 + context_precision * 0.2 + context_recall * 0.2
        )

        return {
            "faithfulness": faithfulness,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "answer_relevance": relevance,
            "overall_score": overall_score,
        }
