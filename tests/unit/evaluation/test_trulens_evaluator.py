"""Unit tests for SimulatedEvaluator."""

import pytest
from src.rag_system.evaluation.trulens_evaluator import SimulatedEvaluator


@pytest.fixture
def evaluator():
    """Create a SimulatedEvaluator instance."""
    return SimulatedEvaluator()


@pytest.mark.asyncio
async def test_evaluate_query_basic(evaluator):
    """Test basic evaluation with a simple query and response."""
    query = "What is RAG?"
    response = {
        "answer": "RAG stands for Retrieval-Augmented Generation, a technique that combines retrieval and generation.",
        "sources": [
            {"text": "RAG is Retrieval-Augmented Generation", "score": 0.9},
            {"text": "It combines retrieval and generation", "score": 0.85},
        ],
    }

    result = await evaluator.evaluate_query(query, response)

    # Verify all expected metrics are present
    assert "faithfulness" in result
    assert "context_precision" in result
    assert "context_recall" in result
    assert "answer_relevance" in result
    assert "overall_score" in result

    # Verify all scores are between 0 and 1
    assert 0 <= result["faithfulness"] <= 1
    assert 0 <= result["context_precision"] <= 1
    assert 0 <= result["context_recall"] <= 1
    assert 0 <= result["answer_relevance"] <= 1
    assert 0 <= result["overall_score"] <= 1


@pytest.mark.asyncio
async def test_evaluate_query_high_overlap(evaluator):
    """Test evaluation with high answer-source overlap (high faithfulness)."""
    query = "What is machine learning?"
    response = {
        "answer": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "sources": [
            {
                "text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                "score": 0.95,
            },
            {"text": "AI systems use machine learning algorithms", "score": 0.9},
        ],
    }

    result = await evaluator.evaluate_query(query, response)

    # High overlap should result in high faithfulness
    assert result["faithfulness"] > 0.5
    # High source scores should result in high context precision
    assert result["context_precision"] > 0.8


@pytest.mark.asyncio
async def test_evaluate_query_low_overlap(evaluator):
    """Test evaluation with low answer-source overlap (low faithfulness)."""
    query = "What is Python?"
    response = {
        "answer": "Python is a programming language used for web development.",
        "sources": [
            {"text": "Java is a programming language", "score": 0.8},
            {"text": "C++ is used for system programming", "score": 0.75},
        ],
    }

    result = await evaluator.evaluate_query(query, response)

    # Low overlap should result in moderate faithfulness (common words like "is", "a", "programming", "language" create overlap)
    # The heuristic is simple word overlap, so it's not perfect
    assert result["faithfulness"] < 0.8


@pytest.mark.asyncio
async def test_evaluate_query_high_relevance(evaluator):
    """Test evaluation with high query-answer relevance."""
    query = "What is deep learning?"
    response = {
        "answer": "Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
        "sources": [{"text": "Deep learning uses neural networks", "score": 0.9}],
    }

    result = await evaluator.evaluate_query(query, response)

    # High query-answer overlap should result in moderate to high relevance
    # Query has 4 words, answer shares 2 ("deep", "learning"), so relevance = 0.5
    assert result["answer_relevance"] >= 0.5


@pytest.mark.asyncio
async def test_evaluate_query_low_relevance(evaluator):
    """Test evaluation with low query-answer relevance."""
    query = "What is the capital of France?"
    response = {
        "answer": "Python is a programming language created by Guido van Rossum.",
        "sources": [{"text": "Python programming language", "score": 0.9}],
    }

    result = await evaluator.evaluate_query(query, response)

    # Low query-answer overlap should result in low relevance
    assert result["answer_relevance"] < 0.3


@pytest.mark.asyncio
async def test_evaluate_query_context_recall(evaluator):
    """Test context recall based on number of sources."""
    query = "Test query"
    response = {
        "answer": "Test answer",
        "sources": [
            {"text": "Source 1", "score": 0.9},
            {"text": "Source 2", "score": 0.85},
            {"text": "Source 3", "score": 0.8},
            {"text": "Source 4", "score": 0.75},
            {"text": "Source 5", "score": 0.7},
        ],
    }

    result = await evaluator.evaluate_query(query, response)

    # 5 sources should result in context_recall of 1.0
    assert result["context_recall"] == 1.0


@pytest.mark.asyncio
async def test_evaluate_query_context_recall_few_sources(evaluator):
    """Test context recall with fewer sources."""
    query = "Test query"
    response = {"answer": "Test answer", "sources": [{"text": "Source 1", "score": 0.9}]}

    result = await evaluator.evaluate_query(query, response)

    # 1 source should result in context_recall of 0.2 (1/5)
    assert result["context_recall"] == 0.2


@pytest.mark.asyncio
async def test_evaluate_query_empty_sources(evaluator):
    """Test evaluation with empty sources."""
    query = "Test query"
    response = {"answer": "Test answer", "sources": []}

    result = await evaluator.evaluate_query(query, response)

    # Empty sources should result in zero context metrics
    assert result["context_precision"] == 0
    assert result["context_recall"] == 0
    # Faithfulness should be 0 since no sources to overlap with
    assert result["faithfulness"] == 0


@pytest.mark.asyncio
async def test_evaluate_query_empty_answer(evaluator):
    """Test evaluation with empty answer."""
    query = "Test query"
    response = {"answer": "", "sources": [{"text": "Source text", "score": 0.9}]}

    result = await evaluator.evaluate_query(query, response)

    # Empty answer should result in zero faithfulness and relevance
    assert result["faithfulness"] == 0
    assert result["answer_relevance"] == 0


@pytest.mark.asyncio
async def test_evaluate_query_overall_score_calculation(evaluator):
    """Test that overall_score is calculated correctly."""
    query = "Test query"
    response = {
        "answer": "Test answer with some words",
        "sources": [{"text": "Test answer with some words", "score": 0.9}],
    }

    result = await evaluator.evaluate_query(query, response)

    # Verify overall_score is weighted average
    expected_overall = (
        result["faithfulness"] * 0.3
        + result["answer_relevance"] * 0.3
        + result["context_precision"] * 0.2
        + result["context_recall"] * 0.2
    )
    assert abs(result["overall_score"] - expected_overall) < 0.001
