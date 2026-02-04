"""Ollama client for LLM generation."""

from httpx import AsyncClient
import logging

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama LLM service."""

    def __init__(self, base_url: str, model: str = "llama3:8b"):
        """Initialize the OllamaClient.

        Args:
            base_url: Base URL of the Ollama service.
            model: Name of the model to use.
        """
        self.base_url = base_url
        self.model = model

    async def chat(self, prompt: str, context: str = "") -> str:
        """Generate response using Ollama chat API.

        Args:
            prompt: The user's prompt/question.
            context: Optional context to include in the prompt.

        Returns:
            The generated response text.

        Raises:
            Exception: If chat generation fails.
        """
        rag_prompt = f"""Use the following context to answer the question.

Context:
{context}

Question:
{prompt}

Answer:"""

        try:
            async with AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": rag_prompt}],
                        "stream": False,
                    },
                    timeout=120.0,
                )
                response.raise_for_status()
                data = response.json()
                return data["message"]["content"]
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            raise
