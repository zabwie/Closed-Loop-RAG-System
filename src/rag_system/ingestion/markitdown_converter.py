"""MarkItDownConverter for converting documents to Markdown."""

from markitdown import MarkItDown
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class MarkItDownConverter:
    """Converter for transforming various document formats to Markdown.

    Supports PDF, Word, Excel, PowerPoint, HTML, and other formats
    using the Microsoft MarkItDown library.
    """

    def __init__(self):
        """Initialize the MarkItDownConverter."""
        self.md = MarkItDown()

    def convert(self, file_path: Path) -> Dict[str, Any]:
        """Convert document to Markdown with metadata.

        Args:
            file_path: Path to the document file.

        Returns:
            Dictionary containing:
                - markdown: The converted Markdown content.
                - metadata: Dictionary with source, format, char_count, word_count.

        Raises:
            Exception: If conversion fails (corrupted file, encoding error, etc.).
        """
        try:
            result = self.md.convert(str(file_path))
            return {
                "markdown": result.text_content,
                "metadata": {
                    "source": str(file_path.name),
                    "format": file_path.suffix.lstrip("."),
                    "char_count": len(result.text_content),
                    "word_count": len(result.text_content.split()),
                },
            }
        except Exception as e:
            logger.error(f"Failed to convert {file_path}: {e}")
            raise
