"""Unit tests for MarkItDownConverter."""

import sys
from unittest.mock import MagicMock

# Mock MarkItDown before importing to avoid RuntimeWarning during test collection
sys.modules["markitdown"] = MagicMock()
sys.modules["markitdown._markitdown"] = MagicMock()

import pytest
from pathlib import Path
from rag_system.ingestion.markitdown_converter import MarkItDownConverter


class TestMarkItDownConverter:
    """Test suite for MarkItDownConverter class."""

    def test_converter_converts_pdf(self, mocker):
        """Test PDF conversion to Markdown."""
        # Arrange
        converter = MarkItDownConverter()
        file_path = Path("/tmp/test.pdf")

        # Mock MarkItDown result
        mock_result = mocker.MagicMock()
        mock_result.text_content = "# Test Document\n\nThis is a test PDF content."

        # Mock MarkItDown.convert method
        mocker.patch.object(converter.md, "convert", return_value=mock_result)

        # Act
        result = converter.convert(file_path)

        # Assert
        assert "markdown" in result
        assert "metadata" in result
        assert result["markdown"] == "# Test Document\n\nThis is a test PDF content."
        assert result["metadata"]["source"] == "test.pdf"
        assert result["metadata"]["format"] == "pdf"
        assert result["metadata"]["char_count"] == 44
        assert result["metadata"]["word_count"] == 9

    def test_converter_converts_markdown(self, mocker):
        """Test Markdown conversion."""
        # Arrange
        converter = MarkItDownConverter()
        file_path = Path("/tmp/test.md")

        # Mock MarkItDown result
        mock_result = mocker.MagicMock()
        mock_result.text_content = "# Existing Markdown\n\nContent here."

        # Mock MarkItDown.convert method
        mocker.patch.object(converter.md, "convert", return_value=mock_result)

        # Act
        result = converter.convert(file_path)

        # Assert
        assert result["markdown"] == "# Existing Markdown\n\nContent here."
        assert result["metadata"]["format"] == "md"
        assert result["metadata"]["source"] == "test.md"

    def test_converter_converts_csv(self, mocker):
        """Test CSV conversion."""
        # Arrange
        converter = MarkItDownConverter()
        file_path = Path("/tmp/test.csv")

        # Mock MarkItDown result
        mock_result = mocker.MagicMock()
        mock_result.text_content = "| Name | Age |\n|------|-----|\n| John | 30  |"

        # Mock MarkItDown.convert method
        mocker.patch.object(converter.md, "convert", return_value=mock_result)

        # Act
        result = converter.convert(file_path)

        # Assert
        assert result["markdown"] == "| Name | Age |\n|------|-----|\n| John | 30  |"
        assert result["metadata"]["format"] == "csv"
        assert result["metadata"]["source"] == "test.csv"

    def test_converter_returns_metadata(self, mocker):
        """Test that metadata is returned correctly."""
        # Arrange
        converter = MarkItDownConverter()
        file_path = Path("/tmp/document.docx")

        # Mock MarkItDown result
        mock_result = mocker.MagicMock()
        mock_result.text_content = "Test content with multiple words."

        # Mock MarkItDown.convert method
        mocker.patch.object(converter.md, "convert", return_value=mock_result)

        # Act
        result = converter.convert(file_path)

        # Assert
        assert "metadata" in result
        assert result["metadata"]["source"] == "document.docx"
        assert result["metadata"]["format"] == "docx"
        assert result["metadata"]["char_count"] == 33
        assert result["metadata"]["word_count"] == 5

    def test_converter_handles_corrupted_file(self, mocker):
        """Test error handling for corrupted files."""
        # Arrange
        converter = MarkItDownConverter()
        file_path = Path("/tmp/corrupted.pdf")

        # Mock MarkItDown.convert to raise exception
        mocker.patch.object(converter.md, "convert", side_effect=Exception("File is corrupted"))

        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            converter.convert(file_path)

        assert "File is corrupted" in str(exc_info.value)

    def test_converter_handles_encoding_errors(self, mocker):
        """Test error handling for encoding issues."""
        # Arrange
        converter = MarkItDownConverter()
        file_path = Path("/tmp/encoding_issue.txt")

        # Mock MarkItDown.convert to raise encoding error
        mocker.patch.object(
            converter.md,
            "convert",
            side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte"),
        )

        # Act & Assert
        with pytest.raises(UnicodeDecodeError):
            converter.convert(file_path)

    def test_converter_initialization(self):
        """Test that converter initializes correctly."""
        # Act
        converter = MarkItDownConverter()

        # Assert
        assert converter.md is not None

    def test_converter_converts_word_document(self, mocker):
        """Test Word document conversion."""
        # Arrange
        converter = MarkItDownConverter()
        file_path = Path("/tmp/document.docx")

        # Mock MarkItDown result
        mock_result = mocker.MagicMock()
        mock_result.text_content = "# Word Document\n\nThis is a Word document."

        # Mock MarkItDown.convert method
        mocker.patch.object(converter.md, "convert", return_value=mock_result)

        # Act
        result = converter.convert(file_path)

        # Assert
        assert result["markdown"] == "# Word Document\n\nThis is a Word document."
        assert result["metadata"]["format"] == "docx"

    def test_converter_converts_excel(self, mocker):
        """Test Excel conversion."""
        # Arrange
        converter = MarkItDownConverter()
        file_path = Path("/tmp/spreadsheet.xlsx")

        # Mock MarkItDown result
        mock_result = mocker.MagicMock()
        mock_result.text_content = "| A | B |\n|---|---|\n| 1 | 2 |"

        # Mock MarkItDown.convert method
        mocker.patch.object(converter.md, "convert", return_value=mock_result)

        # Act
        result = converter.convert(file_path)

        # Assert
        assert result["markdown"] == "| A | B |\n|---|---|\n| 1 | 2 |"
        assert result["metadata"]["format"] == "xlsx"

    def test_converter_converts_powerpoint(self, mocker):
        """Test PowerPoint conversion."""
        # Arrange
        converter = MarkItDownConverter()
        file_path = Path("/tmp/presentation.pptx")

        # Mock MarkItDown result
        mock_result = mocker.MagicMock()
        mock_result.text_content = "# Slide 1\n\nPresentation content."

        # Mock MarkItDown.convert method
        mocker.patch.object(converter.md, "convert", return_value=mock_result)

        # Act
        result = converter.convert(file_path)

        # Assert
        assert result["markdown"] == "# Slide 1\n\nPresentation content."
        assert result["metadata"]["format"] == "pptx"

    def test_converter_converts_html(self, mocker):
        """Test HTML conversion."""
        # Arrange
        converter = MarkItDownConverter()
        file_path = Path("/tmp/page.html")

        # Mock MarkItDown result
        mock_result = mocker.MagicMock()
        mock_result.text_content = "# HTML Page\n\nContent from HTML."

        # Mock MarkItDown.convert method
        mocker.patch.object(converter.md, "convert", return_value=mock_result)

        # Act
        result = converter.convert(file_path)

        # Assert
        assert result["markdown"] == "# HTML Page\n\nContent from HTML."
        assert result["metadata"]["format"] == "html"

    def test_converter_empty_content(self, mocker):
        """Test handling of empty content."""
        # Arrange
        converter = MarkItDownConverter()
        file_path = Path("/tmp/empty.txt")

        # Mock MarkItDown result with empty content
        mock_result = mocker.MagicMock()
        mock_result.text_content = ""

        # Mock MarkItDown.convert method
        mocker.patch.object(converter.md, "convert", return_value=mock_result)

        # Act
        result = converter.convert(file_path)

        # Assert
        assert result["markdown"] == ""
        assert result["metadata"]["char_count"] == 0
        assert result["metadata"]["word_count"] == 0

    def test_converter_whitespace_only_content(self, mocker):
        """Test handling of whitespace-only content."""
        # Arrange
        converter = MarkItDownConverter()
        file_path = Path("/tmp/whitespace.txt")

        # Mock MarkItDown result with whitespace
        mock_result = mocker.MagicMock()
        mock_result.text_content = "   \n\n   "

        # Mock MarkItDown.convert method
        mocker.patch.object(converter.md, "convert", return_value=mock_result)

        # Act
        result = converter.convert(file_path)

        # Assert
        assert result["markdown"] == "   \n\n   "
        assert result["metadata"]["char_count"] == 8
        assert result["metadata"]["word_count"] == 0
