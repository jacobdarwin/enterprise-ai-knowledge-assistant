from pathlib import Path
from typing import List

from app.core.config.logging_config import get_logger
from app.rag.extraction.base import BaseExtractor, ExtractedPage

log = get_logger(__name__)


class TextExtractor(BaseExtractor):
    """Handles .txt and .md — both are just read as UTF-8 text, no page concept."""

    @property
    def supported_extensions(self) -> List[str]:
        return [".txt", ".md"]

    def extract(self, file_path: Path) -> List[ExtractedPage]:
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            log.error("text_extraction_failed", file=str(file_path), error=str(exc))
            raise ValueError(f"Failed to read '{file_path.name}': {exc}") from exc

        if not text.strip():
            log.warning("text_file_empty", file=str(file_path))
            return []
        return [ExtractedPage(page_number=1, text=text)]
