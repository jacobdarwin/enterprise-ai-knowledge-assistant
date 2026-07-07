from pathlib import Path
from typing import List

from pypdf import PdfReader

from app.core.config.logging_config import get_logger
from app.rag.extraction.base import BaseExtractor, ExtractedPage

log = get_logger(__name__)


class PDFExtractor(BaseExtractor):
    @property
    def supported_extensions(self) -> List[str]:
        return [".pdf"]

    def extract(self, file_path: Path) -> List[ExtractedPage]:
        pages: List[ExtractedPage] = []
        try:
            reader = PdfReader(str(file_path))
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(ExtractedPage(page_number=i, text=text))
        except Exception as exc:
            log.error("pdf_extraction_failed", file=str(file_path), error=str(exc))
            raise ValueError(f"Failed to extract PDF '{file_path.name}': {exc}") from exc

        if not pages:
            log.warning("pdf_no_extractable_text", file=str(file_path))
        return pages
