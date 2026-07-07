import csv
from pathlib import Path
from typing import List

from app.core.config.logging_config import get_logger
from app.rag.extraction.base import BaseExtractor, ExtractedPage

log = get_logger(__name__)

ROWS_PER_PSEUDO_PAGE = 50


class CSVExtractor(BaseExtractor):
    @property
    def supported_extensions(self) -> List[str]:
        return [".csv"]

    def extract(self, file_path: Path) -> List[ExtractedPage]:
        try:
            with file_path.open(newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                rows = list(reader)
        except Exception as exc:
            log.error("csv_extraction_failed", file=str(file_path), error=str(exc))
            raise ValueError(f"Failed to read CSV '{file_path.name}': {exc}") from exc

        if not rows:
            return []

        header, data_rows = rows[0], rows[1:]
        pages: List[ExtractedPage] = []

        for i in range(0, len(data_rows), ROWS_PER_PSEUDO_PAGE):
            chunk_rows = data_rows[i : i + ROWS_PER_PSEUDO_PAGE]
            lines = [", ".join(header)]
            lines += [", ".join(row) for row in chunk_rows]
            page_number = (i // ROWS_PER_PSEUDO_PAGE) + 1
            pages.append(ExtractedPage(page_number=page_number, text="\n".join(lines)))

        return pages
