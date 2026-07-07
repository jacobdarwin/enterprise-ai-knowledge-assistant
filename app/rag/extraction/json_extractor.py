import json
from pathlib import Path
from typing import List

from app.core.config.logging_config import get_logger
from app.rag.extraction.base import BaseExtractor, ExtractedPage

log = get_logger(__name__)

ITEMS_PER_PSEUDO_PAGE = 20


class JSONExtractor(BaseExtractor):
    @property
    def supported_extensions(self) -> List[str]:
        return [".json"]

    def extract(self, file_path: Path) -> List[ExtractedPage]:
        try:
            raw = file_path.read_text(encoding="utf-8", errors="replace")
            data = json.loads(raw)
        except Exception as exc:
            log.error("json_extraction_failed", file=str(file_path), error=str(exc))
            raise ValueError(f"Failed to parse JSON '{file_path.name}': {exc}") from exc

        # If it's a list of records (the common "export" shape), page through
        # it in groups so one giant array doesn't become a single mega-chunk.
        if isinstance(data, list):
            pages: List[ExtractedPage] = []
            for i in range(0, len(data), ITEMS_PER_PSEUDO_PAGE):
                group = data[i : i + ITEMS_PER_PSEUDO_PAGE]
                page_number = (i // ITEMS_PER_PSEUDO_PAGE) + 1
                pages.append(
                    ExtractedPage(page_number=page_number, text=json.dumps(group, indent=2, ensure_ascii=False))
                )
            return pages

        # Otherwise treat the whole object as one page.
        return [ExtractedPage(page_number=1, text=json.dumps(data, indent=2, ensure_ascii=False))]
