from pathlib import Path
from typing import Dict, List, Type

from app.rag.extraction.base import BaseExtractor, ExtractedPage
from app.rag.extraction.csv_extractor import CSVExtractor
from app.rag.extraction.docx_extractor import DOCXExtractor
from app.rag.extraction.json_extractor import JSONExtractor
from app.rag.extraction.pdf_extractor import PDFExtractor
from app.rag.extraction.text_extractor import TextExtractor

_EXTRACTOR_CLASSES: List[Type[BaseExtractor]] = [
    PDFExtractor,
    DOCXExtractor,
    TextExtractor,
    CSVExtractor,
    JSONExtractor,
]


def _build_registry() -> Dict[str, BaseExtractor]:
    registry: Dict[str, BaseExtractor] = {}
    for cls in _EXTRACTOR_CLASSES:
        instance = cls()
        for ext in instance.supported_extensions:
            registry[ext] = instance
    return registry


_REGISTRY = _build_registry()


def get_extractor(file_path: Path) -> BaseExtractor:
    ext = file_path.suffix.lower()
    if ext not in _REGISTRY:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[ext]


def extract_text(file_path: Path) -> List[ExtractedPage]:
    """Convenience one-liner used by the ingestion service."""
    return get_extractor(file_path).extract(file_path)
