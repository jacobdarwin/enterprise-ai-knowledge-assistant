"""
Text extraction interface.

Each extractor turns a raw uploaded file into a list of `ExtractedPage`
objects. Keeping "page" as a first-class concept (even for file types
that don't really have pages, like .txt/.csv/.json — they just get page=1)
means citation logic downstream ("cite filenames and page numbers") is
uniform regardless of source format.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class ExtractedPage:
    page_number: int
    text: str


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, file_path: Path) -> List[ExtractedPage]:
        """Extract text, one ExtractedPage per logical page/section."""

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]: ...
