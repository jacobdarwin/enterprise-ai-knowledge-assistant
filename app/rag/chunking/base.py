from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class TextChunk:
    text: str
    chunk_index: int
    start_char: int
    end_char: int


class ChunkingStrategy(ABC):
    def __init__(self, chunk_size: int, chunk_overlap: int = 0):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def split(self, text: str) -> List[TextChunk]: ...
