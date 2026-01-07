from abc import ABC, abstractmethod

class Compressor(ABC):
    @abstractmethod
    def compress(self, chunks, level):
        """
        Compress a list of chunks based on the given level.

        Args:
            chunks (List[dict]): Each dict has 'doc_id', 'chunk_id', 'text', optionally 'embedding'
            level (int or float): Compression level (e.g., 1 = minimal, higher = more aggressive)

        Returns:
            List[dict]: Compressed chunks
        """
        pass