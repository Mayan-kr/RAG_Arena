"""Load raw source documents from the local dataset directory."""

from pathlib import Path

from llama_index.core import SimpleDirectoryReader


def load_source_documents(directory_path: str = "./dataset") -> list:
    """
    Read raw source documentation from the local environment.

    SimpleDirectoryReader walks the folder and wraps each file in a
    LlamaIndex Document (text + metadata). Graph and vector pipelines
    both consume the same list so comparisons stay fair.
    """
    path = Path(directory_path)
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {path.resolve()}")

    reader = SimpleDirectoryReader(input_dir=str(path))
    return reader.load_data()
