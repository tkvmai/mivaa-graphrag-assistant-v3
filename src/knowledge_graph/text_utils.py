"""
Text processing utilities for the knowledge graph generator.
Handles document chunking using LangChain's RecursiveCharacterTextSplitter.
"""
import logging
from typing import List, Optional, Any
import time
# --- ADDED: LangChain Import ---
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    raise ImportError("LangChain text splitter not found. Please install: pip install langchain-text-splitters")

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    chunk_size: int = 500, # Target characters per chunk
    chunk_overlap: int = 50, # Characters to overlap
    separators: Optional[List[str]] = None,
    keep_separator: bool = True # LangChain's default behavior is similar
) -> List[str]:
    """
    Splits text recursively based on separators using LangChain's RecursiveCharacterTextSplitter.

    Args:
        text: The input text to chunk.
        chunk_size: The target maximum size of each chunk (in characters).
        chunk_overlap: The number of characters to overlap between consecutive chunks.
        separators: Optional list of separators to split by, in order of preference.
                    Defaults to LangChain's default separators if None.
        keep_separator: Whether to keep the separator characters attached to the chunks.
                        LangChain's splitter generally keeps separators by default.

    Returns:
        List of chunked text segments.
    """
    if not text or not isinstance(text, str):
        logger.warning("Received empty or non-string input for chunking.")
        return []

    # Define default separators similar to the previous implementation if none provided
    # LangChain's default is often ["\n\n", "\n", " ", ""] but let's be explicit
    if separators is None:
        separators = [
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences (note the space) - Adjust if needed
            "? ",
            "! ",
            ", ",    # Clauses
            " ",     # Words
            "",      # Characters (fallback)
        ]
        logger.debug(f"Using default separators: {separators}")
    else:
         logger.debug(f"Using provided separators: {separators}")


    try:
        # --- Use LangChain's Splitter ---
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len, # Use standard character length
            is_separator_regex=False, # Treat separators as simple strings
            separators=separators,
            keep_separator=keep_separator # Pass the keep_separator flag
        )

        chunks = text_splitter.split_text(text)

        # Filter out any potentially empty chunks just in case
        final_chunks = [chunk for chunk in chunks if chunk.strip()]

        logger.info(f"Chunked text into {len(final_chunks)} chunks using LangChain (target size: {chunk_size} chars, overlap: {chunk_overlap} chars).")
        return final_chunks

    except Exception as e:
        logger.error(f"Error during LangChain text splitting: {e}", exc_info=True)
        # Fallback: return the original text as a single chunk if splitting fails
        return [text]

def resolve_coreferences_spacy(text_content: str, nlp_pipeline: Optional[Any]) -> str:
    """
    Resolves coreferences in text using a spaCy pipeline.
    Relies on the pipeline having a coreference component that provides
    a way to get the resolved text (e.g., doc._.coref_resolved).
    """
    if not nlp_pipeline:
        logger.info("spaCy NLP pipeline not provided. Skipping coreference resolution.")
        return text_content
    if not text_content or not isinstance(text_content, str):
        logger.warning("Empty or invalid text for coreference resolution.")
        return text_content

    logger.info(f"Applying coreference resolution (spaCy) to text of length {len(text_content)}...")
    start_time = time.time()
    try:
        doc = nlp_pipeline(text_content)

        # The way to get resolved text is highly dependent on the specific
        # spaCy version and the coreference component/model used.
        # Check for common attributes set by coreference components:
        if hasattr(doc._, 'coref_resolved') and doc._.coref_resolved: # Common for some libraries
            resolved_text = doc._.coref_resolved
            logger.info(f"Coreference resolution applied using 'doc._.coref_resolved'. Duration: {time.time() - start_time:.2f}s")
            return resolved_text
        elif hasattr(doc._, 'resolved_text') and doc._.resolved_text: # Another possible attribute
            resolved_text = doc._.resolved_text
            logger.info(f"Coreference resolution applied using 'doc._.resolved_text'. Duration: {time.time() - start_time:.2f}s")
            return resolved_text
        # Add other checks here based on the component you end up using.
        # If no direct resolved text is available, you might need to iterate clusters
        # (e.g., doc._.coref_clusters), which is more complex for text replacement.
        # For now, if no direct resolved text attribute is found, we'll log and return original.
        else:
            logger.info("No direct resolved text attribute found (e.g., doc._.coref_resolved). Coreference might not have run or requires cluster iteration. Returning original text.")
            return text_content
    except Exception as e:
        logger.error(f"Error during spaCy coreference resolution: {e}", exc_info=True)
        return text_content # Return original text on error

# --- Example Usage (Optional - Remains the same) ---
if __name__ == "__main__":
    sample_text = """
This is the first paragraph. It provides some initial context. It contains multiple sentences.

This is the second paragraph. It continues the thought process and adding more detail. We want to see how chunking works. This paragraph is a bit longer to test splitting. It might even exceed the chunk size depending on the setting. Let's add another sentence here.

Third paragraph. Short one.
Fourth sentence follows immediately on the same line. Fifth sentence adds complexity with multiple clauses, testing the boundary conditions and ensuring that the overlap mechanism functions as expected, even when sentences vary greatly in length. Sentence number six. And finally, the seventh sentence concludes this example paragraph.
"""
    print("--- Original Text ---")
    print(sample_text)

    print("\n--- Chunking (Size 100, Overlap 20) ---")
    chunks_100_20 = chunk_text(sample_text, chunk_size=100, chunk_overlap=20)
    for i, chunk in enumerate(chunks_100_20):
        print(f"Chunk {i+1} (Length: {len(chunk)}):")
        print(repr(chunk)) # Use repr to see exact content including newlines/spaces
        print("-" * 10)

    print("\n--- Chunking (Size 200, Overlap 50) ---")
    chunks_200_50 = chunk_text(sample_text, chunk_size=200, chunk_overlap=50)
    for i, chunk in enumerate(chunks_200_50):
        print(f"Chunk {i+1} (Length: {len(chunk)}):")
        print(repr(chunk))
        print("-" * 10)

    print("\n--- Chunking (Size 500, Overlap 100) ---")
    chunks_500_100 = chunk_text(sample_text, chunk_size=500, chunk_overlap=100)
    for i, chunk in enumerate(chunks_500_100):
        print(f"Chunk {i+1} (Length: {len(chunk)}):")
        print(repr(chunk))
        print("-" * 10)

    print("\n--- Chunking (Short text) ---")
    short_text = "This is a single short sentence."
    chunks_short = chunk_text(short_text, chunk_size=50, chunk_overlap=10)
    print(f"Result for short text: {chunks_short}")

