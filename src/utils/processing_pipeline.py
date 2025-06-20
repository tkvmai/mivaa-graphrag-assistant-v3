# processing_pipeline.py

import nest_asyncio
nest_asyncio.apply()

import streamlit as st
import logging
import threading
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import traceback # For logging full tracebacks from threads
import base64
import tempfile
import os
import io
import re
import json
import configparser
import sys
import time # Import time for delays and timing
import shutil # Import shutil for copying files
import hashlib # <<< IMPORTED FOR HASHING CHUNK IDs & FILES
from datetime import date # <<< IMPORT FOR DAILY LOGGING (Though logging function will be commented)
import requests # <<< IMPORT requests library
import asyncio # <<< IMPORTED for running async functions >>>
from sentence_transformers import SentenceTransformer # Example
from chromadb.api.models.Collection import Collection # Example
from src.knowledge_graph.text_utils import resolve_coreferences_spacy # Adjust path


# Import necessary modules from your project
# Adjust paths as necessary based on your project structure
# Assuming audit_db_manager is in the same directory or accessible
try:
    import src.utils.audit_db_manager # Use the module created in Step 1
except ImportError:
    st.error("Failed to import audit_db_manager. Ensure it's in the correct path.")
    # Provide dummy functions if import fails, to allow Streamlit to load
    class DummyAuditManager:
        def __getattr__(self, name): return lambda *args, **kwargs: None
    src.utils.audit_db_manager = DummyAuditManager()
    logging.error("Using DummyAuditManager due to import error.", exc_info=True)

# Import functions/classes needed for processing (adapt imports as needed)
try:
    from src.knowledge_graph.text_utils import chunk_text
    from src.knowledge_graph.llm import QuotaError
    from neo4j_exporter import Neo4jExporter # Assuming this exists
    from src.knowledge_graph.text_utils import chunk_text
    from src.knowledge_graph.llm import call_llm, extract_json_from_text, QuotaError
    from src.knowledge_graph.entity_standardization import standardize_entities, infer_relationships, limit_predicate_length
    from src.knowledge_graph.prompts import MAIN_SYSTEM_PROMPT, MAIN_USER_PROMPT  # Ensure MAIN_SYSTEM_PROMPT is accessible

except ImportError as e:
     st.error(f"Import Error in processing_pipeline.py: {e}. Ensure all source files are accessible.")
     logging.error(f"Import Error in processing_pipeline.py: {e}", exc_info=True)
     # Define placeholders if imports fail
     def chunk_text(*args, **kwargs): return ["chunk1", "chunk2"]
     class QuotaError(Exception): pass


# Logger setup
logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers(): # Basic config if needed
    # Load logging configuration from config.toml if available
    log_level = logging.INFO  # Default
    log_format = '%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s'  # Default
    
    try:
        # Use tomli if available (standardized as tomllib)
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                tomllib = None
        
        if tomllib:
            toml_config_path = Path("config.toml")
            if toml_config_path.is_file():
                with open(toml_config_path, "rb") as f:
                    config_toml = dict(tomllib.load(f))
                
                logging_config = config_toml.get("logging", {})
                level_str = logging_config.get("level", "INFO")
                log_format = logging_config.get("format", log_format)
                
                # Convert string level to logging constant
                level_map = {
                    "DEBUG": logging.DEBUG,
                    "INFO": logging.INFO,
                    "WARNING": logging.WARNING,
                    "ERROR": logging.ERROR,
                    "CRITICAL": logging.CRITICAL
                }
                log_level = level_map.get(level_str.upper(), logging.INFO)
    except Exception:
        pass  # Use defaults if config loading fails
    
    logging.basicConfig(level=log_level, format=log_format)

def process_uploaded_file_ocr(uploaded_file: Any, mistral_client: Optional[Any]) -> Optional[str]:
    """
    Handles OCR processing for PDF or Image using Mistral via its API.
    Logs errors/warnings instead of using st.error/st.warning.
    Returns extracted text content as a string, or None on failure.
    REMOVED PDF display path logic for background thread safety.

    Args:
        uploaded_file: The uploaded file object (needs methods like .name, .type, .getvalue()).
                       Using Any type hint for flexibility if not always a Streamlit object.
        mistral_client: An initialized Mistral client instance with OCR capabilities.

    Returns:
        Optional[str]: The extracted text content, or None if processing fails.
    """
    if not mistral_client:
        logger.error("Mistral client not provided or initialized. Cannot perform OCR.")
        return None # Cannot proceed without the client

    ocr_text_content: Optional[str] = None
    # Safely get attributes, providing defaults
    file_name = getattr(uploaded_file, 'name', 'unknown_file')
    file_type = getattr(uploaded_file, 'type', 'unknown_type')

    logger.info(f"Attempting OCR for file: '{file_name}' (Type: {file_type})")

    # Check for supported types before creating temp file
    if file_type not in ["application/pdf", "image/png", "image/jpeg", "image/jpg"]:
         logger.error(f"Unsupported file type for Mistral OCR: {file_type}")
         return None

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / file_name
        try:
            # Get file content bytes
            try:
                 file_content_bytes = uploaded_file.getvalue()
            except AttributeError:
                 logger.error(f"Failed to get content bytes from uploaded_file object for '{file_name}'. Does it have a getvalue() method?")
                 return None # Cannot proceed without content

            # Write to temporary file
            with open(temp_path, "wb") as tmp:
                tmp.write(file_content_bytes)

            file_size = temp_path.stat().st_size
            logger.info(f"Uploading '{file_name}' ({file_size} bytes) to Mistral for OCR...")

            # Upload to Mistral Files API
            with open(temp_path, "rb") as file_obj:
                # Ensure the 'file' argument structure matches the client library's expectation
                file_upload = mistral_client.files.upload(
                    file={"file_name": file_name, "content": file_obj},
                    purpose="ocr" # Specify purpose if required by API/client
                )

            # Get Signed URL for processing (ensure methods exist and work)
            signed_url = mistral_client.files.get_signed_url(file_id=file_upload.id).url
            logger.info(f"File uploaded (Mistral ID: {file_upload.id}), performing OCR using signed URL...")

            # Prepare input for OCR processing API call
            ocr_input = {}
            if file_type == "application/pdf":
                ocr_input = {"document_url": signed_url}
                # --- PDF display path logic REMOVED ---
            elif file_type in ["image/png", "image/jpeg", "image/jpg"]:
                ocr_input = {"image_url": signed_url}
            # No else needed due to earlier check

            # Call the OCR process method
            ocr_response = mistral_client.ocr.process(
                document=ocr_input,
                model="mistral-ocr-latest" # Confirm model name is correct
            )

            # Process the response
            if ocr_response and hasattr(ocr_response, 'pages') and ocr_response.pages:
                all_pages_text = []
                for page in ocr_response.pages:
                     # Check if page object has 'markdown' attribute
                     page_text = getattr(page, 'markdown', None)
                     if page_text:
                         all_pages_text.append(page_text)

                ocr_text_content = "\n\n".join(all_pages_text).strip()

                if ocr_text_content:
                    logger.info(f"OCR successful for '{file_name}'. Extracted ~{len(ocr_text_content)} characters.")
                else:
                    # Log if OCR ran but extracted no text
                    logger.warning(f"OCR completed for '{file_name}', but no text content was extracted from pages.")
            else:
                # Log if the response structure was unexpected
                logger.warning(f"OCR processing for '{file_name}' did not return the expected 'pages' structure or content.")
                ocr_text_content = None # Ensure it's None if response is bad

        except AttributeError as ae:
             # Specific error if mistral_client methods are wrong
             logger.error(f"AttributeError during Mistral API call for '{file_name}'. Check client methods (files.upload, files.get_signed_url, ocr.process). Error: {ae}", exc_info=True)
             ocr_text_content = None
        except Exception as e:
            # Catch-all for other errors during the process
            logger.error(f"An unexpected error occurred during OCR processing for '{file_name}': {e}", exc_info=True)
            ocr_text_content = None

    # Return the extracted text (or None if any step failed)
    return ocr_text_content

def extract_knowledge_graph(
    text_content: str,
    config: Dict[str, Any],
    requests_session: Optional[Any] # Pass requests.Session if used by call_llm
) -> Tuple[List[Dict], List[str], int]: # Return triples, chunks, and extracted_count
    """
    Extracts, standardizes, and infers knowledge graph triples from text.
    Logs progress and errors using the logging module.
    Returns: Tuple containing (list of final triples, list of text chunks, count of initially extracted triples)
    Raises: RuntimeError or ValueError on critical failures (e.g., chunking, LLM config).
    """
    initial_triples = []
    text_chunks = []
    total_extracted = 0 # Initialize count

    if not text_content:
        logger.warning("extract_knowledge_graph called with empty text_content.")
        return initial_triples, text_chunks, total_extracted

    # --- Configuration ---
    chunk_size_chars = config.get('CHUNK_SIZE', 1000)
    overlap_chars = config.get('CHUNK_OVERLAP', 100)
    extraction_model = config.get('TRIPLE_EXTRACTION_LLM_MODEL')
    extraction_api_key = config.get('TRIPLE_EXTRACTION_API_KEY')
    extraction_base_url = config.get('TRIPLE_EXTRACTION_BASE_URL')
    extraction_max_tokens = config.get('TRIPLE_EXTRACTION_MAX_TOKENS', 1500)
    extraction_temperature = config.get('TRIPLE_EXTRACTION_TEMPERATURE', 0.2)

    if not all([extraction_model, extraction_api_key]):
        logger.error("KG Extraction Error! Missing LLM Config for triple extraction.")
        raise ValueError("Missing configuration for Triple Extraction LLM.")

    # --- Chunking ---
    try:
        text_chunks = chunk_text(text_content, chunk_size=chunk_size_chars, chunk_overlap=overlap_chars)
        num_chunks = len(text_chunks)
        logger.info(f"Split text into {num_chunks} chunks (size={chunk_size_chars} chars, overlap={overlap_chars} chars).")
        if num_chunks == 0:
            logger.warning("Text chunking resulted in zero chunks.")
            return [], [], 0 # Return empty if no chunks
    except Exception as e:
        logger.error(f"Error during text chunking: {e}", exc_info=True)
        raise RuntimeError(f"Critical error during text chunking: {e}") from e

    # --- KG Extraction Loop ---
    logger.info(f"Preparing to extract triples from {num_chunks} chunk(s)...")
    system_prompt = MAIN_SYSTEM_PROMPT
    max_retries = 2
    default_retry_delay = 5
    quota_retry_delay = 70
    overall_start_time = time.time()

    for i, chunk in enumerate(text_chunks):
        chunk_start_time = time.time()
        logger.info(f"Processing chunk {i + 1}/{num_chunks} for KG extraction...")

        user_prompt = MAIN_USER_PROMPT + f"\n```text\n{chunk}\n```\n"
        attempt = 0
        success = False
        response_text = None
        valid_chunk_triples = [] # Store valid triples for this chunk

        while attempt < max_retries and not success:
            try:
                logger.debug(f"Chunk {i+1}, Attempt {attempt+1}: Calling LLM...")
                llm_call_start_time = time.time()
                response_text = call_llm(
                    model=extraction_model,
                    user_prompt=user_prompt,
                    api_key=extraction_api_key,
                    system_prompt=system_prompt,
                    max_tokens=extraction_max_tokens,
                    temperature=extraction_temperature,
                    base_url=extraction_base_url,
                    session=requests_session # Pass session if needed by call_llm
                )
                llm_call_duration = time.time() - llm_call_start_time
                logger.debug(f"Chunk {i + 1}, Attempt {attempt+1}: LLM call duration: {llm_call_duration:.2f}s.")

                logger.debug(f"Chunk {i+1}, Attempt {attempt+1}: Extracting JSON...")
                json_extract_start_time = time.time()
                chunk_results = extract_json_from_text(response_text)
                json_extract_duration = time.time() - json_extract_start_time
                logger.debug(f"Chunk {i+1}, Attempt {attempt+1}: JSON extraction duration: {json_extract_duration:.2f}s.")

                if chunk_results and isinstance(chunk_results, list):
                    required_keys = {"subject", "subject_type", "predicate", "object", "object_type"}
                    for item_idx, item in enumerate(chunk_results):
                        if isinstance(item, dict):
                            missing_keys = required_keys - item.keys()
                            invalid_values = {k: item[k] for k in required_keys.intersection(item.keys()) if not isinstance(item[k], str) or not item[k].strip()}
                            if not missing_keys and not invalid_values:
                                chunk_hash = hashlib.sha256(chunk.encode('utf-8')).hexdigest() # Re-calculate hash? Or maybe not needed here if chunk ID added later? Assume added later.
                                item["chunk_text"] = chunk.strip() # Optionally add chunk text reference
                                item["predicate"] = limit_predicate_length(item["predicate"]) # Apply predicate length limit
                                valid_chunk_triples.append(item)
                            else:
                                reason = []
                                if missing_keys: reason.append(f"missing keys: {missing_keys}")
                                if invalid_values: reason.append(f"invalid/empty values: {invalid_values}")
                                logger.warning(f"Invalid triple structure in chunk {i + 1}, item {item_idx+1} ({'; '.join(reason)}): {item}")
                        else:
                            logger.warning(f"Invalid item type (expected dict) in chunk {i + 1}, item {item_idx+1}: {item}")

                    # Update counts and mark success FOR THIS CHUNK'S ATTEMPT
                    chunk_extracted_count = len(valid_chunk_triples)
                    total_extracted += chunk_extracted_count # Add to overall count
                    initial_triples.extend(valid_chunk_triples) # Add triples to overall list
                    success = True # Mark the attempt successful
                    logger.info(f"Chunk {i + 1}/{num_chunks}: Attempt {attempt+1} successful. Extracted {chunk_extracted_count} valid triples.")

                elif chunk_results is None:
                    logger.warning(f"LLM response for chunk {i+1}, attempt {attempt+1} did not contain valid JSON.")
                    success = True # Treat as success (no data found), don't retry
                else:
                    logger.warning(f"No valid list of triples extracted from chunk {i + 1}, attempt {attempt+1}. Response: {response_text[:200]}...")
                    success = True # Treat as success (no data found), don't retry

            except QuotaError as qe:
                attempt += 1
                logger.error(f"Attempt {attempt}/{max_retries}: Quota Error processing chunk {i + 1}: {qe}")
                if attempt >= max_retries:
                    logger.error(f"Max retries reached for chunk {i + 1} due to Quota Error. Skipping chunk.")
                    # Optionally raise an exception or just log and continue to next chunk
                    break # Exit retry loop for this chunk
                else:
                    logger.info(f"Waiting {quota_retry_delay}s before retry...")
                    time.sleep(quota_retry_delay)
            except TimeoutError as te:
                attempt += 1
                logger.error(f"Attempt {attempt}/{max_retries}: Timeout Error processing chunk {i + 1}: {te}")
                if attempt >= max_retries:
                    logger.error(f"Max retries reached for chunk {i + 1} due to Timeout Error. Skipping chunk.")
                    break
                else:
                    logger.info(f"Waiting {default_retry_delay}s before retry...")
                    time.sleep(default_retry_delay)
            except Exception as e:
                attempt += 1
                logger.error(f"Attempt {attempt}/{max_retries}: Unexpected error processing chunk {i + 1}: {e}", exc_info=True)
                if attempt >= max_retries:
                    logger.error(f"Max retries reached for chunk {i + 1} due to Error. Skipping chunk.")
                    # Optionally raise a more specific exception if one chunk failure should stop the whole process
                    break
                else:
                    logger.info(f"Waiting {default_retry_delay}s before retry...")
                    time.sleep(default_retry_delay)
        # --- End of retry loop ---

        chunk_duration = time.time() - chunk_start_time
        logger.info(f"Chunk {i + 1}/{num_chunks}: Finished processing. Duration: {chunk_duration:.2f}s.")
    # --- End of chunk loop ---

    overall_duration = time.time() - overall_start_time
    logger.info(f"Finished initial triple extraction phase. Total extracted: {total_extracted} triples. Total time: {overall_duration:.2f}s.")

    # --- Standardization ---
    processed_triples = initial_triples
    if config.get('STANDARDIZATION_ENABLED', False) and processed_triples:
        logger.info("Applying entity standardization...")
        try:
            standardized_result = standardize_entities(processed_triples, config)
            processed_triples = standardized_result if standardized_result is not None else processed_triples
            logger.info(f"Entity standardization complete. Triple count: {len(processed_triples)}")
        except Exception as e:
            logger.error(f"Error during standardization: {e}", exc_info=True)
            # Decide whether to continue with unstandardized triples or raise error
    else:
        logger.info("Skipping standardization.")

    # --- Inference ---
    if config.get('INFERENCE_ENABLED', False) and processed_triples:
        logger.info("Applying relationship inference...")
        try:
            inferred_result = infer_relationships(processed_triples, config)
            processed_triples = inferred_result if inferred_result is not None else processed_triples
            logger.info(f"Relationship inference complete. Triple count: {len(processed_triples)}")
        except Exception as e:
            logger.error(f"Error during inference: {e}", exc_info=True)
            # Decide whether to continue or raise error
    else:
        logger.info("Skipping inference.")

    # --- Final Return ---
    final_triples = processed_triples
    final_triple_count = len(final_triples)
    logger.info(f"extract_knowledge_graph finished. Final triple count: {final_triple_count}, Initial extracted count: {total_extracted}")
    if final_triples:
        logger.debug(f"Final triple example: {final_triples[0]}")

    # Return final triples, original chunks, and the count of *initially* extracted triples
    return final_triples, text_chunks, total_extracted

EmbeddingModelType = Any
ChromaCollectionType = Any

# --- MODIFIED Function ---
def store_chunks_and_embeddings(
    text_chunks: List[str],
    embedding_model: Optional[EmbeddingModelType], # Accept the model object
    chroma_collection: Optional[ChromaCollectionType], # Accept the collection object
    current_doc_name: str,
    config: Optional[Dict[str, Any]] = None # Config might not be needed now
) -> Tuple[bool, int]: # Return (success, count_stored)
    """
    Generates embeddings for text chunks and stores/updates them in ChromaDB.
    Accepts initialized embedding model and chroma collection objects.
    Logs progress and errors. Returns success status and count stored.

    Args:
        text_chunks: List of text strings to embed and store.
        embedding_model: Initialized SentenceTransformer (or compatible) model object.
        chroma_collection: Initialized ChromaDB collection object.
        current_doc_name: Name of the source document for metadata.
        config: Configuration dictionary (currently unused, kept for potential future use).

    Returns:
        Tuple[bool, int]: (True, num_stored) on success, (False, 0) on failure.
    """
    num_chunks = len(text_chunks)
    if not text_chunks:
        logger.warning(f"No chunks provided for embedding for document '{current_doc_name}'. Skipping storage.")
        return True, 0 # No error, just nothing to store

    # --- Input Validation ---
    if not embedding_model:
        logger.error(f"Embedding model resource was not provided for '{current_doc_name}'. Cannot generate/store embeddings.")
        return False, 0
    if not chroma_collection:
        logger.error(f"ChromaDB collection resource was not provided for '{current_doc_name}'. Cannot store embeddings.")
        return False, 0
    # Check if required methods exist (optional but safer)
    if not hasattr(embedding_model, 'encode'):
         logger.error(f"Provided embedding_model object lacks an 'encode' method for '{current_doc_name}'.")
         return False, 0
    if not hasattr(chroma_collection, 'upsert'):
         logger.error(f"Provided chroma_collection object lacks an 'upsert' method for '{current_doc_name}'.")
         return False, 0

    logger.info(f"Preparing to generate and store embeddings for {num_chunks} chunks from '{current_doc_name}'...")

    # --- Generate IDs ---
    # Consider adding doc name/hash prefix to chunk hash for potentially better uniqueness across docs
    try:
        chunk_ids = [f"{current_doc_name}_{hashlib.sha256(chunk.encode('utf-8')).hexdigest()[:16]}" for chunk in text_chunks]
        logger.debug(f"Generated {len(chunk_ids)} chunk IDs for embedding (Example: {chunk_ids[0] if chunk_ids else 'N/A'}).")
    except Exception as e:
         logger.error(f"Failed to generate chunk IDs for '{current_doc_name}': {e}", exc_info=True)
         return False, 0

    # --- Generate Embeddings ---
    embeddings_list = []
    try:
        logger.info(f"Generating embeddings for {num_chunks} chunks...")
        start_time = time.time()
        # Note: No st.spinner here
        embeddings = embedding_model.encode(text_chunks, show_progress_bar=False) # show_progress_bar=False useful for logs
        embeddings_list = embeddings.tolist() if hasattr(embeddings, 'tolist') else list(embeddings)
        duration = time.time() - start_time
        logger.info(f"Embeddings generated successfully for {num_chunks} chunks ({duration:.2f}s).")
    except Exception as e:
        logger.error(f"Failed to generate embeddings for '{current_doc_name}': {e}", exc_info=True)
        # Removed st.error
        return False, 0

    # --- Store in ChromaDB ---
    num_stored = 0
    try:
        # Use getattr for safe access to collection name if needed for logging
        collection_name = getattr(chroma_collection, 'name', 'Unknown')
        logger.info(f"Storing {len(chunk_ids)} embeddings/documents in ChromaDB collection '{collection_name}'...")
        start_time = time.time()
        # Create metadata for each chunk
        metadatas = [{"source_document": current_doc_name, "original_chunk_index": i} for i in range(num_chunks)]

        # Perform the upsert operation
        chroma_collection.upsert(
            ids=chunk_ids,
            embeddings=embeddings_list,
            documents=text_chunks, # Store the original text chunk as the document
            metadatas=metadatas
        )
        num_stored = len(chunk_ids) # If upsert doesn't raise error, assume all were processed
        duration = time.time() - start_time

        # Log success and potentially the new collection count
        try:
            current_count = chroma_collection.count()
            logger.info(f"Successfully upserted {num_stored} embeddings for '{current_doc_name}' ({duration:.2f}s). Collection count now: {current_count}")
        except Exception as count_e:
            # Log success even if count fails (count might be slow/error prone sometimes)
            logger.info(f"Successfully upserted {num_stored} embeddings for '{current_doc_name}' ({duration:.2f}s). (Could not retrieve updated collection count: {count_e})")

        return True, num_stored # Return success and the number stored/attempted

    except Exception as e:
        logger.error(f"Failed to store embeddings in ChromaDB for '{current_doc_name}': {e}", exc_info=True)
        # Removed st.error
        return False, 0 # Return failure status and 0 count

# --- Define Constants needed by cache functions ---
# You might want to manage this path via config later, but define it here for now
CACHE_DIR = Path("./graphrag_cache")
TRIPLE_CACHE_DIR = CACHE_DIR / "triples"
# Ensure the base cache directory exists when the module loads or in an init function
# CACHE_DIR.mkdir(parents=True, exist_ok=True) # Optional: Create base dir on import

# --- Finalized Helper Functions ---

def get_file_hash(file_content: bytes) -> str:
    """Calculates the SHA256 hash of file content."""
    if not isinstance(file_content, bytes):
        logger.error("get_file_hash requires bytes input.")
        # Decide on error handling: raise TypeError or return a default error hash
        raise TypeError("Input must be bytes to calculate hash.")
        # return "error_invalid_input_type_not_bytes" # Alternative: return error hash
    return hashlib.sha256(file_content).hexdigest()

def load_triples_from_cache(file_hash: str) -> Optional[Tuple[List[Dict], List[str]]]:
    """
    Loads triples and chunks from a cache file based on file hash.

    Args:
        file_hash: The SHA256 hash of the file content.

    Returns:
        A tuple containing (list of triples, list of chunks) if cache exists and is valid,
        otherwise None.
    """
    if not file_hash:
        logger.warning("load_triples_from_cache called with empty file_hash.")
        return None

    cache_file = TRIPLE_CACHE_DIR / f"{file_hash}.json"
    logger.debug(f"Checking cache file: {cache_file}")

    if cache_file.is_file():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Validate cache structure
            if isinstance(data, dict) and "triples" in data and "chunks" in data and isinstance(data["triples"], list) and isinstance(data["chunks"], list):
                logger.info(f"Cache hit: Loaded {len(data['triples'])} triples and {len(data['chunks'])} chunks from {cache_file}")
                return data["triples"], data["chunks"]
            else:
                logger.warning(f"Invalid cache file format: {cache_file}. Required keys 'triples' (list) and 'chunks' (list) not found or invalid type.")
                return None
        except json.JSONDecodeError as jde:
             logger.warning(f"Failed to decode JSON from cache file {cache_file}: {jde}")
             return None
        except Exception as e:
            # Catch other potential errors like file read issues
            logger.warning(f"Failed to load cache file {cache_file}: {e}.", exc_info=True)
            return None
    else:
        logger.debug(f"Cache miss for hash {file_hash} (File not found: {cache_file})")
        return None

def save_triples_to_cache(file_hash: str, triples: List[Dict], chunks: List[str]) -> bool:
    """
    Saves extracted triples and corresponding text chunks to a cache file.

    Args:
        file_hash: The SHA256 hash of the original file content.
        triples: The list of extracted knowledge graph triples (dictionaries).
        chunks: The list of text chunks corresponding to the triples.

    Returns:
        True if saving was successful, False otherwise.
    """
    if not file_hash:
        logger.error("Cannot save cache: file_hash is empty.")
        return False
    # Basic validation of inputs
    if not isinstance(triples, list) or not isinstance(chunks, list):
         logger.error(f"Cannot save cache for hash {file_hash}: triples or chunks are not lists.")
         return False

    try:
        # Ensure the cache directory exists
        TRIPLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = TRIPLE_CACHE_DIR / f"{file_hash}.json"
        data_to_save = {"triples": triples, "chunks": chunks}

        logger.info(f"Saving {len(triples)} triples and {len(chunks)} chunks to cache: {cache_file}")
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2) # Use indent for readability

        logger.info(f"Successfully saved cache file: {cache_file}")
        return True # Indicate success
    except TypeError as te:
         # Catch errors if data isn't JSON serializable
         logger.error(f"Failed to save cache for hash {file_hash} due to non-serializable data: {te}", exc_info=True)
         return False # Indicate failure
    except OSError as oe:
        # Catch file system errors (permissions, disk full, etc.)
        logger.error(f"Failed to save cache file {cache_file} due to OS error: {oe}", exc_info=True)
        return False # Indicate failure
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"An unexpected error occurred saving cache for hash {file_hash}: {e}", exc_info=True)
        return False # Indicate failure

# --- Main Pipeline Function (to be run in a thread) ---

def run_ingestion_pipeline_thread(
    job_id: str,
    uploaded_files: List[Any], # <<< FIX: Changed type hint to List[Any]
    config: Dict[str, Any],
    use_cache: bool,
    # Pass necessary resource objects
    mistral_client: Optional[Any],
    neo4j_exporter: Optional[Neo4jExporter], # Use correct type hint if Neo4jExporter is imported
    embedding_model_resource: Optional[Any],
    chroma_collection_resource: Optional[Any],
    requests_session: Optional[requests.Session],
    nlp_pipeline_resource: Optional[Any]
):
    """
    Orchestrates the ingestion pipeline for a batch of files.
    Designed to be run in a separate thread. Logs progress to the audit DB.
    """
    logger.info(f"[Job {job_id}] Starting ingestion pipeline thread for {len(uploaded_files)} files.")
    files_processed_successfully = 0
    files_failed = 0
    files_cached = 0

    # Iterate through each uploaded file
    for i, uploaded_file in enumerate(uploaded_files):
        # --- Initialize details for THIS file ---
        file_name = uploaded_file.name
        file_processing_id = None
        file_hash = None
        text_content = None
        text_chunks = []
        extracted_triples = []

        text_extracted_success = False
        num_chunks = 0
        num_triples_extracted = 0
        num_triples_loaded = 0
        num_vectors_stored = 0
        cache_hit = False
        final_status = 'Failed - Unknown'  # Default status
        error_msg_details = None
        log_msg_details = None
        # --- End Initialization ---

        try:
            # --- Get File Details & Start Audit Record ---
            logger.info(f"[Job {job_id}] Processing file {i + 1}/{len(uploaded_files)}: '{file_name}'")
            file_content_bytes = uploaded_file.getvalue()
            file_hash = get_file_hash(file_content_bytes)  # Assume defined
            file_size = len(file_content_bytes)
            file_type = uploaded_file.type

            file_processing_id = src.utils.audit_db_manager.start_file_processing(
                job_id=job_id, file_name=file_name, file_size=file_size,
                file_type=file_type, file_hash=file_hash
            )
            if not file_processing_id:
                logger.error(f"[Job {job_id}] Failed to create audit record for file '{file_name}'. Skipping.")
                files_failed += 1
                continue  # Skip to next file

            # --- Cache Check ---
            cached_data = load_triples_from_cache(file_hash) if use_cache else None  # Assume defined
            cache_hit = bool(cached_data)

            if cache_hit:
                logger.info(f"[Job {job_id} | File {file_processing_id}] Cache hit for '{file_name}'.")
                extracted_triples, text_chunks = cached_data
                # Get counts from cache
                num_chunks = len(text_chunks) if text_chunks else 0
                num_triples_extracted = len(extracted_triples) if extracted_triples else 0
                # Assume stored/loaded counts match extracted for cache
                num_triples_loaded = num_triples_extracted
                num_vectors_stored = num_chunks
                text_extracted_success = True  # Assume true if cached
                final_status = 'Cached'
                files_cached += 1
                # --- Skip to final update outside the try block ---

            else:  # Not a cache hit, perform full processing
                cache_hit = False
                # --- Step 1: Extract Text ---
                logger.debug(f"[Job {job_id} | File {file_processing_id}] Extracting text...")
                if file_type == "text/plain":
                    text_content = file_content_bytes.decode('utf-8', errors='replace')
                elif file_type in ["application/pdf", "image/png", "image/jpeg", "image/jpg"]:
                    if mistral_client:
                        text_content = process_uploaded_file_ocr(uploaded_file, mistral_client)  # Assume defined
                    else:
                        raise ValueError("Mistral client not available for OCR.")
                else:
                    raise ValueError(f"Unsupported file type: {file_type}")

                if text_content:
                    text_extracted_success = True
                else:
                    raise ValueError("Text extraction failed or yielded no content.")
                logger.debug(f"[Job {job_id} | File {file_processing_id}] Text extracted successfully.")

                # --- NEW STEP: Coreference Resolution ---
                if config.get('COREFERENCE_RESOLUTION_ENABLED', False) and nlp_pipeline_resource:
                    logger.info(f"[Job {job_id} | File {file_processing_id}] Applying coreference resolution...")
                    resolved_text_content = resolve_coreferences_spacy(text_content, nlp_pipeline_resource)
                    if len(resolved_text_content) != len(text_content):  # Basic check for change
                        logger.info(
                            f"[Job {job_id} | File {file_processing_id}] Coreference resolution modified the text.")
                        text_content = resolved_text_content  # Use the resolved text
                    else:
                        logger.info(
                            f"[Job {job_id} | File {file_processing_id}] Coreference resolution made no changes or was skipped by the resolver.")
                else:
                    logger.info(f"[Job {job_id} | File {file_processing_id}] Skipping coreference resolution.")

                # --- Step 2: Chunk Text ---
                logger.debug(f"[Job {job_id} | File {file_processing_id}] Chunking text...")
                text_chunks = chunk_text(text_content, chunk_size=config.get('CHUNK_SIZE', 1000),
                                         chunk_overlap=config.get('CHUNK_OVERLAP', 100))  # Assume defined
                num_chunks = len(text_chunks)
                if num_chunks == 0: raise ValueError("Text chunking resulted in zero chunks.")
                logger.debug(f"[Job {job_id} | File {file_processing_id}] Chunking complete ({num_chunks} chunks).")

                # --- Step 3: Knowledge Graph Extraction ---
                logger.debug(f"[Job {job_id} | File {file_processing_id}] Extracting Knowledge Graph...")
                # Assume extract_knowledge_graph returns (triples, chunks_from_kg, initial_extracted_count)
                extracted_triples, _, num_triples_extracted = extract_knowledge_graph(
                    text_content=text_content, config=config, requests_session=requests_session
                )
                logger.debug(
                    f"[Job {job_id} | File {file_processing_id}] KG Extraction complete. Initial extracted: {num_triples_extracted}.")

                # --- Step 4: Store Triples in Neo4j ---
                if extracted_triples and neo4j_exporter:
                    num_triples_to_store = len(extracted_triples)
                    logger.debug(
                        f"[Job {job_id} | File {file_processing_id}] Storing {num_triples_to_store} triples in Neo4j...")
                    # Assumes store_triples returns (bool, int)
                    neo4j_success, num_triples_loaded = neo4j_exporter.store_triples(extracted_triples)
                    if not neo4j_success: raise RuntimeError(
                        f"Neo4j storage failed, processed {num_triples_loaded} before error.")
                    logger.debug(
                        f"[Job {job_id} | File {file_processing_id}] Neo4j storage complete ({num_triples_loaded} triples).")
                else:
                    num_triples_loaded = 0
                    logger.info(
                        f"[Job {job_id} | File {file_processing_id}] Skipping Neo4j storage (No triples or exporter unavailable).")

                # --- Step 5: Store Chunks & Embeddings ---
                if text_chunks and embedding_model_resource and chroma_collection_resource:
                    logger.debug(f"[Job {job_id} | File {file_processing_id}] Storing {num_chunks} embeddings...")
                    # Assume store_chunks_and_embeddings returns (bool, int)
                    embedding_success, num_vectors_stored = store_chunks_and_embeddings(
                        text_chunks=text_chunks, embedding_model=embedding_model_resource,
                        chroma_collection=chroma_collection_resource, current_doc_name=file_name, config=config
                    )
                    if not embedding_success: raise RuntimeError(
                        f"Embedding storage failed after processing {num_vectors_stored} vectors.")
                    logger.debug(
                        f"[Job {job_id} | File {file_processing_id}] ChromaDB storage complete ({num_vectors_stored} vectors).")
                else:
                    num_vectors_stored = 0
                    logger.info(
                        f"[Job {job_id} | File {file_processing_id}] Skipping embedding storage (No chunks or resources unavailable).")

                # If we reach here without error and it wasn't a cache hit
                final_status = 'Success'
                files_processed_successfully += 1
                logger.info(f"[Job {job_id} | File {file_processing_id}] Successfully processed file '{file_name}'.")

        # --- Main Exception Handler for the file ---
        except Exception as e:
            logger.error(f"[Job {job_id}] FAILED processing file '{file_name}' (ID: {file_processing_id}). Error: {e}",
                         exc_info=True)
            files_failed += 1
            tb_str = traceback.format_exc()
            # Determine specific failure status more robustly
            if isinstance(e, ValueError) and "OCR" in str(e):
                final_status = 'Failed - OCR'
            elif isinstance(e, ValueError) and "chunking" in str(e):
                final_status = 'Failed - Chunking'
            elif isinstance(e, QuotaError) or "KG Extraction" in tb_str or "extract_knowledge_graph" in tb_str:
                final_status = 'Failed - KG Extract'
            elif "Neo4j" in str(e) or "neo4j_exporter" in tb_str:
                final_status = 'Failed - Neo4j'
            elif "Embedding" in str(e) or "ChromaDB" in str(e) or "store_chunks_and_embeddings" in tb_str:
                final_status = 'Failed - Embedding'
            else:
                final_status = 'Failed - Unknown'
            error_msg_details = f"{type(e).__name__}: {str(e)[:500]}"
            log_msg_details = tb_str[:1500]
            # Keep any counts determined before the error occurred
            # (num_chunks, num_triples_extracted, etc. retain their values)

        # --- Single Audit Update Call for the file (runs after try or except) ---
        if file_processing_id:
            logger.debug(
                f"[Job {job_id} | File {file_processing_id}] Performing final status update: Status='{final_status}', Chunks={num_chunks}, Extracted={num_triples_extracted}, Loaded={num_triples_loaded}, Stored={num_vectors_stored}")
            src.utils.audit_db_manager.update_file_status(
                file_processing_id=file_processing_id,
                status=final_status,
                cache_hit=cache_hit,
                text_extracted=text_extracted_success,
                num_chunks=num_chunks,  # Use final value
                num_triples_extracted=num_triples_extracted,  # Use final value
                num_triples_loaded=num_triples_loaded,  # Use final value
                num_vectors_stored=num_vectors_stored,  # Use final value
                error_message=error_msg_details,
                log_messages=log_msg_details
                # end_timestamp is handled automatically by update_file_status if not provided
            )
        else:
            logger.error(
                f"[Job {job_id}] Could not update audit status for '{file_name}' because file_processing_id was not generated.")

        # --- End of file processing loop ---

        # --- Update Overall Job Status (logic remains the same, based on counts) ---
    final_job_status = 'Failed'
    if files_failed == 0:
        if files_processed_successfully > 0 or files_cached > 0:  # If anything completed (new or cached) with no errors
            final_job_status = 'Completed'
        else:  # No files processed, no failures, no cache? Still treat as completed?
            final_job_status = 'Completed'  # Or perhaps 'Empty'?
    elif files_processed_successfully > 0 or files_cached > 0:  # Failures occurred, but some success/cache
        final_job_status = 'Completed with Errors'
    # If only failures occurred, status remains 'Failed'

    logger.info(
        f"[Job {job_id}] Pipeline thread finished. Success: {files_processed_successfully}, Failed: {files_failed}, Cached: {files_cached}. Final Status: {final_job_status}")
    src.utils.audit_db_manager.update_job_status(job_id=job_id, status=final_job_status)

_pipeline_threads: Dict[str, threading.Thread] = {} # Keep track of running threads

def start_ingestion_job_async(
    uploaded_files: List[Any],
    config: Dict[str, Any],
    use_cache: bool,
    mistral_client: Optional[Any],
    neo4j_exporter: Optional[Neo4jExporter],
    embedding_model_resource: Optional[Any],
    chroma_collection_resource: Optional[Any],
    requests_session_resource: Optional[Any],
    nlp_pipeline_resource: Optional[Any]
    # Add other resources as needed
) -> Optional[str]:
    """
    Creates the audit job record and starts the pipeline in a background thread.
    Returns the job_id if successfully started, otherwise None.
    """
    if not uploaded_files:
        logger.warning("No files provided for ingestion.")
        return None

    # 1. Create the initial job record in the DB
    job_id = src.utils.audit_db_manager.create_ingestion_job(total_files=len(uploaded_files))
    if not job_id:
        logger.error("Failed to create ingestion job record in the database.")
        st.error("Failed to start ingestion job (database error).")
        return None

    # 2. Create and start the background thread
    logger.info(f"Starting background thread for ingestion job {job_id}...")
    thread = threading.Thread(
        target=run_ingestion_pipeline_thread,
        kwargs={
            "job_id": job_id,
            "uploaded_files": uploaded_files,
            "config": config,
            "use_cache": use_cache,
            "mistral_client": mistral_client,
            "neo4j_exporter": neo4j_exporter,
            "embedding_model_resource": embedding_model_resource,
            "chroma_collection_resource": chroma_collection_resource,
            "requests_session": requests_session_resource,
            "nlp_pipeline_resource": nlp_pipeline_resource
            # Pass other resources here
        },
        daemon=True # Allows app to exit even if thread is running (consider implications)
    )
    _pipeline_threads[job_id] = thread # Store thread reference (optional)
    thread.start()
    logger.info(f"Ingestion job {job_id} started in background thread.")

    return job_id

# --- Optional: Function to check thread status (basic) ---
def is_job_running(job_id: str) -> bool:
    """Checks if the thread for a given job ID is still alive."""
    thread = _pipeline_threads.get(job_id)
    if thread:
        return thread.is_alive()
    return False # Thread not found or already finished