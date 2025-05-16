import logging
from typing import Optional, List

# LlamaIndex imports for type hinting (adjust if your specific types differ)
# Using base types for broader compatibility, replace with specific classes if known
# e.g., from llama_index.llms.openai import OpenAI
# e.g., from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core.llms.llm import LLM
from llama_index.core.graph_stores.types import GraphStore
from llama_index.core.prompts import ChatPromptTemplate # Corrected import path

# Setup logger for this module
# Configure logging level and format in your main application entry point
logger = logging.getLogger(__name__)

# System prompt defining the LLM's role as a Cypher expert correcting errors.
CORRECT_CYPHER_SYSTEM_TEMPLATE = """You are a Cypher expert reviewing a statement written by a junior developer.
You need to correct the Cypher statement based on the provided errors. No pre-amble."
Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!"""

# User prompt providing the context (schema, original query, errors) for correction.
CORRECT_CYPHER_USER_TEMPLATE = """Check for invalid syntax or semantics and return a corrected Cypher statement.

Schema:
{schema}

Note: Do not include any explanations or apologies in your responses.
Do not wrap the response in any backticks or anything else.
Respond with a Cypher statement only!

Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.

The question is:
{question}

The Cypher statement is:
{cypher}

The errors are:
{errors}

Corrected Cypher statement: """


async def correct_cypher_step(
    llm: LLM,
    graph_store: GraphStore,
    subquery: str,
    cypher: str,
    errors: str,
    schema_exclude_types: Optional[List[str]] = None # Default to None for broader compatibility
) -> Optional[str]:
    """
    Uses an LLM to correct a given Cypher query based on schema information and error messages.

    Args:
        llm: The LlamaIndex LLM instance to use for correction.
        graph_store: The LlamaIndex GraphStore instance providing schema access.
        subquery: The original natural language question or subquery that led to the Cypher.
        cypher: The incorrect Cypher query string.
        errors: The error message(s) received when executing the incorrect query.
        schema_exclude_types: Optional list of node labels to exclude from the schema string.
                              Defaults to None (include all types). Pass ["Actor", "Director"]
                              if you want to replicate the original behavior.

    Returns:
        Optional[str]: The corrected Cypher query suggested by the LLM, or None if correction fails.
    """
    # --- Input Validation ---
    # Check if essential arguments are provided and have basic validity
    if not llm:
        logger.error("correct_cypher_step: LLM instance is required.")
        return None
    if not graph_store:
        logger.error("correct_cypher_step: GraphStore instance is required.")
        return None
    if not subquery or not isinstance(subquery, str) or not subquery.strip():
        logger.error("correct_cypher_step: Valid 'subquery' string is required.")
        return None
    if not cypher or not isinstance(cypher, str) or not cypher.strip():
        logger.error("correct_cypher_step: Valid 'cypher' query string is required.")
        return None
    if not errors: # Allow empty error string, but log if None
         logger.warning("correct_cypher_step: 'errors' argument is missing or empty.")
         errors = "" # Ensure it's at least an empty string
    elif not isinstance(errors, str):
        logger.warning(f"correct_cypher_step: 'errors' argument is not a string ({type(errors)}). Converting.")
        errors = str(errors) # Convert non-string errors

    logger.info(f"Attempting to correct Cypher for subquery: '{subquery}'")
    logger.debug(f"Original Cypher:\n{cypher}")
    logger.debug(f"Errors:\n{errors}")

    try:
        # --- Schema Retrieval ---
        # Retrieve the graph schema string using the provided graph_store instance.
        # Pass the configurable exclude_types list.
        try:
            schema = graph_store.get_schema()
            logger.debug(f"Using schema (excluding {schema_exclude_types}):\n{schema[:500]}...") # Log start of schema
        except Exception as schema_e:
            logger.error(f"Failed to retrieve schema from graph_store: {schema_e}", exc_info=True)
            return None # Cannot proceed without schema

        # --- Prompt Formatting ---
        # Define the message structure for the LLM.
        correct_cypher_messages = [
            ("system", CORRECT_CYPHER_SYSTEM_TEMPLATE),
            ("user", CORRECT_CYPHER_USER_TEMPLATE),
        ]
        # Create the prompt template object.
        correct_cypher_prompt = ChatPromptTemplate.from_messages(correct_cypher_messages)

        # --- LLM Call with Error Handling ---
        try:
            # Format the prompt with the specific details (question, schema, errors, original cypher).
            # Call the LLM asynchronously to get the corrected query.
            response = await llm.achat(
                correct_cypher_prompt.format_messages(
                    question=subquery,
                    schema=schema,
                    errors=errors,
                    cypher=cypher,
                )
            )
            # Extract the text content from the response message, handling potential None values.
            corrected_query = response.message.content.strip() if response and response.message and response.message.content else None

        except Exception as llm_e:
            # Log any errors encountered during the LLM API call.
            logger.error(f"LLM call failed during Cypher correction: {llm_e}", exc_info=True)
            return None # Indicate correction failure due to LLM error

        # --- Output Validation ---
        # Check if the LLM returned any content.
        if not corrected_query:
            logger.warning("LLM response for correction was empty.")
            return None

        # Basic check if the response looks like a Cypher query.
        # A more robust validation could involve a Cypher parser if available.
        if "MATCH" not in corrected_query.upper() or "RETURN" not in corrected_query.upper():
             logger.warning(f"LLM correction response doesn't seem like a valid query (missing MATCH/RETURN): {corrected_query}")
             # Returning None might be safer than returning potentially non-Cypher text.
             return None

        # Check if the LLM simply returned the original query.
        if corrected_query == cypher.strip():
            logger.warning("LLM correction returned the original query. No change made.")
            # Returning None signifies no *useful* correction was made.
            return None

        # If all checks pass, log and return the corrected query.
        logger.info(f"LLM suggested corrected Cypher:\n{corrected_query}")
        return corrected_query

    except Exception as e:
        # Catch any other unexpected errors during the process.
        logger.error(f"Unexpected error in correct_cypher_step: {e}", exc_info=True)
        return None
