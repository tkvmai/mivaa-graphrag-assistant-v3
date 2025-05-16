#!/usr/bin/env python3
"""
LLM interaction utilities for knowledge graph generation.
Handles LLM API calls and JSON extraction from messy LLM responses.
Supports OpenAI, Mistral AI, Google Gemini, and Ollama compatible endpoints.
"""

import requests
import json
import re
import logging
import time # Import time for potential delays
from typing import Optional

logger = logging.getLogger(__name__)

# --- ADDED: Custom Exception for Quota Errors ---
class QuotaError(Exception):
    """Custom exception for API quota/rate limit errors."""
    pass

# --- MODIFIED call_llm function ---
def call_llm(
    model: str,
    user_prompt: str,
    api_key: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 1000,
    temperature: float = 0.2,
    base_url: Optional[str] = None,
    session: Optional[requests.Session] = None # ADDED: Optional session argument
) -> str:
    """
    Generic LLM API caller supporting OpenAI, Mistral, Google Gemini, and Ollama.
    Uses a provided requests.Session for connection pooling if available.
    Raises QuotaError for HTTP 429 status codes.

    Args:
        model: Model name (e.g., "gpt-4-turbo", "mistral-large-latest", "gemini-1.5-pro-latest", "phi3:mini")
        user_prompt: User prompt string
        api_key: The API key (use "ollama" or any non-empty string for local Ollama)
        system_prompt: Optional system context
        max_tokens: Response length
        temperature: LLM creativity level
        base_url: API endpoint URL (e.g., "https://api.openai.com/v1/chat/completions",
                  "https://api.mistral.ai/v1/chat/completions",
                  "https://generativelanguage.googleapis.com/v1beta/models",
                  "http://localhost:11434/v1/chat/completions")
        session: An optional requests.Session object for making the request.

    Returns:
        LLM Response Text

    Raises:
        ValueError: If base_url or api_key is missing.
        QuotaError: If the API returns a 429 status code.
        TimeoutError: If the request times out.
        requests.exceptions.RequestException: For other request-related errors.
        Exception: For general API errors or response parsing issues.
    """
    if not base_url:
        raise ValueError("base_url must be provided")
    if not api_key: # API key is required, even if just a placeholder for Ollama
         raise ValueError("api_key must be provided (use 'ollama' or any string for local Ollama)")

    headers = {'Content-Type': 'application/json'}
    is_gemini = "generativelanguage.googleapis.com" in base_url
    is_ollama = "localhost" in base_url or "ollama" in base_url # Simple check for local Ollama

    # Add Authorization Header only for non-Gemini, non-Ollama cloud APIs
    if not is_gemini and not is_ollama:
        headers['Authorization'] = f"Bearer {api_key}"

    # Construct payload based on API type
    if is_gemini:
        # Gemini specific payload structure
        payload = {
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }
        if system_prompt:
             payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        api_url = f"{base_url}/{model}:generateContent?key={api_key}"

    else:
        # OpenAI / Mistral / Ollama Style Payload
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': user_prompt})
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        api_url = base_url

    requester = session if session else requests
    response = None
    request_timeout = 60 # Timeout in seconds

    try:
        logger.debug(f"Sending request to {api_url} with model {model} (Timeout: {request_timeout}s)")
        response = requester.post(api_url, headers=headers, json=payload, timeout=request_timeout)

        # --- MODIFIED: Check for 429 before raising general HTTPError ---
        if response.status_code == 429:
            logger.warning(f"API Quota/Rate Limit Exceeded (HTTP 429). Response: {response.text}")
            # Extract potential retry-after header (optional)
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                 logger.warning(f"Server suggests waiting {retry_after} seconds.")
            raise QuotaError(f"API Quota/Rate Limit Exceeded (HTTP 429). Wait and retry. Details: {response.text}")

        response.raise_for_status() # Raise HTTPError for other bad responses (4xx or 5xx)

        # Extract response content based on API type
        response_data = response.json()
        if is_gemini:
            candidates = response_data.get("candidates", [])
            if candidates and isinstance(candidates, list):
                 if candidates[0].get("finishReason") == "SAFETY":
                     logger.warning("Gemini response blocked due to safety settings.")
                     safety_ratings = candidates[0].get("safetyRatings", [])
                     logger.warning(f"Safety Ratings: {safety_ratings}")
                     return "[ERROR: Response blocked by safety settings]"
                 content = candidates[0].get("content", {})
                 parts = content.get("parts", [])
                 if parts and isinstance(parts, list):
                     return parts[0].get("text", "")
            # Check for explicit quota error in Gemini response body (if status wasn't 429)
            if "error" in response_data and "Resource has been exhausted" in response_data["error"].get("message", ""):
                 logger.warning(f"Gemini Quota Exceeded (detected in response body): {response_data['error']}")
                 raise QuotaError(f"Gemini Quota Exceeded. Details: {response_data['error']}")
            logger.warning(f"Could not extract text from Gemini response: {response_data}")
            return ""
        else:
            choices = response_data.get("choices", [])
            if choices and isinstance(choices, list):
                message = choices[0].get("message", {})
                return message.get("content", "")
            if "error" in response_data:
                 # Check for common quota messages in OpenAI/Mistral errors even if status wasn't 429
                 error_msg = response_data['error'].get("message", "").lower()
                 if "rate limit" in error_msg or "quota" in error_msg:
                      logger.warning(f"Cloud API Quota/Rate Limit Error (detected in response body): {response_data['error']}")
                      raise QuotaError(f"Cloud API Quota/Rate Limit Error. Details: {response_data['error']}")
                 else:
                      logger.warning(f"API returned an error: {response_data['error']}")
                      return f"[ERROR: {response_data['error']}]"
            logger.warning(f"Could not extract text from OpenAI/Mistral/Ollama style response: {response_data}")
            return ""

    except requests.exceptions.Timeout:
        logger.error(f"API request timed out after {request_timeout} seconds to {api_url}")
        raise TimeoutError(f"API request timed out to {api_url}")
    except requests.exceptions.HTTPError as e: # Catch other HTTP errors (already checked for 429)
        error_content = response.text if response is not None else "No response received"
        logger.error(f"API HTTP request failed: {e}")
        logger.error(f"Response status: {response.status_code if response is not None else 'N/A'}")
        logger.error(f"Response body: {error_content}")
        raise Exception(f"API request failed with status {response.status_code if response is not None else 'N/A'}: {error_content}") from e
    except requests.exceptions.RequestException as e: # Catch other connection errors
        logger.error(f"API request failed (network/connection): {e}")
        raise Exception(f"API request failed (network/connection): {e}") from e
    except (KeyError, IndexError, TypeError) as e:
         response_data_str = str(response_data) if 'response_data' in locals() else 'N/A'
         logger.error(f"Failed to parse LLM response structure: {e}")
         logger.error(f"Raw response data: {response_data_str[:500]}...")
         raise ValueError(f"Invalid response structure received from LLM: {e}") from e


def fix_malformed_json(text: str) -> str:
    # Replace smart quotes with straight quotes
    text = text.replace("“", "\"").replace("”", "\"")

    # Remove triple backticks
    text = text.replace("```", "")

    # Fix common unclosed quote issues at the end
    text = re.sub(r'“([^”]*)$', r'"\1"', text)

    return text.strip()

def extract_json_from_text(text):
    """
    Extract JSON array from text that might contain extra content, noise, or markdown.

    Args:
        text: Text that may contain JSON

    Returns:
        The parsed JSON if found, else None
    """

    import json
    import re

    if not text or not isinstance(text, str):
        logger.warning("extract_json_from_text received empty or non-string input.")
        return None

    logger.debug(f"Attempting to extract JSON from text starting with: {text[:100]}...") # Log beginning of text

    # Auto-fix: Replace curly quotes with straight quotes
    text = text.replace("“", "\"").replace("”", "\"")

    # Auto-fix: Remove any unexpected trailing backticks or markdown formatting
    text = re.sub(r'^```(?:json)?\s*', '', text) # Remove starting ```json or ```
    text = re.sub(r'\s*```$', '', text) # Remove ending ```

    # Find the start of the first JSON array or object
    start_bracket = text.find('[')
    start_brace = text.find('{')

    if start_bracket == -1 and start_brace == -1:
        logger.warning("No JSON array '[' or object '{' start found in text.")
        return None

    # Determine if we expect an array or object based on which comes first
    if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
        # Looks like a single JSON object, not an array
        start_char = '{'
        end_char = '}'
        start_idx = start_brace
        logger.debug("Detected potential JSON object start '{'.")
    else:
        # Looks like a JSON array
        start_char = '['
        end_char = ']'
        start_idx = start_bracket
        logger.debug("Detected potential JSON array start '['.")

    # Try to find the matching end character
    bracket_count = 0
    end_idx = -1
    for i in range(start_idx, len(text)):
        if text[i] == start_char:
            bracket_count += 1
        elif text[i] == end_char:
            bracket_count -= 1
            if bracket_count == 0:
                end_idx = i
                break # Found the matching end

    if end_idx != -1:
        json_str = text[start_idx : end_idx + 1]
        logger.debug(f"Extracted potential JSON string: {json_str[:100]}...")
        try:
            # Basic cleaning before parsing
            json_str = json_str.replace('½', 'half')
            json_str = json_str.replace('°', ' degrees')
            json_str = json_str.replace('@', ' at ')
            # Attempt direct parsing of the extracted string
            parsed_json = json.loads(json_str)
            logger.info("Successfully parsed extracted JSON.")
            return parsed_json
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse extracted JSON string: {e}")
            logger.debug(f"Problematic JSON string: {json_str}")
            # Optionally add more fixing logic here if needed, like regex fixes
            # For now, just return None if direct parse fails
            return None
    else:
        logger.warning(f"Could not find matching end character '{end_char}' for start at index {start_idx}.")
        # Attempt recovery only if it started with '[' (expecting an array of objects)
        if start_char == '[':
            logger.info("Attempting recovery of JSON objects within the text...")
            objects = []
            # Regex to find structures that look like {"key": "value", ...}
            # This is a simplified regex and might need refinement
            obj_pattern = re.compile(r'\{\s*"[^"]+"\s*:\s*"[^"]*"(?:\s*,\s*"[^"]+"\s*:\s*"[^"]*")*\s*\}')
            for match in obj_pattern.finditer(text):
                objects.append(match.group(0))

            if objects:
                reconstructed_json_str = "[" + ",".join(objects) + "]"
                logger.info(f"Reconstructed JSON array with {len(objects)} objects.")
                try:
                    parsed_json = json.loads(reconstructed_json_str)
                    logger.info("Successfully parsed reconstructed JSON array.")
                    return parsed_json
                except json.JSONDecodeError as e:
                    logger.error(f"Could not parse reconstructed JSON: {e}")
                    logger.debug(f"Reconstructed JSON string: {reconstructed_json_str}")
                    return None
            else:
                 logger.error("Recovery failed: No valid JSON objects found within the text.")
                 return None
        else:
             logger.error("Could not extract a complete JSON structure.")
             return None
