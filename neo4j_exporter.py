import logging
import re
from neo4j import GraphDatabase, Session, Transaction # Import Session/Transaction for typing
from typing import List, Dict, Optional, Any, Set, Tuple
import sys # Import sys for exit handling in main block
from pathlib import Path
import json
import configparser
# Use tomllib if available (Python 3.11+) otherwise fallback to toml
try:
    import tomllib
except ImportError:
    try:
        import toml
    except ImportError:
        tomllib = None # Indicate neither is available
        toml = None

# Setup logger (ensure handlers aren't added multiple times if imported)
logger = logging.getLogger(__name__)
# Set default level - can be overridden by external config
logger.setLevel(logging.DEBUG) # Set to DEBUG to capture detailed logs

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout) # Log to standard output
    # More detailed formatter
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False # Prevent duplicate logs if root logger has handlers

class Neo4jExporter:
    """
    Exports triples (subject, predicate, object) from a list of dictionaries
    into a Neo4j graph database.
    It creates :Entity nodes with additional specific labels (e.g., :Well, :Formation)
    and specific relationship types based on sanitized predicates.
    It also handles linking entities to :Chunk nodes representing source text.

    ASSUMPTION: The 'subject' and 'object' values in the input triples list
    are expected to be the final, standardized entity names.
    ASSUMPTION: Input triples may contain 'subject_type' and 'object_type' keys.
    """
    def __init__(self, uri: str, user: str, password: str):
        """
        Initializes the exporter and connects to the Neo4j database.

        Args:
            uri: The connection URI for the Neo4j database (e.g., "neo4j://localhost:7687").
            user: The username for Neo4j authentication.
            password: The password for Neo4j authentication.

        Raises:
            ConnectionError: If the connection to Neo4j fails.
        """
        self.driver: Optional[GraphDatabase.driver] = None
        try:
            # Establish connection to Neo4j
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # Verify the connection is successful
            self.driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j at %s", uri)
        except Exception as e:
            logger.error("Fatal: Failed to connect to Neo4j at %s. Error: %s", uri, e, exc_info=True)
            # Raise a specific error for the caller to handle
            raise ConnectionError(f"Failed to connect to Neo4j: {e}") from e

    def close(self):
        """Closes the Neo4j database driver connection."""
        if self.driver:
            try:
                self.driver.close()
                logger.info("Closed Neo4j connection.")
                self.driver = None
            except Exception as e:
                logger.error("Error closing Neo4j connection: %s", e, exc_info=True)

    def _sanitize_label(self, label: str) -> Optional[str]:
        """
        Sanitizes a string to be used as a Neo4j node label.
        Removes non-alphanumeric characters. Ensures it starts with a letter.
        Returns None if the label becomes invalid after sanitization.

        Args:
            label: The potential label string (e.g., entity type).

        Returns:
            A sanitized string suitable for use as a label, or None if invalid.
        """
        if not isinstance(label, str) or not label.strip():
            return None # Skip empty or non-string types

        # Remove leading/trailing whitespace
        sanitized = label.strip()
        # Remove non-alphanumeric characters
        sanitized = re.sub(r'[^a-zA-Z0-9]', '', sanitized)
        # Ensure it starts with a letter
        if sanitized and sanitized[0].isalpha():
            # Optional: Convert to TitleCase or keep as is? Let's keep as is for now.
            return sanitized
        else:
            logger.warning("Could not sanitize '%s' into a valid Neo4j label.", label)
            return None # Invalid after sanitization

    def sanitize_predicate(self, predicate: str, to_upper: bool = True, replace_non_alnum: bool = True) -> str:
        """
        Sanitizes a predicate string to be used as a Neo4j relationship type.
        Defaults to converting to uppercase and replacing non-alphanumeric characters with underscores.

        Args:
            predicate: The original predicate string.
            to_upper: Whether to convert the result to uppercase (default: True).
            replace_non_alnum: Whether to replace non-alphanumeric characters with '_' (default: True).

        Returns:
            The sanitized string suitable for use as a relationship type.
        """
        if not isinstance(predicate, str):
            logger.warning("Predicate is not a string: %s. Returning empty string.", predicate)
            return "" # Handle non-string input gracefully
        result = predicate.strip()
        if not result:
             logger.warning("Predicate is empty after stripping. Returning empty string.")
             return "" # Handle empty predicate

        if to_upper:
            result = result.upper()
        if replace_non_alnum:
            # Replace sequences of non-alphanumeric characters with a single underscore
            result = re.sub(r'[^a-zA-Z0-9_]+', '_', result)
            # Ensure it doesn't start or end with an underscore (optional, but good practice)
            result = result.strip('_')
            # Ensure it doesn't start with a number (Neo4j types shouldn't)
            if result and result[0].isdigit():
                result = '_' + result

        # Handle cases where sanitization might result in an empty string
        if not result:
             logger.warning("Predicate '%s' resulted in empty string after sanitization.", predicate)
             return "" # Return empty to prevent Cypher errors

        return result

    def store_triples(self, triples: List[Dict], store_chunks: bool = True, sanitize: bool = True) -> Tuple[bool, int]: # Modified return type
        """
        Stores a list of triples in Neo4j within a single transaction.
        Adds specific labels based on 'subject_type' and 'object_type'.

        Args:
            triples: A list of dictionaries, each representing a triple.
                     Expected keys: 'subject', 'predicate', 'object'.
                     Optional keys: 'inferred', 'confidence', 'chunk_id', 'chunk_text',
                                    'subject_type', 'object_type'.
            store_chunks: Whether to create :Chunk nodes and relationships (default: True).
            sanitize: Whether to sanitize the predicate for use as relationship type (default: True).

        Returns:
            Tuple[bool, int]: (True if transaction was successful, count of triples attempted),
                              (False, 0) if an error occurred.
        """
        num_attempted = len(triples) # Count triples intended for this batch
        if not triples:
            logger.warning("No triples provided to store in Neo4j.")
            return True, 0 # No error, just processed 0

        if not self.driver:
             logger.error("Neo4j driver not initialized. Cannot store triples.")
             return False, 0 # Indicate failure

        processed_chunk_ids: Set[str] = set()

        logger.info(f"Starting batch transaction to store/update {num_attempted} triples...")
        try:
            # Use write_transaction for automatic retry handling on transient errors
            # Ensure the database name is correct if not default 'neo4j'
            with self.driver.session(database="neo4j") as session:
                session.execute_write(
                    self._store_triples_transaction, # The function to execute
                    triples,                         # Arguments for the function
                    processed_chunk_ids,
                    store_chunks,
                    sanitize
                )
            logger.info(f"Batch transaction completed successfully for {num_attempted} attempted triples.")
            return True, num_attempted # Return success and the count attempted

        except Exception as e:
            logger.error(f"Batch transaction failed during execute_write: {e}", exc_info=True)
            # Re-raising the exception might be useful for the calling pipeline thread
            # raise e # Uncomment if you want the pipeline thread to catch the specific Neo4j error
            return False, 0 # Return failure and 0 count

    # Private helper method to run within the transaction context
    def _store_triples_transaction(self, tx: Transaction, triples: List[Dict], processed_chunk_ids: Set[str], store_chunks: bool, sanitize: bool):
        """Internal function executed within a Neo4j transaction to store triples."""
        skipped_in_batch = 0
        processed_in_batch = 0

        for i, triple in enumerate(triples):
            subject = triple.get("subject")
            predicate = triple.get("predicate")
            object_ = triple.get("object")
            subject_type = triple.get("subject_type") # Get optional type
            object_type = triple.get("object_type")   # Get optional type

            # *** ADDED VALIDATION: Check for None or empty strings ***
            if not all(isinstance(item, str) and item.strip() for item in [subject, predicate, object_]):
                logger.warning("Skipping invalid or incomplete triple #%d (empty S/P/O): %s", i, triple)
                skipped_in_batch += 1
                continue

            # Strip whitespace just in case
            subject = subject.strip()
            predicate = predicate.strip()
            object_ = object_.strip()

            # --- Sanitize Labels and Relationship Type ---
            sanitized_subj_label = self._sanitize_label(subject_type) if subject_type else None
            sanitized_obj_label = self._sanitize_label(object_type) if object_type else None
            rel_type = self.sanitize_predicate(predicate) if sanitize else predicate

            if not rel_type: # Skip if sanitization failed or resulted in empty string
                 logger.warning("Skipping triple #%d due to empty sanitized predicate for original: '%s'", i, predicate)
                 skipped_in_batch += 1
                 continue

            # --- Construct Node Patterns with Optional Labels ---
            # Base label is always :Entity
            subj_pattern = f"s:Entity"
            if sanitized_subj_label:
                subj_pattern += f":`{sanitized_subj_label}`" # Add specific label if valid

            obj_pattern = f"o:Entity"
            if sanitized_obj_label:
                obj_pattern += f":`{sanitized_obj_label}`" # Add specific label if valid

            # --- Optional Properties ---
            inferred = triple.get("inferred", False)
            try:
                # Ensure confidence is treated as float, default to 1.0
                confidence = float(triple.get("confidence", 1.0))
            except (ValueError, TypeError):
                confidence = 1.0 # Default on conversion error

            # *** ADDED DEBUG LOGGING before Cypher execution ***
            logger.debug(
                f"Processing Triple #{i}: "
                f"Subject='{subject}' (Type: {subject_type}), Predicate='{predicate}', Object='{object_}' (Type: {object_type}), "
                f"RelType='{rel_type}', Inferred={inferred}, Confidence={confidence}"
            )

            # APOC Version - Requires APOC Plugin Installed in Neo4j
            cypher_rel = f"""
            MERGE (s:Entity {{name: $subject}})
            ON CREATE SET s.created_at = timestamp(), s.name = $subject
            WITH s
            // Use parameter for label, APOC handles if it's null/empty gracefully usually
            CALL apoc.create.addLabels(s, [$subject_label_param]) YIELD node AS subjNode

            MERGE (o:Entity {{name: $object}})
            ON CREATE SET o.created_at = timestamp(), o.name = $object
            WITH subjNode, o // Carry subject node forward
            // Use parameter for label
            CALL apoc.create.addLabels(o, [$object_label_param]) YIELD node AS objNode

            // MERGE relationship between the specific nodes found/created above
            MERGE (subjNode)-[r:`{rel_type}`]->(objNode)
            ON CREATE SET
                r.original = $original_predicate,
                r.inferred = $inferred,
                r.confidence = $confidence,
                r.created_at = timestamp()
            ON MATCH SET
                r.confidence = $confidence,
                r.inferred = $inferred,
                r.updated_at = timestamp()
            """
            # Parameters for APOC version
            params = {
                "subject": subject,
                "object": object_,
                "subject_label_param": sanitized_subj_label,  # Pass label as param (APOC handles null)
                "object_label_param": sanitized_obj_label,  # Pass label as param (APOC handles null)
                "original_predicate": predicate,
                "inferred": inferred,
                "confidence": confidence
            }
            try:
                tx.run(cypher_rel, **params) # Use **params to unpack dictionary
                logger.debug(f"Successfully merged relationship: ('{subject}')-[:`{rel_type}`]->('{object_}') with labels")
            except Exception as e:
                 # Log specific triple that failed if possible
                 logger.error("Failed to run relationship MERGE for triple #%d ('%s' -[:`%s`]-> '%s'): %s",
                              i, subject, rel_type, object_, e, exc_info=True)
                 # Decide whether to continue or raise to abort transaction
                 # For now, log and continue
                 skipped_in_batch += 1
                 continue # Skip chunk handling for this failed triple

            # --- Chunk Handling (Optional) ---
            if store_chunks:
                # Use 'chunk_id' key if present (more robust than 'chunk')
                chunk_id_val = triple.get("chunk_id", triple.get("chunk"))
                chunk_text = triple.get("chunk_text", "")

                if chunk_id_val is not None and isinstance(chunk_text, str) and chunk_text.strip():
                    chunk_id = str(chunk_id_val) # Ensure string ID
                    chunk_text_stripped = chunk_text.strip()
                    logger.debug(f"Processing chunk link for triple #{i}: chunk_id='{chunk_id}'")

                    # MERGE Chunk node if not already processed in this batch
                    if chunk_id not in processed_chunk_ids:
                        logger.debug(f"Merging Chunk node: id='{chunk_id}'")
                        try:
                            # *** MODIFIED: Removed ON MATCH SET c.text ***
                            tx.run(
                                """
                                MERGE (c:Chunk {id: $chunk_id})
                                ON CREATE SET c.text = $chunk_text, c.created_at = timestamp()
                                ON MATCH SET c.updated_at = timestamp()
                                """,
                                chunk_id=chunk_id,
                                chunk_text=chunk_text_stripped
                            )
                            processed_chunk_ids.add(chunk_id)
                            logger.debug(f"Successfully merged Chunk node: id='{chunk_id}'")
                        except Exception as e:
                             logger.error("Failed to MERGE Chunk node with id %s: %s", chunk_id, e, exc_info=True)
                             # Continue without linking if chunk merge fails? Or abort? For now, continue.

                    # Link entities to the chunk (only if chunk merge succeeded implicitly or was skipped)
                    # Check if chunk_id is known (either created now or previously added in this batch)
                    if chunk_id in processed_chunk_ids:
                        logger.debug(f"Merging :FROM_CHUNK links for triple #{i} to chunk '{chunk_id}'")
                        try:
                            # Merge relationships from both subject and object to the chunk
                            # Use the dynamic label patterns here as well for matching
                            tx.run(
                                f"""
                                MATCH ({subj_pattern} {{name: $subject}})
                                MATCH ({obj_pattern} {{name: $object}})
                                MATCH (c:Chunk {{id: $chunk_id}})
                                MERGE (s)-[:FROM_CHUNK]->(c)
                                MERGE (o)-[:FROM_CHUNK]->(c)
                                """,
                                chunk_id=chunk_id,
                                subject=subject,
                                object=object_
                            )
                            logger.debug(f"Successfully merged :FROM_CHUNK links for triple #{i}")
                        except Exception as e:
                             logger.error("Failed to link entities ('%s', '%s') to chunk '%s' for triple #%d: %s",
                                          subject, object_, chunk_id, i, e, exc_info=True)
                             # Log and continue
                # else: logger.debug(f"Skipping chunk link for triple #{i}: missing chunk_id or chunk_text.")

            processed_in_batch += 1 # Increment only if relationship merge succeeded

        logger.info("Transaction function finished. Processed attempts: %d, Skipped/Failed: %d", processed_in_batch, skipped_in_batch)
        # The transaction will commit automatically if no exception is raised here.


    def get_related_facts_with_context(self, entity_name: str, predicate_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves facts (relationships) connected to a given entity, including context from linked chunks.
        Fetches both outgoing and incoming relationships.

        Args:
            entity_name: The name of the central entity.
            predicate_filter: Optional. If provided, filters relationships by the original predicate text
                              (case-insensitive) by searching the 'original' property on the relationship.

        Returns:
            A list of dictionaries, each representing a fact with keys:
            'subject', 'predicate' (original text), 'object', 'chunk_text', 'type' (sanitized type).
        """
        if not self.driver:
             logger.error("Neo4j driver not initialized. Cannot get related facts.")
             return []

        logger.info("Fetching related facts for entity: '%s', predicate filter: '%s'", entity_name, predicate_filter)

        # Query now returns r.original as 'predicate' for user display,
        # and type(r) as 'type' for the sanitized type.
        # Filtering is done on r.original for more natural language matching.
        base_query = """
        // Outgoing relationships
        MATCH (e:Entity)-[r]->(o:Entity)
        WHERE toLower(e.name) = toLower($name)
        OPTIONAL MATCH (e)-[:FROM_CHUNK]->(c1:Chunk)
        OPTIONAL MATCH (o)-[:FROM_CHUNK]->(c2:Chunk)
        WITH e, r, o, coalesce(c1.text, c2.text, "") AS chunk_text
        {predicate_filter_clause} // Apply predicate filter here if provided
        RETURN e.name AS subject, r.original AS predicate, o.name AS object, chunk_text, type(r) as type

        UNION

        // Incoming relationships
        MATCH (s:Entity)-[r]->(e:Entity)
        WHERE toLower(e.name) = toLower($name)
        OPTIONAL MATCH (s)-[:FROM_CHUNK]->(c1:Chunk)
        OPTIONAL MATCH (e)-[:FROM_CHUNK]->(c2:Chunk)
        WITH s, r, e, coalesce(c1.text, c2.text, "") AS chunk_text
        {predicate_filter_clause} // Apply predicate filter here if provided
        RETURN s.name AS subject, r.original AS predicate, e.name AS object, chunk_text, type(r) as type
        """

        # Dynamically add the WHERE clause for predicate filtering on r.original
        clause = ""
        parameters: Dict[str, Any] = {"name": entity_name}
        if predicate_filter:
            # Use CONTAINS for flexible matching on the original predicate text
            clause = "WHERE toLower(r.original) CONTAINS toLower($predicate)"
            parameters["predicate"] = predicate_filter # Use the original filter text

        query = base_query.format(predicate_filter_clause=clause)
        logger.debug("Executing get_related_facts query with params: %s", parameters)
        logger.debug("Full Query:\n%s", query)

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(query, **parameters)
                records = [record.data() for record in result]
                logger.info("Found %d related facts for entity '%s'.", len(records), entity_name)
                return records
        except Exception as e:
            logger.error("Failed to execute get_related_facts query for entity '%s': %s", entity_name, e, exc_info=True)
            return []


    def get_graph_stats(self) -> Dict[str, int]:
        """
        Calculates basic statistics about the graph (node and relationship counts).
        Uses the recommended CALL {} syntax for subqueries.

        Returns:
            A dictionary containing counts for entities, chunks, and total relationships.
            Returns zeros if the query fails or the graph is empty.
        """
        if not self.driver:
             logger.error("Neo4j driver not initialized. Cannot get graph stats.")
             return {"entity_count": 0, "chunk_count": 0, "relationship_count": 0}

        logger.info("Calculating graph statistics...")
        # Updated query using CALL {} syntax to address deprecation warning
        query = """
        CALL { MATCH (n:Entity) RETURN count(n) AS entity_count }
        CALL { MATCH (n:Chunk) RETURN count(n) AS chunk_count }
        CALL { MATCH ()-[r]->() RETURN count(r) AS relationship_count }
        RETURN entity_count, chunk_count, relationship_count
        """
        try:
            # Use execute_query for managed read transactions
            records, summary, keys = self.driver.execute_query(query, database_="neo4j")
            record = records[0] if records else None # Get the first (only) record

            if record:
                stats = {
                    "entity_count": record.get("entity_count", 0), # Use .get for safety
                    "chunk_count": record.get("chunk_count", 0),
                    "relationship_count": record.get("relationship_count", 0),
                }
                logger.info("Graph Stats: %s", stats)
                return stats
            else:
                logger.warning("Graph stats query returned no record.")
                return {"entity_count": 0, "chunk_count": 0, "relationship_count": 0}
        except Exception as e:
            logger.error("Failed to execute get_graph_stats query: %s", e, exc_info=True)
            return {"entity_count": 0, "chunk_count": 0, "relationship_count": 0}

    # --- Context Manager Methods --- ADDED
    def __enter__(self):
        """
        Enter the runtime context related to this object. Allows using 'with Neo4jExporter(...)'.
        """
        # The connection is established in __init__. Just return self.
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the runtime context related to this object. Ensures connection closure.
        """
        logger.debug("Exiting context manager, ensuring Neo4j connection is closed.")
        self.close() # Call the existing close method
        # Return False to propagate any exceptions that occurred within the 'with' block
        return False


# --- Example Usage (main block) ---
if __name__ == "__main__":
    # Imports moved inside main block where they are used
    # from pathlib import Path # Imported globally now
    # import json # Imported globally now
    # import configparser # Imported globally now
    # import sys # Imported globally now

    print("--- Neo4j Exporter Script Start ---")

    # --- Configuration Loading ---
    config_path = Path("graph_config.ini")
    if not config_path.is_file():
        print(f"ERROR: Configuration file not found at {config_path}")
        logger.critical("Configuration file not found at %s", config_path)
        sys.exit(1)

    config = configparser.ConfigParser()
    config.read(config_path)

    print("[DEBUG] Config Sections found:", config.sections())
    try:
        # Use .get with fallback=None for safer access
        uri = config.get("neo4j", "uri", fallback=None)
        user = config.get("neo4j", "user", fallback=None)
        password = config.get("neo4j", "password", fallback=None)
        file_path = None # Initialize file_path
        if "data" in config and "json_file" in config["data"]:
             file_path_str = config.get("data", "json_file", fallback=None)
             if file_path_str:
                  file_path = Path(file_path_str)
        if not file_path:
             logger.error("Missing [data] section or 'json_file' key in graph_config.ini, or path is empty.")
             # Decide if this is fatal or if script can run without input file (e.g., only run stats)
             # For now, let's assume it's required for the main logic below.


        # Validate essential config
        if not all([uri, user, password, file_path]): # Check file_path here
             missing = [
                 "neo4j.uri" if not uri else None,
                 "neo4j.user" if not user else None,
                 "neo4j.password" if not password else None,
                 "data.json_file" if not file_path else None, # Check if file_path was successfully determined
             ]
             missing_str = ", ".join(filter(None, missing))
             error_msg = f"ERROR: Missing required configuration values: {missing_str}"
             print(error_msg)
             logger.critical(error_msg)
             sys.exit(1)

        print(f"[INFO] Neo4j URI: {uri}")
        print(f"[INFO] Input JSON file path: {file_path}")

        # --- File Processing and Export ---
        # Check file existence *after* validation ensures file_path is not None
        if file_path.exists():
            print(f"[INFO] Loading triples from {file_path}...")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    triples_data = json.load(f)
                if not isinstance(triples_data, list):
                     print(f"ERROR: Expected a JSON list in {file_path}, but got {type(triples_data)}.")
                     logger.error("Input JSON file does not contain a list.")
                     sys.exit(1)
                print(f"[INFO] Loaded {len(triples_data)} potential triples from JSON.")

                # --- Add Sample Entity Types for Testing ---
                # This simulates the output from the updated extraction prompt
                for i, t in enumerate(triples_data):
                     if i % 4 == 0:
                         t['subject_type'] = 'Well'
                         t['object_type'] = 'Formation'
                     elif i % 4 == 1:
                         t['subject_type'] = 'Company'
                         t['object_type'] = 'Well'
                     elif i % 4 == 2:
                         t['subject_type'] = 'Formation'
                         t['object_type'] = 'Concept'
                     else:
                         t['subject_type'] = 'Concept'
                         t['object_type'] = 'Location'
                     # Ensure all triples have the keys, even if None
                     t.setdefault('subject_type', None)
                     t.setdefault('object_type', None)
                print("[INFO] Added sample entity types to triples for testing.")
                # -----------------------------------------

            except json.JSONDecodeError as e:
                 print(f"ERROR: Failed to decode JSON from {file_path}: {e}")
                 logger.error("Failed to decode input JSON file: %s", e, exc_info=True)
                 sys.exit(1)
            except Exception as e:
                 print(f"ERROR: Failed to read file {file_path}: {e}")
                 logger.error("Failed to read input JSON file: %s", e, exc_info=True)
                 sys.exit(1)


            # --- Use Context Manager for Exporter ---
            try:
                print("[INFO] Initializing Neo4jExporter via context manager...")
                # The 'with' statement now correctly uses __enter__ and __exit__
                with Neo4jExporter(uri, user, password) as exporter:
                    print("[INFO] Storing triples in Neo4j...")
                    exporter.store_triples(triples_data)

                    # --- Optional: Fetching Example Data and Stats ---
                    print("\n--- Post-Import Verification ---")
                    entity_to_check = "henry" # Example entity
                    print(f"\nFetching related facts for entity: '{entity_to_check}'...")
                    # Test fetching with and without predicate filter
                    related_facts_all = exporter.get_related_facts_with_context(entity_to_check)
                    # Example filter using original predicate text
                    related_facts_inspired = exporter.get_related_facts_with_context(entity_to_check, predicate_filter="inspired by")

                    if related_facts_all:
                        print(f"Found {len(related_facts_all)} total facts related to '{entity_to_check}' (showing max 10):")
                        for i, fact in enumerate(related_facts_all[:10]):
                            # Display original predicate, object, and the sanitized type
                            print(f"  - {fact.get('subject')} -[{fact.get('predicate')}]-> {fact.get('object')} (Type: {fact.get('type')})")
                        if len(related_facts_all) > 10: print("  ...")
                    else:
                        print(f"No related facts found for '{entity_to_check}'.")

                    if related_facts_inspired:
                        print(f"\nFound {len(related_facts_inspired)} facts for '{entity_to_check}' matching filter 'inspired by':")
                        for fact in related_facts_inspired:
                             print(f"  - {fact.get('subject')} -[{fact.get('predicate')}]-> {fact.get('object')} (Type: {fact.get('type')})")
                    else:
                         print(f"\nNo related facts found for '{entity_to_check}' matching filter 'inspired by'.")


                    print("\nFetching graph statistics...")
                    stats = exporter.get_graph_stats()
                    print("\nGraph Summary:")
                    print(f"  Entities (:Entity): {stats.get('entity_count', 'N/A')}")
                    print(f"  Chunks (:Chunk):   {stats.get('chunk_count', 'N/A')}")
                    print(f"  Relationships:     {stats.get('relationship_count', 'N/A')}")

            except ConnectionError as e:
                 # Handle connection error during exporter init specifically
                 print(f"\nERROR: Could not connect to Neo4j. Please check URI, credentials, and database status.")
                 logger.error("Neo4j connection failed during exporter initialization or operation.")
                 sys.exit(1)
            # No finally block needed for exporter.close() when using 'with' statement

        else:
            print(f"ERROR: Input JSON file not found at {file_path}")
            logger.warning("No JSON file found at %s", file_path)
            sys.exit(1)

    except configparser.Error as e:
         print(f"ERROR reading configuration file: {e}")
         logger.error("Failed to read configuration file.", exc_info=True)
         sys.exit(1)
    except Exception as e:
        print(f"ERROR: An unexpected error occurred in the main script: {e}")
        logger.exception("An unexpected error occurred in the main script.")
        sys.exit(1)

    print("\n--- Neo4j Exporter Script End ---")
