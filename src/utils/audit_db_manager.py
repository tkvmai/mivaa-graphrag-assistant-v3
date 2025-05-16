# audit_db_manager.py
import sqlite3
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

# --- Configuration ---
DB_FILE_PATH = Path("./ingestion_audit.db") # Store DB in the root directory

# --- Logger Setup ---
# Use a specific logger for this module
logger = logging.getLogger(__name__)
# Basic config if run standalone or not configured elsewhere
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s')

# --- Database Schema ---
# Define table creation SQL statements
CREATE_JOBS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS IngestionJobs (
    job_id TEXT PRIMARY KEY,
    start_timestamp TEXT NOT NULL,
    end_timestamp TEXT,
    status TEXT NOT NULL CHECK(status IN ('Queued', 'Running', 'Completed', 'Completed with Errors', 'Failed')),
    total_files_in_job INTEGER NOT NULL,
    submitted_by TEXT -- Optional for future use
);
"""

CREATE_FILES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ProcessedFiles (
    file_processing_id TEXT PRIMARY KEY,
    ingestion_job_id TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_size_bytes INTEGER,
    file_type TEXT,
    file_hash TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('Processing', 'Success', 'Failed - OCR', 'Failed - KG Extract', 'Failed - Neo4j', 'Failed - Embedding', 'Failed - Chunking', 'Failed - Unknown', 'Cached')),
    processing_start_timestamp TEXT NOT NULL,
    processing_end_timestamp TEXT,
    error_message TEXT,
    log_messages TEXT, -- Consider if needed, could become large
    text_extracted BOOLEAN,
    num_chunks_generated INTEGER,
    num_triples_extracted INTEGER,
    num_triples_loaded_neo4j INTEGER,
    num_vectors_stored_chroma INTEGER,
    cache_hit BOOLEAN,
    FOREIGN KEY (ingestion_job_id) REFERENCES IngestionJobs (job_id)
);
"""

# --- Helper Function ---
def _get_db_connection() -> sqlite3.Connection:
    """Establishes and returns a database connection."""
    try:
        conn = sqlite3.connect(DB_FILE_PATH, check_same_thread=False) # Allow access from multiple threads (important for background tasks)
        conn.row_factory = sqlite3.Row # Return rows as dictionary-like objects
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error to {DB_FILE_PATH}: {e}", exc_info=True)
        raise

# --- Core Functions ---

def initialize_database():
    """Creates the database file and tables if they don't exist."""
    logger.info(f"Initializing audit database at {DB_FILE_PATH}...")
    try:
        with _get_db_connection() as conn:
            cursor = conn.cursor()
            logger.debug("Executing CREATE TABLE for IngestionJobs if not exists.")
            cursor.execute(CREATE_JOBS_TABLE_SQL)
            logger.debug("Executing CREATE TABLE for ProcessedFiles if not exists.")
            cursor.execute(CREATE_FILES_TABLE_SQL)
            # You could add index creation here for performance later if needed
            # e.g., cursor.execute("CREATE INDEX IF NOT EXISTS idx_job_id ON ProcessedFiles (ingestion_job_id);")
            conn.commit()
        logger.info("Audit database initialization complete.")
    except sqlite3.Error as e:
        logger.error(f"Failed to initialize database schema: {e}", exc_info=True)
        raise

def create_ingestion_job(total_files: int, submitted_by: Optional[str] = None) -> Optional[str]:
    """
    Creates a new ingestion job record and returns the job ID.
    Initial status is 'Running'.
    """
    job_id = uuid.uuid4().hex
    start_time = datetime.now(timezone.utc).isoformat()
    status = 'Running' # Start immediately in 'Running' state
    logger.info(f"Creating new ingestion job (ID: {job_id}) for {total_files} files.")

    sql = """
    INSERT INTO IngestionJobs (job_id, start_timestamp, status, total_files_in_job, submitted_by)
    VALUES (?, ?, ?, ?, ?);
    """
    try:
        with _get_db_connection() as conn:
            conn.execute(sql, (job_id, start_time, status, total_files, submitted_by))
            conn.commit()
        logger.info(f"Successfully created job {job_id}.")
        return job_id
    except sqlite3.Error as e:
        logger.error(f"Failed to create ingestion job record for job {job_id}: {e}", exc_info=True)
        return None

def start_file_processing(job_id: str, file_name: str, file_size: int, file_type: str, file_hash: str) -> Optional[str]:
    """
    Creates a record for a file being processed within a job.
    Initial status is 'Processing'. Returns the file processing ID.
    """
    file_processing_id = uuid.uuid4().hex
    start_time = datetime.now(timezone.utc).isoformat()
    status = 'Processing'
    logger.info(f"Starting processing for file '{file_name}' (File ID: {file_processing_id}, Job ID: {job_id})")

    sql = """
    INSERT INTO ProcessedFiles (file_processing_id, ingestion_job_id, file_name, file_size_bytes, file_type, file_hash, status, processing_start_timestamp, cache_hit)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    try:
        with _get_db_connection() as conn:
            # Initialize cache_hit as False
            conn.execute(sql, (file_processing_id, job_id, file_name, file_size, file_type, file_hash, status, start_time, False))
            conn.commit()
        logger.debug(f"Successfully created initial record for file {file_processing_id}.")
        return file_processing_id
    except sqlite3.Error as e:
        logger.error(f"Failed to create file processing record for '{file_name}' (Job: {job_id}): {e}", exc_info=True)
        return None

def update_file_status(
    file_processing_id: str,
    status: str,
    end_timestamp: Optional[str] = None,
    error_message: Optional[str] = None,
    log_messages: Optional[str] = None,
    text_extracted: Optional[bool] = None,
    num_chunks: Optional[int] = None,
    num_triples_extracted: Optional[int] = None,
    num_triples_loaded: Optional[int] = None,
    num_vectors_stored: Optional[int] = None,
    cache_hit: Optional[bool] = None
):
    """Updates the status and details of a processed file record."""
    if not end_timestamp:
        end_timestamp = datetime.now(timezone.utc).isoformat()

    # Build the SET part of the SQL query dynamically based on provided arguments
    set_clauses = []
    params = []

    set_clauses.append("status = ?")
    params.append(status)
    set_clauses.append("processing_end_timestamp = ?")
    params.append(end_timestamp)

    if error_message is not None:
        set_clauses.append("error_message = ?")
        params.append(error_message)
    if log_messages is not None:
        set_clauses.append("log_messages = ?")
        params.append(log_messages)
    if text_extracted is not None:
        set_clauses.append("text_extracted = ?")
        params.append(int(text_extracted)) # Store boolean as 0 or 1
    if num_chunks is not None:
        set_clauses.append("num_chunks_generated = ?")
        params.append(num_chunks)
    if num_triples_extracted is not None:
        set_clauses.append("num_triples_extracted = ?")
        params.append(num_triples_extracted)
    if num_triples_loaded is not None:
        set_clauses.append("num_triples_loaded_neo4j = ?")
        params.append(num_triples_loaded)
    if num_vectors_stored is not None:
        set_clauses.append("num_vectors_stored_chroma = ?")
        params.append(num_vectors_stored)
    if cache_hit is not None:
        set_clauses.append("cache_hit = ?")
        params.append(int(cache_hit)) # Store boolean as 0 or 1

    params.append(file_processing_id) # For the WHERE clause

    sql = f"UPDATE ProcessedFiles SET {', '.join(set_clauses)} WHERE file_processing_id = ?;"

    logger.info(f"Updating status for file {file_processing_id} to '{status}'.")
    logger.debug(f"Update SQL: {sql} | Params: {params}")

    try:
        with _get_db_connection() as conn:
            conn.execute(sql, tuple(params))
            conn.commit()
        logger.debug(f"Successfully updated record for file {file_processing_id}.")
        return True
    except sqlite3.Error as e:
        logger.error(f"Failed to update file processing record for {file_processing_id}: {e}", exc_info=True)
        return False


def update_job_status(job_id: str, status: str):
    """Updates the status and end timestamp of an ingestion job."""
    end_time = datetime.now(timezone.utc).isoformat()
    sql = "UPDATE IngestionJobs SET status = ?, end_timestamp = ? WHERE job_id = ?;"
    logger.info(f"Updating status for job {job_id} to '{status}'.")
    try:
        with _get_db_connection() as conn:
            conn.execute(sql, (status, end_time, job_id))
            conn.commit()
        logger.info(f"Successfully updated job {job_id} status.")
        return True
    except sqlite3.Error as e:
        logger.error(f"Failed to update ingestion job record for {job_id}: {e}", exc_info=True)
        return False

def get_recent_jobs(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves a summary of the most recent ingestion jobs."""
    sql = "SELECT job_id, start_timestamp, end_timestamp, status, total_files_in_job FROM IngestionJobs ORDER BY start_timestamp DESC LIMIT ?;"
    logger.debug(f"Retrieving recent {limit} ingestion jobs.")
    try:
        with _get_db_connection() as conn:
            cursor = conn.execute(sql, (limit,))
            jobs = [dict(row) for row in cursor.fetchall()] # Convert rows to dicts
        logger.debug(f"Retrieved {len(jobs)} job summaries.")
        return jobs
    except sqlite3.Error as e:
        logger.error(f"Failed to retrieve recent jobs: {e}", exc_info=True)
        return []

def get_job_details(job_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves the job record and all associated file records for a specific job."""
    job_sql = "SELECT job_id, start_timestamp, end_timestamp, status, total_files_in_job FROM IngestionJobs WHERE job_id = ?;"
    files_sql = """
    SELECT
        file_processing_id, file_name, file_size_bytes, file_type, file_hash, status,
        processing_start_timestamp, processing_end_timestamp, error_message,
        text_extracted, num_chunks_generated, num_triples_extracted,
        num_triples_loaded_neo4j, num_vectors_stored_chroma, cache_hit
    FROM ProcessedFiles
    WHERE ingestion_job_id = ?
    ORDER BY file_name ASC;
    """
    logger.debug(f"Retrieving details for job {job_id}.")
    job_details: Optional[Dict[str, Any]] = None
    try:
        with _get_db_connection() as conn:
            job_cursor = conn.execute(job_sql, (job_id,))
            job_row = job_cursor.fetchone()
            if job_row:
                job_details = dict(job_row)
                logger.debug(f"Retrieved job info for {job_id}.")
                files_cursor = conn.execute(files_sql, (job_id,))
                # Convert boolean 0/1 back to True/False for easier use
                file_records = []
                for row in files_cursor.fetchall():
                    file_dict = dict(row)
                    if file_dict.get('text_extracted') is not None:
                        file_dict['text_extracted'] = bool(file_dict['text_extracted'])
                    if file_dict.get('cache_hit') is not None:
                        file_dict['cache_hit'] = bool(file_dict['cache_hit'])
                    file_records.append(file_dict)

                job_details['processed_files'] = file_records
                logger.debug(f"Retrieved {len(file_records)} associated file records for job {job_id}.")
            else:
                logger.warning(f"Job ID {job_id} not found.")
        return job_details
    except sqlite3.Error as e:
        logger.error(f"Failed to retrieve details for job {job_id}: {e}", exc_info=True)
        return None

# --- Optional: Main block for testing ---
if __name__ == "__main__":
    print("Running Audit DB Manager setup and basic tests...")
    initialize_database()

    # Example usage:
    print("\nCreating Job 1...")
    job1_id = create_ingestion_job(total_files=2)
    if job1_id:
        print(f"Job 1 ID: {job1_id}")
        file1_id = start_file_processing(job1_id, "doc1.pdf", 1024, "application/pdf", "hash1")
        file2_id = start_file_processing(job1_id, "doc2.txt", 512, "text/plain", "hash2")
        print(f"File 1 ID: {file1_id}")
        print(f"File 2 ID: {file2_id}")

        if file1_id:
            # Simulate processing file 1 (Success)
            update_file_status(
                file_processing_id=file1_id, status='Success', text_extracted=True,
                num_chunks=10, num_triples_extracted=50, num_triples_loaded=48,
                num_vectors_stored=10, cache_hit=False
            )
            print("Updated File 1 status to Success.")

        if file2_id:
             # Simulate processing file 2 (Failure)
             update_file_status(
                 file_processing_id=file2_id, status='Failed - KG Extract',
                 error_message="LLM Quota Exceeded", text_extracted=True, num_chunks=5
             )
             print("Updated File 2 status to Failed.")

        # Finish Job 1
        update_job_status(job1_id, status='Completed with Errors')
        print("Updated Job 1 status.")

    print("\nCreating Job 2 (Cached)...")
    job2_id = create_ingestion_job(total_files=1)
    if job2_id:
        file3_id = start_file_processing(job2_id, "doc1_cached.pdf", 1024, "application/pdf", "hash1") # Same hash
        if file3_id:
            update_file_status(file_processing_id=file3_id, status='Cached', cache_hit=True)
            print("Updated File 3 status to Cached.")
        update_job_status(job2_id, status='Completed')
        print("Updated Job 2 status.")

    print("\nRecent Jobs:")
    recent_jobs = get_recent_jobs(limit=5)
    for job in recent_jobs:
        print(f"- Job ID: {job['job_id']}, Status: {job['status']}, Start: {job['start_timestamp']}")

    print("\nDetails for Job 1:")
    job1_details = get_job_details(job1_id)
    if job1_details:
        print(f"  Job Status: {job1_details['status']}")
        for file_rec in job1_details.get('processed_files', []):
             print(f"  - File: {file_rec['file_name']}, Status: {file_rec['status']}, Triples: {file_rec['num_triples_extracted']}, Vectors: {file_rec['num_vectors_stored_chroma']}, Error: {file_rec.get('error_message')}")

    print("\nAudit DB Manager tests finished.")