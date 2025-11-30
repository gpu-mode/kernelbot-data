import argparse
import gc
import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import pyarrow as pa
import pyarrow.parquet as pq
import glob
from dedup import deduplicate_df

load_dotenv()

DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")

if not DB_USER or not DB_PASSWORD:
    raise ValueError(
        "DB_USER and DB_PASSWORD environment variables must be set. "
        "Please create a .env file or export them in your shell."
    )

DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "postgres")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# The leaderboard IDs to export
LEADERBOARD_IDS = [463, 430, 399, 398, 563, 564, 565]


def fetch_and_save_leaderboards(engine, leaderboard_ids, output_path):
    """
    Fetches leaderboard data from the database and saves it directly to parquet.

    This function queries the database for specific leaderboards, selecting
    key fields and fetching all associated GPU types for each leaderboard
    using a subquery. It saves the leaderboards directly to parquet.

    Args:
        engine: The SQLAlchemy engine instance for database connection.
        leaderboard_ids: A list of integer IDs for the leaderboards to fetch.

    Returns:
        The number of leaderboards.
    """
    print("Fetching and saving leaderboards...")
    query = text("""
        SELECT
            id,
            name,
            deadline AT TIME ZONE 'UTC' AS deadline,
            task->>'lang' AS lang,
            description,
            task->'files'->>'reference.py' AS reference,
            (
                SELECT array_agg(gpu_type)
                FROM leaderboard.gpu_type
                WHERE leaderboard_id = leaderboard.leaderboard.id
            ) AS gpu_types
        FROM leaderboard.leaderboard
        WHERE id = ANY(:leaderboard_ids)
    """)
    df = pd.read_sql_query(query, engine, params={'leaderboard_ids': leaderboard_ids})
    df.to_parquet(output_path, index=False)
    print(f"Leaderboards saved to {output_path}")


def anonymize_users_in_db(engine, leaderboard_ids):
    """Create a temporary mapping table in the database."""
    with engine.begin() as conn:
        # Create temporary table with anonymized IDs
        conn.execute(text("""
            CREATE TEMP TABLE user_mapping AS
            SELECT
                user_id as original_user_id,
                ROW_NUMBER() OVER (ORDER BY RANDOM()) as anonymized_user_id
            FROM (
                SELECT DISTINCT user_id
                FROM leaderboard.submission
                WHERE leaderboard_id = ANY(:leaderboard_ids)
            ) AS distinct_users
        """), {'leaderboard_ids': leaderboard_ids})


def handle_empty_structs(df):
    """
    Replace empty struct/dict values with None to avoid PyArrow serialization errors.

    PyArrow cannot write empty struct types to Parquet. This function checks
    columns that contain dict/struct values and replaces empty ones with None.
    """
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if column contains dict-like objects
            sample = df[col].dropna().head(1)
            if len(sample) > 0 and isinstance(sample.iloc[0], dict):
                # Replace empty dicts with None
                df[col] = df[col].apply(lambda x: None if isinstance(x, dict) and len(x) == 0 else x)
    return df


def fetch_and_save_submissions(engine, leaderboard_ids, output_path, chunksize=8192):
    """
    Fetches and processes submission data from the database.

    This function queries the database for submissions associated with specific
    leaderboards. It performs a join across `submission`, `runs`, and
    `code_files` tables to create a denormalized view of the submission,
    its execution details, and the source code.

    Args:
        engine: The SQLAlchemy engine instance for database connection.
        leaderboard_ids: A list of integer IDs for the leaderboards whose
            submissions are to be fetched.
    """
    print("Fetching submissions...")
    query = """
        SELECT
            s.id AS submission_id,
            s.leaderboard_id,
            um.anonymized_user_id AS user_id,
            s.submission_time AT TIME ZONE 'UTC' AS submission_time,
            s.file_name,
            c.code,
            c.id AS code_id,
            r.id AS run_id,
            r.start_time AT TIME ZONE 'UTC' AS run_start_time,
            r.end_time AT TIME ZONE 'UTC' AS run_end_time,
            r.mode AS run_mode,
            r.score AS run_score,
            r.passed AS run_passed,
            r.result AS run_result,
            r.compilation as run_compilation,
            r.meta as run_meta,
            r.system_info AS run_system_info
        FROM leaderboard.submission AS s
        LEFT JOIN leaderboard.runs AS r ON s.id = r.submission_id
        JOIN leaderboard.code_files AS c ON s.code_id = c.id
        LEFT JOIN user_mapping um ON s.user_id = um.original_user_id
        WHERE s.leaderboard_id = ANY(:leaderboard_ids)
    """

    part = 0

    with engine.connect().execution_options(stream_results=True) as conn:
        for chunk_df in pd.read_sql_query(
            text(query),
            conn,
            params={'leaderboard_ids': leaderboard_ids},
            chunksize=chunksize
        ):
            # Decode hex values code column
            if 'code' in chunk_df.columns:
                chunk_df['code'] = chunk_df['code'].apply(decode_hex_if_needed)

            # Convert nullable integer columns to consistent types
            # This prevents type mismatches when some chunks have all NULLs
            nullable_int_cols = ['run_id', 'code_id', 'submission_id', 'leaderboard_id', 'user_id']
            for col in nullable_int_cols:
                if col in chunk_df.columns:
                    chunk_df[col] = chunk_df[col].astype('Int64')

            # Handle empty structs that PyArrow can't serialize
            chunk_df = handle_empty_structs(chunk_df)

            # Convert to arrow table
            table = pa.Table.from_pandas(chunk_df)

            # Write chunk as separate parquet file
            filename = os.path.join(output_path, f"submissions_part_{part:05d}.parquet")
            pq.write_table(table, filename)

            print(f"  Wrote {len(chunk_df)} submissions to part {part}")

            # Filter for and save successful submissions
            if 'run_passed' in chunk_df.columns:
                success_mask = chunk_df['run_passed'] == True
                if success_mask.any():
                    success_df = chunk_df[success_mask]
                    success_table = pa.Table.from_pandas(success_df)
                    success_filename = os.path.join(output_path, f"successful_submissions_part_{part:05d}.parquet")
                    pq.write_table(success_table, success_filename)
                    print(f"  Wrote {len(success_df)} successful submissions to part {part}")
                    del success_df, success_table, success_mask

            del chunk_df, table
            gc.collect()

            part += 1

    print(f"Submissions saved to {part} parquet files in {output_path}")


def decode_hex_if_needed(code_val: str) -> str:
    """Decodes a string from hexadecimal if it starts with '\\x'.

    Args:
        code_val: The value from the 'code' column, expected to be a string.

    Returns:
        The decoded UTF-8 string, or the original value if it's not a
        hex-encoded string or if decoding fails.
    """
    if isinstance(code_val, str) and code_val.startswith('\\x'):
        try:
            # Strip the '\\x' prefix and convert from hex to bytes
            hex_string = code_val[2:]
            byte_data = bytes.fromhex(hex_string)
            # Decode bytes to a UTF-8 string, replacing errors
            return byte_data.decode('utf-8', 'replace')
        except ValueError:
            # Handles errors like non-hex characters in the string
            return code_val
    return code_val


def consolidate_parquet_files(input_dir, pattern, output_file):
    """
    Consolidates multiple parquet part files into a single parquet file.

    Args:
        input_dir: Directory containing the parquet part files
        pattern: Glob pattern to match the part files (e.g., "submissions_part_*.parquet")
        output_file: Path to the output consolidated parquet file
        skip_deduplication: Whether to skip deduplication step (default: False)
    """

    # Find all matching parquet files
    part_files = sorted(glob.glob(os.path.join(input_dir, pattern)))

    if not part_files:
        print(f"  No files found matching pattern {pattern}")
        return

    print(f"  Consolidating {len(part_files)} {pattern} files into {output_file}...")

    # First pass: Read only schemas (not data) from all files to unify them
    schemas = []
    for part_file in part_files:
        parquet_file = pq.ParquetFile(part_file)
        schemas.append(parquet_file.schema_arrow)

    # Unify schemas across all tables to handle struct field variations
    unified_schema = pa.unify_schemas(schemas)

    # Second pass: Read each file, cast to unified schema, and write incrementally
    total_rows = 0
    with pq.ParquetWriter(output_file, unified_schema) as writer:
        for part_file in part_files:
            # Read one file at a time
            table = pq.read_table(part_file)

            # Cast to unified schema (fills missing fields with nulls)
            unified_table = table.cast(unified_schema)

            # Write to output file
            writer.write_table(unified_table)

            total_rows += len(unified_table)

    print(f"  Done! Consolidated {len(part_files)} files ({total_rows} total rows)")

def deduplicate_parquet_file(input_file, output_file):
    """
    Deduplicates a parquet file using the dedup.py script.
    """
    
    # load the parquet file into a pandas dataframe
    df = pd.read_parquet(input_file)
    
    # deduplicate the dataframe
    deduplicated_df = deduplicate_df(df)
    
    # save the deduplicated dataframe to a new parquet file
    deduplicated_df.to_parquet(output_file)

def main(output_dir, skip_deduplication):
    """
    Orchestrates the data export process.

    This function initializes the database connection, fetches leaderboard
    and submission data, anonymizes user IDs, and saves the results to
    separate Parquet files: `leaderboards.parquet`, `submissions.parquet`,
    and `successful_submissions.parquet`. The user ID mapping is not saved.
    Temporary files are not deleted and should be manually removed if
    desired.

    Args:
        output_dir (str): The local directory path to save the Parquet files.
    """
    engine = create_engine(DATABASE_URL)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Fetch and save leaderboards
    leaderboards_output_path = os.path.join(output_dir, "leaderboards.parquet")
    fetch_and_save_leaderboards(engine, LEADERBOARD_IDS, leaderboards_output_path)

    anonymize_users_in_db(engine, LEADERBOARD_IDS)

    # Fetch submissions
    submissions_output_path = os.path.join(output_dir, "submissions")
    os.makedirs(submissions_output_path, exist_ok=True)
    fetch_and_save_submissions(engine, LEADERBOARD_IDS, submissions_output_path)

    # Consolidate part files into single parquet files
    consolidate_parquet_files(
        submissions_output_path,
        "submissions_part_*.parquet",
        os.path.join(output_dir, "submissions.parquet")
    )

    consolidate_parquet_files(
        submissions_output_path,
        "successful_submissions_part_*.parquet",
        os.path.join(output_dir, "successful_submissions.parquet")
    )

    if not skip_deduplication:
        deduplicated_submissions_output_path = os.path.join(output_dir, "deduplicated_submissions")
        deduplicated_successful_submissions_output_path = os.path.join(output_dir, "deduplicated_successful_submissions")
        os.makedirs(deduplicated_submissions_output_path, exist_ok=True)
        # we do this as everything combined can be too much for pandas to handle
        # if things get too big I'd multiprocess this
        for file in glob.glob(os.path.join(output_dir, "submissions_part_*.parquet")):
            deduplicate_parquet_file(file, os.path.join(deduplicated_submissions_output_path, os.path.basename(file)))
        for file in glob.glob(os.path.join(output_dir, "successful_submissions_part_*.parquet")):
            deduplicate_parquet_file(file, os.path.join(deduplicated_successful_submissions_output_path, os.path.basename(file)))
        consolidate_parquet_files(
            deduplicated_submissions_output_path,
            "submissions_part_*.parquet",
            os.path.join(output_dir, "deduplicated_submissions.parquet")
        )
        consolidate_parquet_files(
            deduplicated_successful_submissions_output_path,
            "successful_submissions_part_*.parquet",
            os.path.join(output_dir, "deduplicated_successful_submissions.parquet")
        )
        original_submission_rows = pd.read_parquet(os.path.join(output_dir, "submissions.parquet")).shape[0]
        deduplicated_submission_rows = pd.read_parquet(os.path.join(output_dir, "deduplicated_submissions.parquet")).shape[0]
        original_successful_submission_rows = pd.read_parquet(os.path.join(output_dir, "successful_submissions.parquet")).shape[0]
        deduplicated_successful_submission_rows = pd.read_parquet(os.path.join(output_dir, "deduplicated_successful_submissions.parquet")).shape[0]

        print("Deduplication results Summary:")
        print(f"Original submissions rows: {original_submission_rows}")
        print(f"Deduplicated submissions rows: {deduplicated_submission_rows}")
        print(f"Removed {original_submission_rows - deduplicated_submission_rows} duplicates ({100 * (original_submission_rows - deduplicated_submission_rows) / original_submission_rows:.1f}%)")
        print(f"Original successful submissions rows: {original_successful_submission_rows}")
        print(f"Deduplicated successful submissions rows: {deduplicated_successful_submission_rows}")
        print(f"Removed {original_successful_submission_rows - deduplicated_successful_submission_rows} duplicates ({100 * (original_successful_submission_rows - deduplicated_successful_submission_rows) / original_successful_submission_rows:.1f}%)")
    else:
        print("Skipping deduplication step")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export leaderboard data to a Hugging Face dataset.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset",
        help="Directory to save the Hugging Face dataset."
    )
    parser.add_argument(
        "--skip_deduplication",
        action="store_true",
        help="Skip deduplication step"
    )
    args = parser.parse_args()
    main(args.output_dir, args.skip_deduplication) 