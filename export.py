import argparse
import os
import numpy as np
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

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
LEADERBOARD_IDS = [463, 430, 399, 398]


def fetch_leaderboards(engine, leaderboard_ids) -> Dataset:
    """
    Fetches and processes leaderboard data from the database.

    This function queries the database for specific leaderboards, selecting
    key fields and fetching all associated GPU types for each leaderboard
    using a subquery.

    Args:
        engine: The SQLAlchemy engine instance for database connection.
        leaderboard_ids: A list of integer IDs for the leaderboards to fetch.

    Returns:
        A Hugging Face `Dataset` object containing the leaderboard data.
    """
    print("Fetching leaderboards...")
    query = text("""
        SELECT
            id,
            name,
            deadline,
            task->>'lang' AS lang,
            task->>'description' AS description,
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
    return Dataset.from_pandas(df)


def fetch_submissions(engine, leaderboard_ids) -> Dataset:
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

    Returns:
        A Hugging Face `Dataset` object containing the submissions data.
    """
    print("Fetching submissions...")
    query = text("""
        SELECT
            s.id AS submission_id,
            s.leaderboard_id,
            s.user_id,
            s.submission_time,
            s.file_name,
            c.code,
            c.id AS code_id,
            r.id AS run_id,
            r.start_time AS run_start_time,
            r.end_time AS run_end_time,
            r.mode AS run_mode,
            r.score AS run_score,
            r.passed AS run_passed,
            r.result AS run_result,
            r.compilation as run_compilation,
            r.meta as run_meta,
            r.system_info AS run_system_info
        FROM leaderboard.submission AS s
        JOIN leaderboard.runs AS r ON s.id = r.submission_id
        JOIN leaderboard.code_files AS c ON s.code_id = c.id
        WHERE s.leaderboard_id = ANY(:leaderboard_ids)
    """)
    df = pd.read_sql_query(query, engine, params={'leaderboard_ids': leaderboard_ids})
    return Dataset.from_pandas(df)


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


def main(output_dir):
    """
    Orchestrates the data export process.

    This function initializes the database connection, fetches leaderboard
    and submission data, anonymizes user IDs, and saves the results to
    separate Parquet files: `leaderboards.parquet`, `submissions.parquet`,
    and `successful_submissions.parquet`. The user ID mapping is not saved.

    Args:
        output_dir (str): The local directory path to save the Parquet files.
    """
    engine = create_engine(DATABASE_URL)
    rng = np.random.default_rng()

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Fetch and save leaderboards
    leaderboards_dataset = fetch_leaderboards(engine, LEADERBOARD_IDS)
    leaderboards_output_path = os.path.join(output_dir, "leaderboards.parquet")
    leaderboards_dataset.to_parquet(leaderboards_output_path)
    print(f"Leaderboards dataset successfully saved to {leaderboards_output_path}")

    # Fetch submissions
    submissions_dataset = fetch_submissions(engine, LEADERBOARD_IDS)
    submissions_df = submissions_dataset.to_pandas()

    # Decode hexadecimal 'code' values
    if 'code' in submissions_df.columns:
        print("Decoding 'code' column from hexadecimal where necessary...")
        submissions_df['code'] = submissions_df['code'].apply(decode_hex_if_needed)

    # Anonymize user IDs if submissions exist
    if not submissions_df.empty and 'user_id' in submissions_df.columns:
        print("Anonymizing user IDs...")
        unique_user_ids = submissions_df['user_id'].unique()
        num_unique_users = len(unique_user_ids)

        # Create a randomly permuted mapping in memory
        permuted_ids = rng.permutation(range(1, num_unique_users + 1))
        user_map_df = pd.DataFrame({
            'original_user_id': unique_user_ids,
            'anonymized_user_id': permuted_ids
        })

        # Replace original user IDs with anonymized IDs
        original_cols = list(submissions_df.columns)
        user_id_index = original_cols.index('user_id')
        
        submissions_df = submissions_df.merge(user_map_df, left_on='user_id', right_on='original_user_id')
        submissions_df = submissions_df.drop(columns=['user_id', 'original_user_id'])
        submissions_df = submissions_df.rename(columns={'anonymized_user_id': 'user_id'})

        # Restore original column order
        new_order = [col for col in original_cols if col != 'user_id']
        new_order.insert(user_id_index, 'user_id')
        submissions_df = submissions_df[new_order]

        # Convert back to a dataset
        submissions_dataset = Dataset.from_pandas(submissions_df)

    # Save the submissions dataset (anonymized or original if empty)
    submissions_output_path = os.path.join(output_dir, "submissions.parquet")
    submissions_dataset.to_parquet(submissions_output_path)
    print(f"Submissions dataset successfully saved to {submissions_output_path}")

    # Filter for and save successful submissions from the anonymized data
    if 'run_passed' in submissions_df.columns:
        print("Creating successful submissions dataset...")
        successful_submissions_df = submissions_df[submissions_df['run_passed'] == True].copy()

        # Convert to dataset and save
        successful_submissions_dataset = Dataset.from_pandas(successful_submissions_df)
        successful_output_path = os.path.join(
            output_dir, "successful_submissions.parquet"
        )
        successful_submissions_dataset.to_parquet(successful_output_path)
        print(
            "Successful submissions dataset successfully saved to "
            f"{successful_output_path}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export leaderboard data to a Hugging Face dataset.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset",
        help="Directory to save the Hugging Face dataset."
    )
    args = parser.parse_args()
    main(args.output_dir) 