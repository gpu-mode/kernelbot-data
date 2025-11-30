"""
Deduplication Pipeline for Code Submissions
============================================

This module removes duplicate code submissions using a two-stage approach:

1. EXACT DEDUPLICATION (remove_duplicates)
   - Computes SHA-256 hash of each code submission
   - Groups submissions by run_mode, run_passed, and score/duration
   - Within each group, keeps only unique code (by hash)
   - When duplicates exist, keeps the one with better metrics (lower score or faster duration)

2. FUZZY DEDUPLICATION (fuzzy_filter)
   - Uses MinHash + Locality Sensitive Hashing (LSH) to find near-duplicates
   - Process:
     a) Convert each code submission to a set of character n-grams (default: 5-char)
     b) Create MinHash signature for each submission (compact fingerprint)
     c) Use LSH to efficiently find candidate pairs with high Jaccard similarity
     d) Group similar submissions into clusters
     e) Keep one representative from each cluster (highest submission ID)

Usage:

   In practice this should be part of export.py, but if
   you need to run things adhoc just do:

   python dedup.py input.parquet output.parquet

"""

from datasets import load_dataset
import tqdm
from collections import defaultdict
import hashlib
from typing import Dict, List, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import datasketch
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import os

# =============================================================================
# DEDUPLICATION CONFIGURATION CONSTANTS
# =============================================================================

# Fuzzy Deduplication Parameters
FUZZY_SIMILARITY_THRESHOLD = 0.8
"""
Jaccard similarity threshold for considering two documents as duplicates.
Range: 0.0 to 1.0
- 0.8 = High threshold, only very similar documents are considered duplicates
- 0.7 = Medium threshold, moderately similar documents are duplicates  
- 0.5 = Low threshold, loosely similar documents are duplicates
Higher values = more strict deduplication, fewer items removed
"""

NGRAM_SIZE = 5
"""
Size of character n-grams used for MinHash fingerprinting.
- Smaller values (3-4): More sensitive to small changes, better for short text
- Larger values (5-7): Less sensitive to minor variations, better for longer text
- Too small: May create false positives (different texts seem similar)
- Too large: May miss actual duplicates with small variations
"""

LSH_BANDS = 16
"""
Number of bands for Locality Sensitive Hashing (LSH).
Used to speed up similarity detection by grouping similar hashes.
- More bands = faster but less accurate similarity detection
- Fewer bands = slower but more accurate similarity detection
Must divide evenly into ROWS_PER_BAND * LSH_BANDS = total permutations
"""

ROWS_PER_BAND = 128
"""
Number of rows per band in LSH configuration.
Total MinHash permutations = ROWS_PER_BAND * LSH_BANDS
- More rows per band = higher precision, may miss some similar pairs
- Fewer rows per band = higher recall, may include more false positives
Default: 128 rows × 16 bands = 2048 total permutations
"""

# Score Processing Parameters
LEADERBOARD_SCORE_PRECISION = 4
"""
Number of decimal places to round leaderboard scores when grouping submissions.
Used to group submissions with very similar scores together.
- Higher precision (more decimal places): More granular grouping
- Lower precision (fewer decimal places): Broader grouping of similar scores
"""

DURATION_PRECISION = 0
"""
Number of decimal places to round execution duration (in seconds).
Used to group submissions with similar execution times.
- 0: Round to nearest second (1.7s → 2s)
- 1: Round to nearest 0.1s (1.73s → 1.7s)
"""

# =============================================================================
# CONFIGURATION SUMMARY
# =============================================================================
"""
Current deduplication configuration:
├─ Similarity Detection: 0.8 threshold (strict)
├─ Text Fingerprinting: 5-character n-grams  
├─ LSH Performance: 16 bands × 128 rows = 2048 permutations
├─ Score Grouping: 4 decimal places for leaderboard scores
└─ Duration Grouping: 0 decimal places for execution times

To adjust deduplication sensitivity:
- Increase FUZZY_SIMILARITY_THRESHOLD (0.8→0.9) for stricter deduplication
- Decrease FUZZY_SIMILARITY_THRESHOLD (0.8→0.7) for more aggressive deduplication  
- Adjust NGRAM_SIZE for different text lengths (3-4 for short, 5-7 for long)
"""

def remove_duplicates(data_dict: Dict[str, Dict[bool, Dict[Union[float, int], List[Dict]]]]):
    """
    Remove exact duplicates from the nested data structure returned by get_sorted_hf_data.

    Args:
        data_dict: Nested dictionary structure from get_sorted_hf_data

    Returns:
        Dictionary with same structure but duplicates removed
    """
    deduplicated_dict = {}

    for run_mode, score_duration_dict in data_dict.items():
        deduplicated_dict[run_mode] = {}

        for run_success, run_success_dict in score_duration_dict.items():
            deduplicated_dict[run_mode][run_success] = {}
            for score_duration, rows in run_success_dict.items():
                # Use a dictionary to track unique entries by their content hash
                unique_entries = {}

                for row in rows:
                    content = row.get('code', "")
                    content_hash = hashlib.sha256(content.encode()).hexdigest()

                    if content_hash not in unique_entries:
                        unique_entries[content_hash] = row
                    else:
                        # If duplicate found, keep the one with better metrics
                        existing_row = unique_entries[content_hash]

                        # For leaderboard mode with successful runs, prefer lower scores / faster times
                        if run_mode == 'leaderboard' and row.get('run_passed') == True:
                            if row.get('run_score', 0) < existing_row.get('run_score', 0):
                                unique_entries[content_hash] = row
                        # For other cases, prefer shorter duration (faster execution)
                        else:
                            existing_duration = existing_row.get('run_meta', {}).get('duration', float('inf'))
                            current_duration = row.get('run_meta', {}).get('duration', float('inf'))
                            if current_duration < existing_duration:
                                unique_entries[content_hash] = row

                deduplicated_dict[run_mode][run_success][score_duration] = list(unique_entries.values())

    return deduplicated_dict


def _create_single_minhash(args: Tuple[str, str, int, int]) -> Tuple[str, datasketch.MinHash]:
    """Create a MinHash for a single document. Used for parallel processing."""
    submission_id, text, ngram_size, num_permutations = args
    minhash = datasketch.MinHash(num_perm=num_permutations)
    text_lower = text.lower()
    text_bytes = text_lower.encode('utf8')

    # Generate n-grams directly as bytes to avoid repeated encoding
    for i in range(len(text_bytes) - ngram_size + 1):
        minhash.update(text_bytes[i:i + ngram_size])

    return submission_id, minhash


def create_minhashes(
    documents: List[Dict[str, str]],
    ngram_size: int = NGRAM_SIZE,
    bands: int = LSH_BANDS,
    rows_per_band: int = ROWS_PER_BAND,
    n_jobs: int = None,
    position: int = 0,
) -> Dict[str, datasketch.MinHash]:
    """
    Create MinHash signatures for a list of documents with LSH bands configuration.

    Args:
        documents: List of dictionaries, each containing 'submission_id' and 'code' keys
        ngram_size: Size of n-grams to generate from input text (default: 5)
        bands: Number of bands for LSH (default: 16)
        rows_per_band: Rows per band for LSH (default: 128)
        n_jobs: Number of parallel workers. Defaults to CPU count.
        position: Position for nested tqdm progress bar

    Returns:
        Dictionary mapping document submission_ids to their MinHash signatures
    """
    num_permutations = rows_per_band * bands

    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()

    # Prepare arguments for parallel processing
    args_list = [
        (doc["submission_id"], doc["code"], ngram_size, num_permutations)
        for doc in documents
    ]

    # Use parallel processing for large datasets
    if len(documents) > 100 and n_jobs > 1:
        minhash_dict = {}
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(_create_single_minhash, args): args[0] for args in args_list}
            for future in tqdm.tqdm(as_completed(futures), total=len(futures),
                                     desc="Creating minhashes", position=position, leave=False):
                submission_id, minhash = future.result()
                minhash_dict[submission_id] = minhash
        return minhash_dict

    # Sequential processing for small datasets
    minhash_dict = {}
    for args in tqdm.tqdm(args_list, desc="Creating minhashes", position=position, leave=False):
        submission_id, minhash = _create_single_minhash(args)
        minhash_dict[submission_id] = minhash

    return minhash_dict


def create_similarity_matrix(
    minhashes: Dict[str, datasketch.MinHash],
    rows_per_band: int,
    num_bands: int,
    threshold: float,
) -> Dict[str, List[str]]:
    """Build LSH index and query for similar documents."""
    lsh = datasketch.MinHashLSH(threshold=threshold, num_perm=num_bands * rows_per_band)

    # Batch insert for better performance
    for submission_id, minhash in minhashes.items():
        lsh.insert(submission_id, minhash)

    # Query all at once
    similarity_matrix = {}
    for submission_id, minhash in minhashes.items():
        similar_ids = lsh.query(minhash)
        # Remove self from results inline
        similarity_matrix[submission_id] = [s for s in similar_ids if s != submission_id]

    return similarity_matrix


def filter_matrix(
    similarity_matrix: Dict[str, List[str]]
) -> set:
    good_submission_ids = set()
    processed = set()
    
    for submission_id, similar_submission_ids in similarity_matrix.items():
        if submission_id in processed:
            continue
            
        # Find all submissions in the similarity cluster
        cluster = {submission_id}
        cluster.update(similar_submission_ids)
        
        # Keep the one with the largest ID (tiebreaker)
        keeper = max(cluster)
        good_submission_ids.add(keeper)
        
        # Mark all in cluster as processed
        processed.update(cluster)
    
    return good_submission_ids


def fuzzy_filter(
    data_dict: Dict[str, Dict[bool, Dict[Union[float, int], List[Dict]]]],
    threshold: float = FUZZY_SIMILARITY_THRESHOLD,
    ngram_size: int = NGRAM_SIZE,
    bands: int = LSH_BANDS,
    rows_per_band: int = ROWS_PER_BAND,
) -> Dict[str, Dict[bool, Dict[Union[float, int], List[Dict]]]]:
    """Apply fuzzy deduplication to the nested data structure."""
    deduped_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Count total groups for progress bar
    total_groups = sum(
        len(score_duration_dict)
        for run_success_dict in data_dict.values()
        for score_duration_dict in run_success_dict.values()
    )

    with tqdm.tqdm(total=total_groups, desc="Fuzzy dedup groups", position=0) as pbar:
        for run_mode, run_success_dict in data_dict.items():
            for run_success, score_duration_dict in run_success_dict.items():
                for score_duration, rows in score_duration_dict.items():
                    pbar.set_postfix({"mode": run_mode, "rows": len(rows)})
                    deduped_data[run_mode][run_success][score_duration] = _fuzzy_filter(
                        rows, threshold, ngram_size, bands, rows_per_band, position=1
                    )
                    pbar.update(1)

    return deduped_data


def _fuzzy_filter(
    data_list: List[Dict],
    threshold: float = FUZZY_SIMILARITY_THRESHOLD,
    ngram_size: int = NGRAM_SIZE,
    bands: int = LSH_BANDS,
    rows_per_band: int = ROWS_PER_BAND,
    position: int = 0,
) -> List[Dict]:
    """
    Apply fuzzy deduplication to a list of documents.

    Args:
        data_list: List of row dictionaries
        threshold: Similarity threshold for LSH
        ngram_size: Size of n-grams for MinHash
        bands: Number of bands for LSH
        rows_per_band: Rows per band for LSH
        position: Position for nested tqdm progress bar

    Returns:
        List with fuzzy duplicates removed
    """
    if len(data_list) <= 1:
        return data_list

    # Build documents list without tqdm overhead
    all_documents = [
        {"submission_id": str(i), "code": row.get('code', str(row)), "original_row": row}
        for i, row in enumerate(data_list)
    ]

    # Apply fuzzy deduplication
    minhashes = create_minhashes(
        all_documents, ngram_size=ngram_size, bands=bands, rows_per_band=rows_per_band,
        position=position
    )
    similarity_matrix = create_similarity_matrix(
        minhashes, rows_per_band=rows_per_band, num_bands=bands, threshold=threshold
    )

    good_submission_ids = filter_matrix(similarity_matrix)

    # Keep only the documents that passed the filter
    return [all_documents[int(sid)]["original_row"] for sid in good_submission_ids]


def convert_df_to_dict(df: pd.DataFrame) -> Dict[str, Dict[bool, Dict[Union[float, int], List[Dict]]]]:
    """
    Convert a pandas DataFrame to a nested dictionary structure.

    Args:
        df: pandas DataFrame

    Returns:
        Nested dictionary structure grouped by run_mode, run_passed, and duration
    """
    # Extract duration from run_meta column (vectorized where possible)
    if 'run_meta' in df.columns:
        durations = df['run_meta'].apply(lambda x: x.get('duration', 0) if isinstance(x, dict) else 0)
    else:
        durations = pd.Series([0] * len(df))

    # Add duration as a column for grouping
    df = df.copy()
    df['_duration'] = durations

    data_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Group by run_mode and run_passed, then iterate groups (much faster than iterrows)
    for (run_mode, run_passed), group in df.groupby(['run_mode', 'run_passed'], sort=False):
        # Convert group to list of dicts at once
        records = group.drop(columns=['_duration']).to_dict('records')
        group_durations = group['_duration'].tolist()

        for record, duration in zip(records, group_durations):
            data_dict[run_mode][run_passed][duration].append(record)

    return data_dict

def flatten_data(data_dict: Dict[str, Dict[Union[float, int], List[Dict]]]) -> List[Dict]:
    """
    Flatten the nested data structure to a list of documents with metadata.

    Args:
        data_dict: Nested dictionary structure from get_sorted_hf_data

    Returns:
        List of documents with additional metadata fields
    """
    flattened = []
    for run_mode, run_success_dict in data_dict.items():
        for run_success, score_duration_dict in run_success_dict.items():
            for score_duration, rows in score_duration_dict.items():
                for row in rows:
                    # Add metadata directly to dict (avoid copy if possible)
                    if isinstance(row, dict):
                        row['_run_mode'] = run_mode
                        row['_run_success'] = run_success
                        row['_score_duration'] = score_duration
                        flattened.append(row)
                    else:
                        # Handle pandas Series
                        row_dict = row.to_dict() if hasattr(row, 'to_dict') else dict(row)
                        row_dict['_run_mode'] = run_mode
                        row_dict['_run_success'] = run_success
                        row_dict['_score_duration'] = score_duration
                        flattened.append(row_dict)
    return flattened

def dedup_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate a pandas DataFrame.
    
    Args:
        df: pandas DataFrame
    """
    # convert to dict
    data_dict = convert_df_to_dict(df)
    # deduplicate
    deduplicated_data = fuzzy_filter(
        data_dict, 
        threshold=FUZZY_SIMILARITY_THRESHOLD, 
        ngram_size=NGRAM_SIZE, 
        bands=LSH_BANDS, 
        rows_per_band=ROWS_PER_BAND
    )
    # convert to df
    flattened_data = flatten_data(deduplicated_data)
    df = pd.DataFrame(flattened_data)
    return df

def create_parquet_file(data_dict: Dict[str, Dict[Union[float, int], List[Dict]]], filename: str):
    """
    Create a Parquet file from the nested data structure.

    Args:
        data_dict: Nested dictionary structure from get_sorted_hf_data
        filename: Name of the output Parquet file
    """
    # Flatten the data
    flattened_data = flatten_data(data_dict)

    # Create a pandas DataFrame from the flattened data
    df = pd.DataFrame(flattened_data)
    # Convert the DataFrame to a Parquet file
    df.to_parquet(filename, index=False)


def _count_items(data_dict: Dict[str, Dict[bool, Dict[Union[float, int], List[Dict]]]]) -> int:
    """
    Count total number of items in the nested data structure. (useful for testing)

    Args:
        data_dict: Nested dictionary structure from get_sorted_hf_data

    Returns:
        Total number of items
    """
    total = 0
    for run_mode in data_dict.values():
        for run_success_dict in run_mode.values():
            for rows in run_success_dict.values():
                total += len(rows)
    return total


# Columns required for deduplication
REQUIRED_COLUMNS = ['code', 'run_mode', 'run_passed', 'run_meta', 'submission_id']


def dedup_file(input_path: str, output_path: str) -> None:
    """
    Deduplicate a parquet file and save the result.

    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file
    """
    # Show file size
    file_size = os.path.getsize(input_path)
    print(f"Loading {input_path} ({file_size / 1e9:.2f} GB)...")

    # Use PyArrow for faster loading, only load required columns
    pf = pq.ParquetFile(input_path)
    available_columns = pf.schema.names
    columns_to_load = [c for c in REQUIRED_COLUMNS if c in available_columns]

    print(f"Loading columns: {columns_to_load}")
    table = pq.read_table(input_path, columns=columns_to_load)
    df = table.to_pandas()
    print(f"Loaded {len(df)} rows")

    # Decode bytes to string if needed
    if 'code' in df.columns and len(df) > 0:
        if isinstance(df['code'].iloc[0], bytes):
            print("Decoding code column from bytes...")
            df['code'] = df['code'].apply(
                lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
            )

    original_count = len(df)

    # Convert to nested dict structure
    print("Converting to nested structure...")
    data_dict = convert_df_to_dict(df)

    # Apply exact deduplication
    print("Applying exact deduplication...")
    exact_deduped = remove_duplicates(data_dict)
    exact_count = _count_items(exact_deduped)


    # Apply fuzzy deduplication
    print("Applying fuzzy deduplication...")
    fuzzy_deduped = fuzzy_filter(
        exact_deduped,
        threshold=FUZZY_SIMILARITY_THRESHOLD,
        ngram_size=NGRAM_SIZE,
        bands=LSH_BANDS,
        rows_per_band=ROWS_PER_BAND
    )

    # Flatten and save
    print("Flattening and saving...")
    flattened = flatten_data(fuzzy_deduped)
    result_df = pd.DataFrame(flattened)
    result_df.to_parquet(output_path, index=False)

    final_count = len(result_df)

    print("Deduplication results Summary:")
    print(f"Original rows: {original_count}")
    print(f"After hash based dedup dedup: {exact_count} rows")
    print(f"Final rows: {final_count}")
    print(f"Removed {original_count - final_count} duplicates ({100 * (original_count - final_count) / original_count:.1f}%)")
    print(f"Saved to {output_path}")


def main():
    import sys

    if len(sys.argv) == 3:
        # File-based deduplication
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        dedup_file(input_path, output_path)
    else:
        print("Usage: python dedup.py <input.parquet> <output.parquet>")
        sys.exit(1)


if __name__ == "__main__":
    main()
