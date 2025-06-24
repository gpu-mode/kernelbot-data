# script to dedup a huggingface dataset

from datasets import load_dataset
import tqdm
from collections import defaultdict
import hashlib
from typing import Dict, List, Tuple, Union

import datasketch
import pandas as pd

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
    
    for run_mode, score_duration_dict in tqdm.tqdm(data_dict.items(), desc="Processing run modes"):
        deduplicated_dict[run_mode] = {}

        for run_success, run_success_dict in tqdm.tqdm(score_duration_dict.items(), desc=f"Processing {run_mode}", leave=False):
            deduplicated_dict[run_mode][run_success] = {}
            for score_duration, rows in tqdm.tqdm(run_success_dict.items(), desc=f"Processing {run_mode}", leave=False):
                # Use a dictionary to track unique entries by their content hash
                unique_entries = {}
                
                for row in tqdm.tqdm(rows, desc=f"Processing {run_mode} {score_duration}", leave=False):
                    # Create a hash of the relevant content (assuming 'input' or similar field exists)
                    # If the row has an 'input' field, use that; otherwise use the entire row
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


def create_minhashes(
    documents: List[Dict[str, str]],
    ngram_size: int = NGRAM_SIZE,
    bands: int = LSH_BANDS,
    rows_per_band: int = ROWS_PER_BAND,
) -> Tuple[Dict[str, datasketch.MinHash], int]:
    """
    Create MinHash signatures for a list of documents with LSH bands configuration.

    Args:
        documents: List of dictionaries, each containing 'submission_id' and 'input' keys
        num_permutations: Number of hash functions to use (default: 100)
        ngram_size: Size of n-grams to generate from input text (default: 3)
        bands: Number of bands for LSH (default: 20)

    Returns:
        Tuple containing:
        - Dictionary mapping document submission_ids to their MinHash signatures
        - Rows per band (num_permutations / bands)

    Raises:
        ValueError: If num_permutations is not divisible by bands
    """

    num_permutations = rows_per_band * bands

    def generate_ngrams(text: str, n: int) -> List[str]:
        """Generate n-grams from input text."""
        return [text[i : i + n] for i in range(len(text) - n + 1)]

    # Initialize result dictionary
    minhash_dict = {}
    # Process each document
    for doc in tqdm.tqdm(documents, desc="Creating minhashes"):
        minhash = datasketch.MinHash(num_perm=num_permutations)
        submission_id = doc["submission_id"]
        text = doc["code"].lower()  # Convert to lowercase for consistency

        # Generate n-grams
        ngrams = generate_ngrams(text, ngram_size)
        for ngram in ngrams:
            minhash.update(ngram.encode("utf8"))

        minhash_dict[submission_id] = minhash

    return minhash_dict


# 16 bands with 128 rows
def create_similarity_matrix(
    minhashes: Dict[str, datasketch.MinHash],
    rows_per_band: int,
    num_bands: int,
    threshold: float,
) -> Dict[str, List[str]]:
    lsh = datasketch.MinHashLSH(threshold=threshold, num_perm=num_bands * rows_per_band)
    print(f"num_perm: {num_bands*rows_per_band}")
    similarity_matrix = {}
    for submission_id, minhash in tqdm.tqdm(minhashes.items(), desc="Inserting minhashes into LSH"):
        lsh.insert(submission_id, minhash)
    for submission_id, minhash in tqdm.tqdm(minhashes.items(), desc="Querying LSH"):
        similar_submission_ids = lsh.query(minhash)
        similarity_matrix[submission_id] = similar_submission_ids
    for submission_id, similar_submission_ids in tqdm.tqdm(
        similarity_matrix.items(), desc="Removing self-similarities"
    ):
        if submission_id in similar_submission_ids:
            similar_submission_ids.remove(submission_id)
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
    
    total_categories = 0
    for run_mode, run_success_dict in data_dict.items():
        for run_success, score_duration_dict in run_success_dict.items():
            for score_duration, rows in score_duration_dict.items():
                total_categories += 1

    deduped_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    current_category = 0
    for run_mode, run_success_dict in data_dict.items():
        for run_success, score_duration_dict in run_success_dict.items():
            for score_duration, rows in score_duration_dict.items():
                print(f"Processing {run_mode} {run_success} {score_duration} {len(rows)}")
                print(f"This is {current_category} of {total_categories}")
                current_category += 1
                deduped_data[run_mode][run_success][score_duration] = _fuzzy_filter(rows, threshold, ngram_size, bands, rows_per_band)

    return deduped_data

def _fuzzy_filter(
    data_list: List[Dict],
    threshold: float = FUZZY_SIMILARITY_THRESHOLD,
    ngram_size: int = NGRAM_SIZE,
    bands: int = LSH_BANDS,
    rows_per_band: int = ROWS_PER_BAND,
) -> List[Dict]:
    """
    Apply fuzzy deduplication to the nested data structure returned by get_sorted_hf_data.
    
    Args:
        data_dict: Nested dictionary structure from get_sorted_hf_data
        threshold: Similarity threshold for LSH
        ngram_size: Size of n-grams for MinHash
        bands: Number of bands for LSH
        rows_per_band: Rows per band for LSH
        create_histogram: Whether to create similarity histogram
        
    Returns:
        Dictionary with same structure but fuzzy duplicates removed
    """
    # Flatten the data for processing
    
    # Create documents for MinHash processing

    if len(data_list) <= 1:
        return data_list

    all_documents = []
    for i, row in tqdm.tqdm(enumerate(data_list), desc="Creating documents for MinHash"):
        # Use 'input' field if available, otherwise use a string representation
        content = row.get('code', str(row))
        document = {
            "submission_id": str(i),
            "code": content,
            "original_row": row
        }
        all_documents.append(document)
    
    # Apply fuzzy deduplication
    minhashes = create_minhashes(
        all_documents, ngram_size=ngram_size, bands=bands, rows_per_band=rows_per_band
    )
    similarity_matrix = create_similarity_matrix(
        minhashes, rows_per_band=rows_per_band, num_bands=bands, threshold=threshold
    )
    
    good_submission_ids = filter_matrix(similarity_matrix)
    
    # Keep only the documents that passed the filter
    good_documents = [all_documents[int(submission_id)]["original_row"] for submission_id in good_submission_ids]
    
    # Reconstruct the nested structure
    return good_documents

def get_hf_data() -> Dict[str, Dict[Union[float, int], List[Dict]]]:
    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("GPUMODE/kernelbot-data", "submissions")

    # we should divide things up into type
    # run_mode
    # run_sucess
    # if run_mode is leaderboard then use score
    # otherwise use run_meta[duration]


    data = ds['train']

    run_mode_dict = defaultdict(list)
    run_success_dict = defaultdict(lambda: defaultdict(list))
    run_duration_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for _, row in tqdm.tqdm(enumerate(data), desc="Processing dataset rows"):
        run_mode = row['run_mode']
        run_mode_dict[run_mode].append(row)

    for run_mode, rows in tqdm.tqdm(run_mode_dict.items(), desc="Processing run modes"):
        for row in tqdm.tqdm(rows, desc=f"Processing {run_mode} success/failure", leave=False):
            run_success_dict[run_mode][row['run_passed']].append(row)

    for run_mode, mode_dict in tqdm.tqdm(run_success_dict.items(), desc="Processing success/failure groups"):
        for run_success, rows in tqdm.tqdm(mode_dict.items(), desc=f"Processing {run_mode}", leave=False):
            for row in tqdm.tqdm(rows, desc=f"Processing {run_mode} {run_success} rows", leave=False):
                if run_mode == 'leaderboard' and run_success == True:
                    rounded_score = round(float(row['run_score']), LEADERBOARD_SCORE_PRECISION)
                    run_duration_dict[run_mode][run_success][rounded_score].append(row)
                else:
                    rounded_duration = round(float(row['run_meta']['duration']), DURATION_PRECISION)
                    run_duration_dict[run_mode][run_success][rounded_duration].append(row)

    return run_duration_dict

def convert_df_to_dict(df: pd.DataFrame) -> Dict[str, Dict[bool, Dict[Union[float, int], List[Dict]]]]:
    """
    Convert a pandas DataFrame to a nested dictionary structure.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        Nested dictionary structure
    """
    data_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for _, row in tqdm.tqdm(df.iterrows(), desc="Processing DataFrame rows"):
        run_mode = row['run_mode']
        run_success = row['run_passed']
        score_duration = row['run_meta']['duration']
        data_dict[run_mode][run_success][score_duration].append(row)
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
    for run_mode, run_success_dict in tqdm.tqdm(data_dict.items(), desc="Flattening data"):
        for run_success, score_duration_dict in run_success_dict.items():
            for score_duration, rows in score_duration_dict.items():
                for row in tqdm.tqdm(rows, desc=f"Processing {run_mode} {score_duration}", leave=False):
                    # Add metadata to each row
                    row_with_metadata = row.copy()
                    row_with_metadata['_run_mode'] = run_mode
                    row_with_metadata['_run_success'] = run_success
                    row_with_metadata['_score_duration'] = score_duration
                    flattened.append(row_with_metadata)
    return flattened

def count_items(data_dict: Dict[str, Dict[bool, Dict[Union[float, int], List[Dict]]]]) -> int:
    """
    Count total number of items in the nested data structure.
    
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


def example_usage():
    """
    Example of how to use the deduplication functions with get_hf_data output.
    """
    # Load the data
    data = get_hf_data()
    
    print(f"Original data has {count_items(data)} total items")
    
    # Remove exact duplicates
    deduplicated_data = remove_duplicates(data)
    print(f"After exact deduplication: {count_items(deduplicated_data)} items")
    
    # Apply fuzzy deduplication
    fuzzy_deduplicated_data = fuzzy_filter(
        deduplicated_data,
        threshold=FUZZY_SIMILARITY_THRESHOLD,
        ngram_size=NGRAM_SIZE,
        bands=LSH_BANDS,
        rows_per_band=ROWS_PER_BAND
    )
    # convert to df
    flattened_data = flatten_data(fuzzy_deduplicated_data)
    df = pd.DataFrame(flattened_data)
    
    return df

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



def main():
    example_usage()

if __name__ == "__main__":
    main()
