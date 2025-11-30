"""Script to create small test fixtures from the actual parquet data."""

import pyarrow.parquet as pq
import pandas as pd
import os

FIXTURE_DIR = os.path.dirname(__file__)

def create_fixture(parquet_name: str) -> pd.DataFrame:
    """Create a small fixture with 5 rows from the specified parquet file.
    
    Args:
        parquet_name: Name of the parquet file (e.g., 'submissions' or 'successful_submissions')
    
    Returns:
        DataFrame with the fixture data
    """
    source_path = os.path.join(FIXTURE_DIR, f'../../data/{parquet_name}.parquet')
    output_path = os.path.join(FIXTURE_DIR, f'{parquet_name}_fixture.parquet')

    if not os.path.exists(source_path):
        print(f"Error: {source_path} does not exist")
        print(f"Please place a recent {parquet_name}.parquet in {source_path} and rerun this script")
        exit(1)

    columns = [
        'submission_id', 'leaderboard_id', 'user_id', 'submission_time',
        'file_name', 'code', 'code_id', 'run_id', 'run_start_time',
        'run_end_time', 'run_mode', 'run_score', 'run_passed',
        'run_compilation', 'run_meta', 'run_system_info'
    ]

    pf = pq.ParquetFile(source_path)
    table = pf.read_row_group(0, columns=columns)
    df = table.to_pandas().head(5)

    df.to_parquet(output_path, index=False)
    print(f"Created {output_path} with {len(df)} rows")
    return df

if __name__ == '__main__':
    create_fixture('submissions')
    create_fixture('successful_submissions')
