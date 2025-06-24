# Hugging Face Dataset Exporter

This script exports data from a Postgres database to a Hugging Face dataset in Parquet format.

## Setup

1.  **Install dependencies:**

    Navigate to this directory and install the required Python packages.

    ```bash
    pip install -r requirements.txt
    ```

2.  **Set environment variables:**

    The script uses environment variables for database credentials. You can set them in your shell or use a `.env` file.

    ```bash
    export DB_USER="your_db_user"
    export DB_PASSWORD="your_db_password"
    export DB_HOST="localhost"
    export DB_PORT="5432"
    export DB_NAME="your_db_name"
    ```

## Usage

Run the script from the root of the repository:

```bash
python export.py
```

The script will create a directory at the specified output path containing the dataset in Parquet format. If `--output_dir` is not provided, it will save to `dataset` in the current working directory.

## Tests
The deduplication scripts can be tested by running
```bash
python test_dedup.py
# if you have pytest you can run 
python -m pytest test_dedup.py -v
```
To test things we actually create a fake dataset. Here are the features of it
The test creates a 50-entry dataset with:
- **Exact duplicates**: First 5 entries use identical code
- **Fuzzy duplicates**: Next 5 entries use similar code with small variations
- **Multiple run modes**: `leaderboard`, `test`, `benchmark`
- **Mixed success states**: Both `True` and `False` values for `run_passed`
- **Realistic struct data**: Complex nested structures for `run_result`, `run_compilation`, `run_meta`, and `run_system_info`
- **Proper timestamps**: All timestamp fields include timezone information
