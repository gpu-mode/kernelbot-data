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
python scripts/hf_export/export.py --output_dir /path/to/your/dataset
```

The script will create a directory at the specified output path containing the dataset in Parquet format. If `--output_dir` is not provided, it will save to `hf_dataset` in the current working directory. 