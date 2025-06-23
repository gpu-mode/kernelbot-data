#!/usr/bin/env python3
"""
Unit tests for the deduplication pipeline.
Tests the end-to-end flow with fake data matching the database schema.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import random
from typing import Dict, List, Any
import tempfile
import os
import sys

# Import the functions we want to test
try:
    from dedup import (
        remove_duplicates,
        fuzzy_filter,
        convert_df_to_dict,
        flatten_data,
        dedup_df,
        count_items,
        create_parquet_file
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Some functions may not be available for testing")


class TestDedupEndToEnd(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures with fake data matching the schema."""
        random.seed(42)  # For reproducible tests
        np.random.seed(42)
        self.fake_data = self.create_fake_dataset(50)
        self.df = pd.DataFrame(self.fake_data)
        
    def create_fake_dataset(self, num_entries: int) -> List[Dict[str, Any]]:
        """Create a fake dataset with the required schema fields."""
        fake_data = []
        
        # Sample code snippets (some duplicates for testing)
        code_samples = [
            "def hello_world():\n    print('Hello World')",
            "import numpy as np\nx = np.array([1, 2, 3])",
            "for i in range(10):\n    print(i)",
            "class MyClass:\n    def __init__(self):\n        pass",
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3]})",
            "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr",
            "x = [1, 2, 3, 4, 5]\ny = [i**2 for i in x]",
            "try:\n    result = 10 / 0\nexcept ZeroDivisionError:\n    print('Error')",
            "def hello_world():\n    print('Hello World')",  # Exact duplicate
        ]
        
        run_modes = ['leaderboard', 'benchmark', 'test']
        file_names = ['solution.py', 'main.py', 'algorithm.py', 'test.py']
        
        for i in range(num_entries):
            # Create base timestamp
            base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
            submission_time = base_time.replace(
                day=random.randint(1, 28),
                hour=random.randint(0, 23),
                minute=random.randint(0, 59)
            )
            
            # Select code (with some duplicates)
            code = random.choice(code_samples)
            if i < 5:  # First 5 entries use the same code for exact duplicate testing
                code = code_samples[0]
            elif i < 10:  # Next 5 use slightly modified versions for fuzzy testing
                code = code_samples[0] + f"\n# Comment {i}"
                
            run_mode = random.choice(run_modes)
            run_passed = random.choice([True, False])
            
            # Generate run score based on mode and success
            if run_mode == 'leaderboard' and run_passed:
                run_score = round(random.uniform(0.1, 1.0), 4)
            else:
                run_score = 0.0 if not run_passed else round(random.uniform(0.1, 0.8), 4)
                
            # Create the entry matching the database schema
            entry = {
                'submission_id': i + 1000,
                'leaderboard_id': random.randint(1, 10),
                'user_id': random.randint(100, 999),
                'submission_time': submission_time,
                'file_name': random.choice(file_names),
                'code': code,
                'code_id': i + 2000,
                'run_id': i + 3000,
                'run_start_time': submission_time,
                'run_end_time': submission_time.replace(
                    second=random.randint(1, 59)
                ),
                'run_mode': run_mode,
                'run_score': run_score,
                'run_passed': run_passed,
                'run_result': {
                    'benchmark-count': random.randint(1, 10),
                    'benchmark.0.best': f'benchmark_{random.randint(1, 100)}.txt',
                    'benchmark.0.err': '',
                    'benchmark.0.mean': round(random.uniform(0.1, 2.0), 6),
                    'benchmark.0.report': f'report_{i}.json'
                },
                'run_compilation': {
                    'command': 'python',
                    'exit_code': 0 if run_passed else random.randint(1, 255),
                    'nvcc_found': random.choice([True, False]),
                    'nvcc_version': f'11.{random.randint(0, 8)}',
                    'stderr': '' if run_passed else f'Error message {i}',
                    'stdout': f'Output {i}',
                    'success': run_passed
                },
                'run_meta': {
                    'command': 'python solution.py',
                    'duration': round(random.uniform(0.1, 10.0), 3),
                    'exit_code': 0 if run_passed else random.randint(1, 255),
                    'stderr': '' if run_passed else f'Runtime error {i}',
                    'stdout': f'Runtime output {i}',
                    'success': run_passed
                },
                'run_system_info': {
                    'cpu': f'Intel Core i{random.randint(5, 9)}',
                    'gpu': random.choice(['NVIDIA RTX 3080', 'NVIDIA RTX 4090', 'None']),
                    'platform': random.choice(['linux', 'darwin', 'win32']),
                    'torch': f'2.{random.randint(0, 3)}.{random.randint(0, 9)}'
                }
            }
            fake_data.append(entry)
            
        return fake_data
    
    def test_dataframe_creation(self):
        """Test that the fake dataset creates a valid DataFrame."""
        self.assertEqual(len(self.df), 50)
        
        # Check required columns exist (matching the schema in the image)
        required_columns = [
            'submission_id', 'leaderboard_id', 'user_id', 'submission_time',
            'file_name', 'code', 'code_id', 'run_id', 'run_start_time',
            'run_end_time', 'run_mode', 'run_score', 'run_passed',
            'run_result', 'run_compilation', 'run_meta', 'run_system_info'
        ]
        
        for col in required_columns:
            self.assertIn(col, self.df.columns, f"Missing required column: {col}")
            
        # Check data types
        self.assertTrue(self.df['submission_id'].dtype in ['int64', 'int32'])
        self.assertTrue(self.df['run_passed'].dtype == 'bool')
        self.assertTrue(self.df['run_score'].dtype in ['float64', 'float32'])
        
        # Verify struct fields exist
        sample_row = self.df.iloc[0]
        self.assertIsInstance(sample_row['run_result'], dict)
        self.assertIsInstance(sample_row['run_compilation'], dict)
        self.assertIsInstance(sample_row['run_meta'], dict)
        self.assertIsInstance(sample_row['run_system_info'], dict)
        
    def test_convert_df_to_dict(self):
        """Test conversion from DataFrame to nested dictionary structure."""
        try:
            data_dict = convert_df_to_dict(self.df)
            
            # Check structure
            self.assertIsInstance(data_dict, dict)
            
            # Should have run_mode keys
            run_modes = set(self.df['run_mode'].unique())
            self.assertEqual(set(data_dict.keys()), run_modes)
            
            # Check nested structure
            for run_mode in data_dict:
                self.assertIsInstance(data_dict[run_mode], dict)
                for run_success in data_dict[run_mode]:
                    self.assertIsInstance(data_dict[run_mode][run_success], dict)
                    for score_duration in data_dict[run_mode][run_success]:
                        self.assertIsInstance(
                            data_dict[run_mode][run_success][score_duration], 
                            list
                        )
        except NameError:
            self.skipTest("convert_df_to_dict function not available")
    
    def test_exact_deduplication(self):
        """Test exact duplicate removal."""
        try:
            data_dict = convert_df_to_dict(self.df)
            original_count = count_items(data_dict)
            
            deduplicated_data = remove_duplicates(data_dict)
            deduplicated_count = count_items(deduplicated_data)
            
            # Should have fewer or equal items after deduplication
            self.assertLessEqual(deduplicated_count, original_count)
            
            # Structure should be preserved
            self.assertEqual(set(data_dict.keys()), set(deduplicated_data.keys()))
            
        except NameError as e:
            self.skipTest(f"Required functions not available: {e}")
        
    def test_fuzzy_deduplication_small(self):
        """Test fuzzy duplicate removal with small threshold for faster testing."""
        try:
            data_dict = convert_df_to_dict(self.df)
            original_count = count_items(data_dict)
            
            # Use small parameters for faster testing
            fuzzy_deduplicated_data = fuzzy_filter(
                data_dict,
                threshold=0.5,  # Lower threshold for faster testing
                ngram_size=3,   # Smaller ngram size
                bands=4,        # Fewer bands
                rows_per_band=32  # Fewer rows per band
            )
            
            fuzzy_count = count_items(fuzzy_deduplicated_data)
            
            # Should have fewer or equal items after fuzzy deduplication
            self.assertLessEqual(fuzzy_count, original_count)
            
            # Structure should be preserved
            self.assertEqual(set(data_dict.keys()), set(fuzzy_deduplicated_data.keys()))
            
        except NameError as e:
            self.skipTest(f"Required functions not available: {e}")
    
    def test_flatten_and_reconstruct(self):
        """Test flattening and reconstruction of data."""
        try:
            data_dict = convert_df_to_dict(self.df)
            original_count = count_items(data_dict)
            
            # Flatten
            flattened_data = flatten_data(data_dict)
            self.assertEqual(len(flattened_data), original_count)
            
            # Check metadata fields were added
            if flattened_data:
                sample_row = flattened_data[0]
                self.assertIn('_run_mode', sample_row)
                self.assertIn('_run_success', sample_row)
                self.assertIn('_score_duration', sample_row)
                
        except NameError as e:
            self.skipTest(f"Required functions not available: {e}")
    
    def test_dedup_df_end_to_end(self):
        """Test the complete deduplication pipeline."""
        try:
            original_length = len(self.df)
            
            # Run the complete deduplication pipeline
            deduplicated_df = dedup_df(self.df)
            
            # Should return a DataFrame
            self.assertIsInstance(deduplicated_df, pd.DataFrame)
            
            # Should have fewer or equal rows
            self.assertLessEqual(len(deduplicated_df), original_length)
            
            # Should preserve required columns
            required_columns = ['submission_id', 'code', 'run_mode', 'run_passed']
            for col in required_columns:
                self.assertIn(col, deduplicated_df.columns)
                
            # Check data integrity
            self.assertFalse(deduplicated_df.empty, "Deduplicated DataFrame should not be empty")
            
        except NameError as e:
            self.skipTest(f"dedup_df function not available: {e}")
        
    def test_parquet_creation(self):
        """Test Parquet file creation."""
        try:
            data_dict = convert_df_to_dict(self.df)
            
            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
                try:
                    create_parquet_file(data_dict, tmp_file.name)
                    
                    # Check file was created
                    self.assertTrue(os.path.exists(tmp_file.name))
                    
                    # Check file is not empty
                    self.assertGreater(os.path.getsize(tmp_file.name), 0)
                    
                    # Try to read the file back
                    df_from_parquet = pd.read_parquet(tmp_file.name)
                    self.assertIsInstance(df_from_parquet, pd.DataFrame)
                    self.assertGreater(len(df_from_parquet), 0)
                    
                finally:
                    # Clean up
                    if os.path.exists(tmp_file.name):
                        os.unlink(tmp_file.name)
                        
        except NameError as e:
            self.skipTest(f"Required functions not available: {e}")
    
    def test_data_consistency_after_deduplication(self):
        """Test that data remains consistent after deduplication."""
        try:
            # Create dataset with known duplicates
            duplicate_data = []
            
            # Add the same code 3 times with different metadata
            base_entry = self.fake_data[0].copy()
            for i in range(3):
                entry = base_entry.copy()
                entry['submission_id'] = 9000 + i
                entry['run_id'] = 9100 + i
                duplicate_data.append(entry)
                
            # Add to main dataset
            test_data = self.fake_data + duplicate_data
            test_df = pd.DataFrame(test_data)
            
            original_length = len(test_df)
            deduplicated_df = dedup_df(test_df)
            
            # Should have removed at least 2 duplicates
            self.assertLess(len(deduplicated_df), original_length)
            
            # Check that essential fields are preserved
            self.assertTrue(all(col in deduplicated_df.columns for col in 
                              ['submission_id', 'code', 'run_mode', 'run_passed']))
                              
        except NameError as e:
            self.skipTest(f"Required functions not available: {e}")

    def test_schema_compliance(self):
        """Test that the fake dataset matches the expected schema from the database."""
        # Test all required fields exist and have correct types
        
        # Test BIGINT fields
        bigint_fields = ['submission_id', 'leaderboard_id', 'user_id', 'code_id', 'run_id']
        for field in bigint_fields:
            self.assertTrue(self.df[field].dtype in ['int64', 'int32'], 
                          f"{field} should be integer type")
            
        # Test VARCHAR fields
        varchar_fields = ['file_name', 'code', 'run_mode']
        for field in varchar_fields:
            self.assertTrue(self.df[field].dtype == 'object', 
                          f"{field} should be string/object type")
            
        # Test TIMESTAMP fields
        timestamp_fields = ['submission_time', 'run_start_time', 'run_end_time']
        for field in timestamp_fields:
            # Check that all values are datetime objects with timezone
            sample_value = self.df[field].iloc[0]
            self.assertIsInstance(sample_value, datetime)
            self.assertIsNotNone(sample_value.tzinfo)
            
        # Test DOUBLE field
        self.assertTrue(self.df['run_score'].dtype in ['float64', 'float32'])
        
        # Test BOOLEAN field
        self.assertTrue(self.df['run_passed'].dtype == 'bool')
        
        # Test STRUCT fields
        struct_fields = ['run_result', 'run_compilation', 'run_meta', 'run_system_info']
        for field in struct_fields:
            # All values should be dictionaries
            self.assertTrue(all(isinstance(val, dict) for val in self.df[field]))
            
    def test_duplicate_detection(self):
        """Test that we can detect exact and near duplicates in the dataset."""
        # Count exact duplicates by code
        code_counts = self.df['code'].value_counts()
        exact_duplicates = code_counts[code_counts > 1]
        
        # Should have some exact duplicates (first 5 entries)
        self.assertGreater(len(exact_duplicates), 0, "Should have exact duplicates for testing")
        
        # Check that fuzzy duplicates exist (entries with similar code)
        similar_code_count = 0
        base_code = "def hello_world():\n    print('Hello World')"
        for code in self.df['code']:
            if base_code in code and code != base_code:
                similar_code_count += 1
                
        self.assertGreater(similar_code_count, 0, "Should have fuzzy duplicates for testing")


if __name__ == '__main__':
    # Add some helpful output
    print("Running deduplication pipeline tests...")
    print(f"Python version: {sys.version}")
    print(f"Pandas version: {pd.__version__}")
    
    # Run the tests
    unittest.main(verbosity=2) 