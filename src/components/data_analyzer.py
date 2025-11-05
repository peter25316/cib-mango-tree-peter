# File: src/component/data_analyzer.py

import polars as pl
import os


class DataAnalyzer:
    """
    Handles loading, preparing, and transforming the time-series data.
    """

    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        self.df = None
        self.ts_df = None
        self.posts_per_hour = None
        self.posts_per_hour_transformed = None

    def load_data(self):
        """Loads the CSV file from the specified path."""
        print("Loading data...")
        try:
            self.df = pl.read_csv(self.data_file_path)
            print(f"Data loaded successfully. {len(self.df)} rows.")
            print(self.df.head())
        except FileNotFoundError:
            print(f"Error: Data file not found at {self.data_file_path}")
            raise
        except Exception as e:
            print(f"An error occurred during data loading: {e}")
            raise

    def prepare_timestamps(self):
        """Isolates, renames, and converts the timestamp column."""
        if self.df is None:
            print("Error: Data not loaded. Call load_data() first.")
            return

        print("Preparing timestamps...")
        try:
            ts_df = self.df.select(
                pl.col('created_at').alias('post_timestamp')
            )

            # Only attempt conversion for string columns
            if ts_df['post_timestamp'].dtype == pl.Utf8:
                print("Converting string (ISO 8601 format) to datetime...")
                # First, try Polars' parser without unsupported kwargs
                try:
                    ts_df = ts_df.with_columns(
                        pl.col('post_timestamp').str.to_datetime().alias('post_timestamp')
                    )
                except Exception as e:
                    # Fall back to an eager, Python-based parsing approach.
                    print("Fast parse failed, falling back to Python parsing due to:", e)
                    from dateutil import parser as _parser

                    raw_values = ts_df['post_timestamp'].to_list()
                    parsed = []
                    for s in raw_values:
                        if s is None or s == "":
                            parsed.append(None)
                            continue
                        try:
                            parsed.append(_parser.isoparse(s))
                        except Exception:
                            # If parsing fails, append None to keep lengths consistent
                            parsed.append(None)

                    # Replace the column with a new Series of parsed datetimes
                    ts_df = ts_df.with_columns(
                        pl.Series('post_timestamp', parsed).cast(pl.Datetime)
                    )

            self.ts_df = ts_df.sort("post_timestamp")
            print("Timestamp preparation complete.")
            print(self.ts_df.head())

        except pl.ColumnNotFoundError:
            print("Error: Column 'created_at' not found in the DataFrame.")
            raise
        except Exception as e:
            print(f"An error occurred during conversion: {e}")
            raise

    def aggregate_by_hour(self):
        """Aggregates the data into posts per hour."""
        if self.ts_df is None:
            print("Error: Timestamps not prepared. Call prepare_timestamps() first.")
            return

        print("Aggregating posts per hour...")
        self.posts_per_hour = self.ts_df.group_by_dynamic(
            index_column="post_timestamp",
            every="1h",
        ).agg(
            pl.len().alias('post_count')
        ).sort("post_timestamp")
        print("Aggregation complete.")
        print(self.posts_per_hour.head())

    def apply_log_transform(self):
        """Applies a log(1+x) transformation to stabilize variance."""
        if self.posts_per_hour is None:
            print("Error: Data not aggregated. Call aggregate_by_hour() first.")
            return

        print("Applying log transformation...")
        self.posts_per_hour_transformed = self.posts_per_hour.with_columns(
            pl.col('post_count').log1p().alias('log_post_count')
        )
        print("Transformation complete.")
        print(self.posts_per_hour_transformed.head())

    def run_all(self):
        """Convenience method to run all data preparation steps."""
        self.load_data()
        self.prepare_timestamps()
        self.aggregate_by_hour()
        self.apply_log_transform()