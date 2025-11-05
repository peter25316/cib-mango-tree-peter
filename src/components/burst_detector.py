# File: src/component/burst_detector.py

import polars as pl
import pandas as pd
import datetime

# Import the kleinberg function from its utility file
from .kleinberg_utils import kleinberg


class BurstDetector:
    """
    Runs the Kleinberg burst detection algorithm and processes the results.
    """

    def __init__(self, s=2.0, gamma=1.0):
        self.s = s
        self.gamma = gamma
        self.burst_list = []
        self.posts_per_hour_with_bursts = None

    def detect_bursts(self, ts_df: pl.DataFrame, posts_per_hour_transformed: pl.DataFrame):
        """
        Detects bursts from the raw timestamp data and maps results to the
        aggregated dataframe.
        """
        if ts_df is None or posts_per_hour_transformed is None:
            print("Error: Missing required dataframes (ts_df or posts_per_hour_transformed).")
            return None, None

        print("Preparing raw numeric timestamps for burst detection...")
        raw_timestamps = (
            ts_df
            .sort('post_timestamp')
            .select(
                pl.col('post_timestamp').dt.epoch(time_unit="s").alias('unix_timestamp')
            )
            .unique()
            ['unix_timestamp']
            .to_list()
        )
        print(f"Prepared data: {len(raw_timestamps)} unique timestamps.")

        print("Running Kleinberg's algorithm...")
        bursts_raw = kleinberg(raw_timestamps, s=self.s, gamma=self.gamma)
        print("Algorithm complete.")

        # Process results
        self._process_burst_results(bursts_raw, posts_per_hour_transformed)

        return self.burst_list, self.posts_per_hour_with_bursts

    def _process_burst_results(self, bursts_raw, posts_per_hour_transformed):
        """Helper method to process the raw output from the Kleinberg algorithm."""
        print("Processing burst detection results...")
        self.posts_per_hour_with_bursts = posts_per_hour_transformed.with_columns(
            pl.lit(0).alias('burst_level')
        )
        self.burst_list = []

        for item in bursts_raw:
            level = float(item[0])
            start_unix = int(item[1])
            end_unix = int(item[2])

            if level > 0:
                # Use fromtimestamp with explicit UTC, then convert to naive datetimes
                # to keep comparisons compatible with Polars' naive datetime dtype.
                start_time = datetime.datetime.fromtimestamp(start_unix, tz=datetime.timezone.utc).replace(tzinfo=None)
                end_time = datetime.datetime.fromtimestamp(end_unix, tz=datetime.timezone.utc).replace(tzinfo=None)

                self.burst_list.append({
                    'level': level,
                    'start_time': start_time,
                    'end_time': end_time,
                })

                # Map burst levels to the aggregated DataFrame for plotting
                cond_after_start = pl.col('post_timestamp') >= start_time
                cond_before_end = pl.col('post_timestamp') <= end_time

                self.posts_per_hour_with_bursts = self.posts_per_hour_with_bursts.with_columns(
                    pl.when(cond_after_start & cond_before_end)
                    .then(pl.lit(level))
                    .otherwise(pl.col('burst_level'))
                    .alias('burst_level')
                )

        print(f"Found {len(self.burst_list)} bursts.")
        print(pl.DataFrame(self.burst_list))