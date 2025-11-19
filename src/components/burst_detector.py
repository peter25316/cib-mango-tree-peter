# File: src/component/burst_detector.py

import polars as pl
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
        self.burst_contributors = []

    def detect_bursts(self, ts_df: pl.DataFrame, posts_per_hour_transformed: pl.DataFrame, posts_df: pl.DataFrame = None):
        """
        Detects bursts from the raw timestamp data and maps results to the
        aggregated dataframe. If `posts_df` (the original posts with account
        and content fields) is provided, the method will also record which
        posts and accounts contribute to each detected burst interval.
        """
        if ts_df is None or posts_per_hour_transformed is None:
            print("Error: Missing required dataframes (ts_df or posts_per_hour_transformed).")
            return None, None, None

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

        # If full posts DataFrame provided, map contributors
        if posts_df is not None:
            try:
                self._map_contributors(posts_df)
            except Exception as e:
                print(f"Warning: failed to map contributors to bursts: {e}")
                self.burst_contributors = []

        return self.burst_list, self.posts_per_hour_with_bursts, self.burst_contributors

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

    def _map_contributors(self, posts_df: pl.DataFrame):
        """
        For each burst interval, find posts and accounts from `posts_df` whose
        timestamp falls within the burst window. The function will attach a
        `contributors` field to each burst entry containing a list of post
        dicts and top account contributors.
        """
        print("Mapping posts/accounts to detected bursts...")

        df = posts_df
        # Determine which timestamp column to use
        if 'created_at' in df.columns:
            time_col = 'created_at'
        elif 'post_timestamp' in df.columns:
            time_col = 'post_timestamp'
        else:
            print("No timestamp column found in posts_df; skipping mapping.")
            self.burst_contributors = []
            return

        # Ensure timestamp column is Datetime type
        try:
            if df[time_col].dtype == pl.Utf8:
                df = df.with_columns(pl.col(time_col).str.to_datetime().alias(time_col))
        except Exception:
            # If conversion fails, attempt a cast
            try:
                df = df.with_columns(pl.col(time_col).cast(pl.Datetime).alias(time_col))
            except Exception:
                print("Failed to parse timestamps in posts_df; skipping mapping.")
                self.burst_contributors = []
                return

        contributors_list = []

        for b in self.burst_list:
            start = b['start_time']
            end = b['end_time']

            # Filter posts inside the interval
            try:
                mask = (pl.col(time_col) >= start) & (pl.col(time_col) <= end)
                posts_in_burst = df.filter(mask)
            except Exception as e:
                print(f"Error filtering posts for burst interval: {e}")
                posts_in_burst = df.head(0)

            # Select useful columns if they exist
            cols = posts_in_burst.columns
            selected_cols = []
            for c in ['id', 'content_cleaned', 'account.username', 'account.display_name', 'account.id']:
                if c in cols:
                    selected_cols.append(c)

            posts_list = []
            if posts_in_burst.height > 0 and selected_cols:
                posts_list = posts_in_burst.select(selected_cols).to_dicts()
            elif posts_in_burst.height > 0:
                # fallback: include all columns (limited size)
                posts_list = posts_in_burst.select(posts_in_burst.columns).to_dicts()

            # Compute top accounts by post count if username column exists
            top_accounts = []
            if 'account.username' in cols:
                # Use a robust Python-side counter to avoid Polars version method
                # differences. This will work even if the DataFrame is a pandas
                # object or Polars DataFrame.
                try:
                    # Prefer Polars fast path
                    usernames = posts_in_burst.select('account.username').to_series().to_list()
                except Exception:
                    try:
                        usernames = posts_in_burst['account.username'].to_list()
                    except Exception:
                        usernames = []

                from collections import Counter
                counts = Counter([u for u in usernames if u is not None])
                top_accounts = [{'account.username': u, 'count': c} for u, c in counts.most_common(10)]

            contributors = {
                'start_time': start,
                'end_time': end,
                'post_count': posts_in_burst.height,
                'posts': posts_list,
                'top_accounts': top_accounts,
            }

            # attach to burst dict
            b['contributors'] = contributors
            contributors_list.append(contributors)

        self.burst_contributors = contributors_list
        print(f"Mapped contributors for {len(self.burst_contributors)} bursts.")
