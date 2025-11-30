# File: src/components/burst_detector_enhanced.py

import polars as pl
import datetime

# Import the kleinberg function from its utility file
from .kleinberg_utils import kleinberg


class BurstDetectorEnhanced:
    """
    Enhanced version of BurstDetector with adaptive contributor selection
    for better coordination network detection.
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
        aggregated dataframe. Enhanced version with adaptive contributor selection.
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

        # If full posts DataFrame provided, map contributors with enhanced detection
        if posts_df is not None:
            try:
                self._map_contributors_enhanced(posts_df)
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

    def _map_contributors_enhanced(self, posts_df: pl.DataFrame):
        """
        Enhanced contributor mapping with adaptive selection for coordination detection.
        Uses multiple thresholds instead of fixed top 10 to capture coordination patterns.
        """
        print("Mapping posts/accounts to detected bursts (enhanced)...")

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
                posts_list = posts_in_burst.select(posts_in_burst.columns).to_dicts()

            # Enhanced adaptive contributor selection
            significant_accounts = []
            coordination_metrics = {}

            if 'account.username' in cols:
                try:
                    usernames = posts_in_burst.select('account.username').to_series().to_list()
                except Exception:
                    try:
                        usernames = posts_in_burst['account.username'].to_list()
                    except Exception:
                        usernames = []

                from collections import Counter
                counts = Counter([u for u in usernames if u is not None])
                total_posts = len(usernames)

                if total_posts > 0:
                    # Adaptive threshold approach for coordination detection

                    # Strategy 1: Minimum activity threshold (at least 3 posts or 2% of burst)
                    min_posts_threshold = max(3, int(total_posts * 0.02))

                    # Strategy 2: Capture accounts contributing to significant portion of activity
                    accounts_by_count = counts.most_common()
                    cumulative_posts = 0

                    for username, count in accounts_by_count:
                        percentage = (count / total_posts) * 100

                        # Include if meets minimum threshold OR is in top contributors up to 85% coverage
                        if count >= min_posts_threshold or (cumulative_posts / total_posts) < 0.85:
                            significant_accounts.append({
                                'account.username': username,
                                'count': count,
                                'percentage': round(percentage, 2)
                            })
                            cumulative_posts += count

                        # Safety limits for analysis efficiency
                        if len(significant_accounts) >= 25:  # Max 25 accounts
                            break

                    # Ensure minimum coverage - add top contributors if needed
                    if len(significant_accounts) < 5:
                        top_accounts = counts.most_common(5)
                        for username, count in top_accounts:
                            if not any(acc['account.username'] == username for acc in significant_accounts):
                                percentage = (count / total_posts) * 100
                                significant_accounts.append({
                                    'account.username': username,
                                    'count': count,
                                    'percentage': round(percentage, 2)
                                })

            contributors = {
                'start_time': start,
                'end_time': end,
                'post_count': posts_in_burst.height,
                'posts': posts_list,
                'top_accounts': significant_accounts  # Adaptive contributor selection for content analysis
            }

            # Attach to burst dict
            b['contributors'] = contributors
            contributors_list.append(contributors)

        self.burst_contributors = contributors_list

        # Enhanced statistics reporting
        total_contributors = sum(len(c['top_accounts']) for c in contributors_list)
        avg_contributors = total_contributors / len(contributors_list) if contributors_list else 0
        print(f"Mapped contributors for {len(self.burst_contributors)} bursts.")
        print(f"Average {avg_contributors:.1f} significant contributors per burst (adaptive threshold).")

        print("🚨 Burst detection complete. Coordination analysis will be handled by content analysis layer.")

