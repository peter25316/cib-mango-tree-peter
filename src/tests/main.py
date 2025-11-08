# File: src/tests/main.py

import os
import sys
import pickle
import polars as pl
from typing import Optional

# This is a crucial step to ensure Python can find your component modules.
# It adds the 'src' directory (one level up from 'tests') to the Python path.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Now we can import from the 'component' package
from components.data_analyzer import DataAnalyzer
from components.burst_detector import BurstDetector
import components.visualizer as viz
from components.temporal_clusterer import TemporalClusterer

# Cache file paths
CACHE_DIR = os.path.abspath(os.path.join(project_root, '..', 'cache'))
ANALYZER_CACHE = os.path.join(CACHE_DIR, 'analyzer.pkl')
BURST_CACHE = os.path.join(CACHE_DIR, 'burst_results.pkl')
TIMESTAMP_CACHE = os.path.join(CACHE_DIR, 'timestamp_df.pkl')
POSTS_PER_HOUR_CACHE = os.path.join(CACHE_DIR, 'posts_per_hour.pkl')
POSTS_PER_HOUR_TRANSFORMED_CACHE = os.path.join(CACHE_DIR, 'posts_per_hour_transformed.pkl')

# Add new cache paths for temporal clustering
TEMPORAL_FEATURES_CACHE = os.path.join(CACHE_DIR, 'temporal_features.pkl')
TEMPORAL_CLUSTERS_CACHE = os.path.join(CACHE_DIR, 'temporal_clusters.pkl')

# Add new cache path for plots
PLOTS_DIR = os.path.abspath(os.path.join(project_root, '..', 'plots'))


def ensure_cache_dir():
    """Ensure the cache directory exists"""
    os.makedirs(CACHE_DIR, exist_ok=True)


def ensure_plots_dir():
    """Ensure the plots directory exists"""
    os.makedirs(PLOTS_DIR, exist_ok=True)


def load_cached_analyzer() -> Optional[DataAnalyzer]:
    """Load cached analyzer if it exists"""
    if os.path.exists(ANALYZER_CACHE):
        try:
            with open(ANALYZER_CACHE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading analyzer cache: {e}")
    return None


def load_cached_bursts():
    """Load cached burst results if they exist"""
    if os.path.exists(BURST_CACHE):
        try:
            with open(BURST_CACHE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading burst cache: {e}")
    return None, None


def load_timestamp_df():
    """Load cached timestamp DataFrame if it exists"""
    if os.path.exists(TIMESTAMP_CACHE):
        try:
            with open(TIMESTAMP_CACHE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading timestamp cache: {e}")
    return None


def load_posts_per_hour():
    """Load cached posts per hour if it exists"""
    if os.path.exists(POSTS_PER_HOUR_CACHE):
        try:
            with open(POSTS_PER_HOUR_CACHE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading posts per hour cache: {e}")
    return None


def load_transformed_data():
    """Load cached transformed posts per hour if it exists"""
    if os.path.exists(POSTS_PER_HOUR_TRANSFORMED_CACHE):
        try:
            with open(POSTS_PER_HOUR_TRANSFORMED_CACHE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading transformed data cache: {e}")
    return None


def save_analyzer(analyzer: DataAnalyzer):
    """Save analyzer state to cache"""
    ensure_cache_dir()
    with open(ANALYZER_CACHE, 'wb') as f:
        pickle.dump(analyzer, f)


def save_burst_results(burst_list, posts_with_bursts):
    """Save burst detection results to cache"""
    ensure_cache_dir()
    with open(BURST_CACHE, 'wb') as f:
        pickle.dump((burst_list, posts_with_bursts), f)


def save_timestamp_df(ts_df):
    """Save prepared timestamp DataFrame to cache"""
    ensure_cache_dir()
    with open(TIMESTAMP_CACHE, 'wb') as f:
        pickle.dump(ts_df, f)


def save_posts_per_hour(posts_per_hour):
    """Save posts per hour aggregation to cache"""
    ensure_cache_dir()
    with open(POSTS_PER_HOUR_CACHE, 'wb') as f:
        pickle.dump(posts_per_hour, f)


def save_transformed_data(posts_per_hour_transformed):
    """Save transformed posts per hour to cache"""
    ensure_cache_dir()
    with open(POSTS_PER_HOUR_TRANSFORMED_CACHE, 'wb') as f:
        pickle.dump(posts_per_hour_transformed, f)


def save_temporal_features(features_df):
    """Save temporal features to cache"""
    ensure_cache_dir()
    with open(TEMPORAL_FEATURES_CACHE, 'wb') as f:
        pickle.dump(features_df, f)


def load_temporal_features():
    """Load cached temporal features if they exist"""
    if os.path.exists(TEMPORAL_FEATURES_CACHE):
        try:
            with open(TEMPORAL_FEATURES_CACHE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading temporal features cache: {e}")
    return None


def save_temporal_clusters(clusters_df):
    """Save temporal clustering results to cache"""
    ensure_cache_dir()
    with open(TEMPORAL_CLUSTERS_CACHE, 'wb') as f:
        pickle.dump(clusters_df, f)


def load_temporal_clusters():
    """Load cached temporal clustering results if they exist"""
    if os.path.exists(TEMPORAL_CLUSTERS_CACHE):
        try:
            with open(TEMPORAL_CLUSTERS_CACHE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading temporal clusters cache: {e}")
    return None


def get_plot_path(plot_name: str) -> str:
    """Get the full path for a plot file"""
    ensure_plots_dir()
    return os.path.join(PLOTS_DIR, f"{plot_name}.png")


def run_data_preparation(force_rerun=False) -> Optional[DataAnalyzer]:
    """Run data preparation steps and cache results"""
    if not force_rerun:
        cached_analyzer = load_cached_analyzer()
        cached_ts_df = load_timestamp_df()
        if cached_analyzer is not None and cached_ts_df is not None:
            print("Using cached analyzer and timestamp results")
            return cached_analyzer, cached_ts_df

    # Define File Path
    DATA_FILE_PATH = os.path.abspath(os.path.join(
        project_root,
        '..',
        'data',
        'sampledata_truthsocial.csv'
    ))

    print(f"Looking for data file at: {DATA_FILE_PATH}")

    try:
        analyzer = DataAnalyzer(data_file_path=DATA_FILE_PATH)
        analyzer.run_all()

        # Prepare timestamp DataFrame
        ts_df = analyzer.ts_df

        save_analyzer(analyzer)
        save_timestamp_df(ts_df)
        return analyzer, ts_df
    except Exception as e:
        print(f"Fatal error during data preparation: {e}")
        return None, None


def run_posts_per_hour_aggregation(ts_df, force_rerun=False):
    """Run hourly aggregation and cache results"""
    if not force_rerun:
        cached_posts = load_posts_per_hour()
        if cached_posts is not None:
            print("Using cached posts per hour aggregation")
            return cached_posts

    if ts_df is None:
        print("No timestamp data available. Run data preparation first.")
        return None

    try:
        # Sort data by timestamp
        ts_df_sorted = ts_df.sort("post_timestamp")

        # Aggregate Data by Hour
        posts_per_hour = ts_df_sorted.group_by_dynamic(
            index_column="post_timestamp",
            every="1h",
        ).agg(
            pl.len().alias('post_count')
        ).sort("post_timestamp")

        save_posts_per_hour(posts_per_hour)
        return posts_per_hour
    except Exception as e:
        print(f"Error during hourly aggregation: {e}")
        return None


def run_data_transformation(posts_per_hour, force_rerun=False):
    """Run data transformation and cache results"""
    if not force_rerun:
        cached_transformed = load_transformed_data()
        if cached_transformed is not None:
            print("Using cached transformed data")
            return cached_transformed

    if posts_per_hour is None:
        print("No hourly data available. Run hourly aggregation first.")
        return None

    try:
        # Apply Log Transformation
        posts_per_hour_transformed = posts_per_hour.with_columns(
            pl.col('post_count').log1p().alias('log_post_count')
        )

        save_transformed_data(posts_per_hour_transformed)
        return posts_per_hour_transformed
    except Exception as e:
        print(f"Error during data transformation: {e}")
        return None


def run_diagnostic_plots(analyzer: DataAnalyzer):
    """Run diagnostic plots"""
    if analyzer is None:
        print("No analyzer available. Run data preparation first.")
        return

    print("\n--- Running Diagnostic Plots ---")

    # Create and save hourly posts plot
    print("Creating plot: Posts Per Hour...")
    hourly_plot_path = get_plot_path("hourly_posts")
    viz.plot_hourly_posts(analyzer.posts_per_hour, save_path=hourly_plot_path)

    # Create and save ACF plot
    print("Creating plot: Autocorrelation Function (ACF)...")
    acf_plot_path = get_plot_path("acf_plot")
    viz.plot_acf(analyzer.posts_per_hour, save_path=acf_plot_path)

    # Create and save transformation comparison plot
    print("Creating plot: Variance Comparison...")
    transform_plot_path = get_plot_path("transformation_comparison")
    viz.plot_transformation_comparison(
        analyzer.posts_per_hour,
        analyzer.posts_per_hour_transformed,
        save_path=transform_plot_path
    )


def run_burst_detection(analyzer: DataAnalyzer, force_rerun=False):
    """Run burst detection and cache results"""
    if analyzer is None:
        print("No analyzer available. Run data preparation first.")
        return None, None

    if not force_rerun:
        cached_results = load_cached_bursts()
        if cached_results[0] is not None:
            print("Using cached burst detection results")
            return cached_results

    print("\n--- Running Burst Detection ---")
    detector = BurstDetector(s=2.0, gamma=1.0)
    burst_list, posts_with_bursts = detector.detect_bursts(
        ts_df=analyzer.ts_df,
        posts_per_hour_transformed=analyzer.posts_per_hour_transformed
    )

    if burst_list:
        save_burst_results(burst_list, posts_with_bursts)
    else:
        print("No bursts were detected.")

    return burst_list, posts_with_bursts


def run_final_visualizations(analyzer: DataAnalyzer, burst_list):
    """Run final visualizations"""
    if analyzer is None or not burst_list:
        print("Missing required data. Run previous steps first.")
        return

    print("\n--- Running Final Visualizations ---")

    # Create and save burst rectangles plot
    print("Creating plot: Burst Detection with Rectangles...")
    burst_rect_path = get_plot_path("burst_rectangles")
    viz.plot_burst_rectangles(
        analyzer.posts_per_hour_transformed,
        burst_list,
        save_path=burst_rect_path
    )

    # Create and save burst Gantt chart
    print("Creating plot: Burst Gantt Chart...")
    burst_gantt_path = get_plot_path("burst_gantt")
    viz.plot_burst_gantt(burst_list, save_path=burst_gantt_path)


def run_temporal_feature_engineering(df, force_rerun=False):
    """Run temporal feature engineering and cache results"""
    if not force_rerun:
        cached_features = load_temporal_features()
        if cached_features is not None:
            print("Using cached temporal features")
            return cached_features

    try:
        clusterer = TemporalClusterer(min_posts=5)
        features_df = clusterer.engineer_features(df)
        save_temporal_features(features_df)
        return features_df
    except Exception as e:
        print(f"Error during temporal feature engineering: {e}")
        return None


def run_temporal_clustering(features_df, force_rerun=False):
    """Run temporal clustering analysis and cache results"""
    if not force_rerun:
        cached_clusters = load_temporal_clusters()
        if cached_clusters is not None:
            print("Using cached temporal clustering results")
            return cached_clusters

    try:
        # Initialize clusterer and prepare data
        clusterer = TemporalClusterer(min_posts=5)
        clusterer.account_features = features_df
        scaled_features = clusterer.prepare_for_clustering()

        # Find optimal k using elbow method
        inertia = clusterer.find_optimal_k()
        clusterer.plot_elbow(inertia, save_path=get_plot_path("elbow_plot"))

        # Run final clustering with k=3 (based on elbow analysis)
        cluster_results = clusterer.run_clustering(n_clusters=3)

        # Create cluster visualization
        clusterer.plot_clusters(save_path=get_plot_path("temporal_clusters"))

        # Save results
        save_temporal_clusters(cluster_results)
        return cluster_results
    except Exception as e:
        print(f"Error during temporal clustering: {e}")
        return None


def run_complete_analysis(force_rerun=False):
    """Run the complete analysis pipeline"""
    print("\n=== Starting Complete Analysis Pipeline ===\n")

    # 1. Data Preparation
    print("Step 1: Data Preparation...")
    analyzer, ts_df = run_data_preparation(force_rerun=force_rerun)
    if analyzer is None or ts_df is None:
        print("Fatal error in data preparation. Stopping analysis.")
        return
    print("✓ Data preparation complete\n")

    # 2. Hourly Aggregation
    print("Step 2: Hourly Aggregation...")
    posts_per_hour = run_posts_per_hour_aggregation(ts_df, force_rerun=force_rerun)
    if posts_per_hour is None:
        print("Fatal error in hourly aggregation. Stopping analysis.")
        return
    print("✓ Hourly aggregation complete\n")

    # 3. Data Transformation
    print("Step 3: Data Transformation...")
    posts_per_hour_transformed = run_data_transformation(posts_per_hour, force_rerun=force_rerun)
    if posts_per_hour_transformed is None:
        print("Fatal error in data transformation. Stopping analysis.")
        return
    print("✓ Data transformation complete\n")

    # 4. Diagnostic Plots
    print("Step 4: Generating Diagnostic Plots...")
    run_diagnostic_plots(analyzer)
    print("✓ Diagnostic plots complete\n")

    # 5. Burst Detection
    print("Step 5: Burst Detection Analysis...")
    burst_list, posts_with_bursts = run_burst_detection(analyzer, force_rerun=force_rerun)
    if burst_list is None:
        print("Fatal error in burst detection. Stopping analysis.")
        return
    print("✓ Burst detection complete\n")

    # 6. Final Visualizations
    print("Step 6: Generating Final Visualizations...")
    run_final_visualizations(analyzer, burst_list)
    print("✓ Final visualizations complete\n")

    # 7. Temporal Feature Engineering
    print("Step 7: Temporal Feature Engineering...")
    temporal_features = run_temporal_feature_engineering(df, force_rerun=force_rerun)
    if temporal_features is None:
        print("Fatal error in temporal feature engineering. Stopping analysis.")
        return
    print("✓ Temporal feature engineering complete\n")

    # 8. Temporal Clustering
    print("Step 8: Temporal Clustering Analysis...")
    temporal_clusters = run_temporal_clustering(temporal_features, force_rerun=force_rerun)
    if temporal_clusters is None:
        print("Fatal error in temporal clustering. Stopping analysis.")
        return
    print("✓ Temporal clustering complete\n")

    print("=== Complete Analysis Pipeline Finished ===")
    return analyzer, ts_df, posts_per_hour, posts_per_hour_transformed, burst_list, posts_with_bursts, temporal_features, temporal_clusters

if __name__ == "__main__":
    # For first run or to force everything to run fresh, use force_rerun=True
    results = run_complete_analysis(force_rerun=True)
