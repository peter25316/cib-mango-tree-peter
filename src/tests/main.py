# File: src/tests/main.py

import os
import sys

# This is a crucial step to ensure Python can find your component modules.
# It adds the 'src' directory (one level up from 'tests') to the Python path.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Now we can import from the 'component' package
from components.data_analyzer import DataAnalyzer
from components.burst_detector import BurstDetector
import components.visualizer as viz


def run_analysis():
    """
    Executes the full data analysis and burst detection pipeline.
    """

    # --- 1. Define File Path ---
    # This path goes up two levels (from src/tests to the root) then into data/
    DATA_FILE_PATH = os.path.abspath(os.path.join(
        project_root,
        '..',
        'data',
        'sampledata_truthsocial.csv'  # <-- Make sure this filename is correct
    ))

    print(f"Looking for data file at: {DATA_FILE_PATH}")

    # --- 2. Data Loading and Preparation ---
    try:
        analyzer = DataAnalyzer(data_file_path=DATA_FILE_PATH)
        analyzer.run_all()  # Runs load, prepare, aggregate, and transform
    except Exception as e:
        print(f"Fatal error during data preparation: {e}")
        return

    # --- 3. Initial Diagnostic Plots (Optional) ---
    # These plots correspond to Week 1 in your script
    # You can comment these out if you only want the final results
    print("\n--- Running Diagnostic Plots ---")
    viz.plot_hourly_posts(analyzer.posts_per_hour)
    viz.plot_acf(analyzer.posts_per_hour)
    viz.plot_transformation_comparison(
        analyzer.posts_per_hour,
        analyzer.posts_per_hour_transformed
    )

    # --- 4. Burst Detection ---
    print("\n--- Running Burst Detection ---")
    detector = BurstDetector(s=2.0, gamma=1.0)
    burst_list, posts_with_bursts = detector.detect_bursts(
        ts_df=analyzer.ts_df,
        posts_per_hour_transformed=analyzer.posts_per_hour_transformed
    )

    if not burst_list:
        print("No bursts were detected.")
        return

    # --- 5. Final Visualization ---
    print("\n--- Running Final Visualizations ---")
    # This is the "smoking gun" plot
    viz.plot_burst_rectangles(
        analyzer.posts_per_hour_transformed,
        burst_list
    )

    # This is the Gantt chart summary
    viz.plot_burst_gantt(burst_list)

    print("\n--- Analysis Complete ---")


if __name__ == "__main__":
    run_analysis()