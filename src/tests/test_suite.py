#!/usr/bin/env python3
"""
Comprehensive test suite for Social Media Burst Detection project.
Tests all major components and functionality.
"""

import sys
import os

# Add the parent directory to the Python path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from components.data_analyzer import DataAnalyzer
from components.burst_detector import BurstDetector
from components.temporal_clusterer import TemporalClusterer
from components.visualizer import plot_hourly_posts


def test_data_analyzer():
    """Test data loading and preprocessing"""
    print("Testing DataAnalyzer...")
    analyzer = DataAnalyzer('data/sampledata_truthsocial.csv')
    analyzer.run_all()

    assert analyzer.df is not None, "Data loading failed"
    assert len(analyzer.df) > 0, "No data loaded"
    assert analyzer.posts_per_hour is not None, "Hourly aggregation failed"
    print("✅ DataAnalyzer tests passed")
    return analyzer


def test_burst_detector(analyzer):
    """Test burst detection functionality"""
    print("Testing BurstDetector...")
    detector = BurstDetector(s=2.0, gamma=1.0)
    burst_list, posts_with_bursts, contributors = detector.detect_bursts(
        ts_df=analyzer.ts_df,
        posts_per_hour_transformed=analyzer.posts_per_hour_transformed,
        posts_df=analyzer.df
    )

    assert isinstance(burst_list, list), "Burst list should be a list"
    assert len(burst_list) > 0, "No bursts detected"
    print(f"✅ BurstDetector tests passed - {len(burst_list)} bursts detected")
    return burst_list, posts_with_bursts, contributors


def test_temporal_clusterer(analyzer):
    """Test temporal clustering functionality"""
    print("Testing TemporalClusterer...")
    clusterer = TemporalClusterer(min_posts=5)

    # Test feature engineering
    account_features = clusterer.engineer_features(analyzer.df)
    assert account_features is not None, "Feature engineering failed"

    # Test basic clustering
    clusterer.prepare_for_clustering()
    basic_results = clusterer.run_clustering(n_clusters=3)
    assert basic_results is not None, "Basic clustering failed"

    # Test 24-D clustering
    results_24d, personas = clusterer.run_full_24d_analysis(optimal_k_24d=4)
    assert results_24d is not None, "24-D clustering failed"
    assert personas is not None, "Persona validation failed"
    assert len(personas) > 0, "No personas identified"

    print(f"✅ TemporalClusterer tests passed - {len(personas)} personas from {len(results_24d)} accounts")
    return clusterer, basic_results, results_24d, personas


def test_visualizations(analyzer, clusterer):
    """Test visualization generation"""
    print("Testing visualizations...")

    # Create plots directory
    os.makedirs('plots', exist_ok=True)

    # Test basic plots
    plot_hourly_posts(analyzer.posts_per_hour, save_path='plots/test_hourly_posts.png')

    # Test clustering plots
    clusterer.plot_24d_cluster_heatmap('plots/test_heatmap.html')
    clusterer.plot_cluster_fingerprints('plots/test_fingerprints.html')
    clusterer.plot_weekday_weekend_comparison('plots/test_weekday_weekend.html')

    print("✅ Visualization tests passed")


def save_test_results(analyzer, burst_results, clustering_results):
    """Save test results for future use"""
    print("Saving test results...")

    import pickle
    os.makedirs('cache', exist_ok=True)

    # Save analyzer
    with open('cache/analyzer.pkl', 'wb') as f:
        pickle.dump(analyzer, f)

    # Save burst results
    burst_list, posts_with_bursts, contributors = burst_results
    with open('cache/burst_results.pkl', 'wb') as f:
        pickle.dump((burst_list, posts_with_bursts, contributors), f)

    # Save clustering results
    clusterer, basic_results, results_24d, personas = clustering_results
    with open('cache/temporal_clustering_complete.pkl', 'wb') as f:
        pickle.dump({
            'basic_results': basic_results,
            'results_24d': results_24d,
            'personas': personas,
            'account_features': clusterer.account_features
        }, f)

    print("✅ Results saved to cache/")


def main():
    """Run comprehensive test suite"""
    print("🧪 Running Comprehensive Test Suite")
    print("=" * 50)

    try:
        # Test all components
        analyzer = test_data_analyzer()
        burst_results = test_burst_detector(analyzer)
        clustering_results = test_temporal_clusterer(analyzer)
        test_visualizations(analyzer, clustering_results[0])

        # Save results
        save_test_results(analyzer, burst_results, clustering_results)

        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print(f"✅ Burst Detection: {len(burst_results[0])} bursts")
        print(f"✅ Temporal Clustering: {len(clustering_results[3])} personas")
        print("✅ Visualizations generated")
        print("✅ Results cached")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
