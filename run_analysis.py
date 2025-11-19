#!/usr/bin/env python3
"""
Main analysis runner for Social Media Burst Detection project.
Provides options to run different types of analysis with clean output.
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from components.data_analyzer import DataAnalyzer
from components.burst_detector import BurstDetector
from components.temporal_clusterer import TemporalClusterer


def run_burst_analysis(data_path='data/sampledata_truthsocial.csv', s_param=2.0, gamma_param=1.0):
    """Run burst detection analysis"""
    print("=== Running Burst Detection Analysis ===")
    
    # Load data
    analyzer = DataAnalyzer(data_path)
    analyzer.run_all()
    
    # Run burst detection
    detector = BurstDetector(s=s_param, gamma=gamma_param)
    burst_list, posts_with_bursts, contributors = detector.detect_bursts(
        ts_df=analyzer.ts_df,
        posts_per_hour_transformed=analyzer.posts_per_hour_transformed,
        posts_df=analyzer.df
    )
    
    print(f"✅ Detected {len(burst_list)} bursts")
    
    # Save results
    import pickle
    os.makedirs('cache', exist_ok=True)
    with open('cache/burst_results.pkl', 'wb') as f:
        pickle.dump((burst_list, posts_with_bursts, contributors), f)
    
    return burst_list, posts_with_bursts, contributors


def run_temporal_clustering(data_path='data/sampledata_truthsocial.csv', min_posts=5, k_clusters=4):
    """Run temporal clustering analysis"""
    print("=== Running Temporal Clustering Analysis ===")
    
    # Load data
    analyzer = DataAnalyzer(data_path)
    analyzer.run_all()
    
    # Run clustering
    clusterer = TemporalClusterer(min_posts=min_posts)
    clusterer.engineer_features(analyzer.df)
    
    # Run full 24-D analysis
    results, personas = clusterer.run_full_24d_analysis(optimal_k_24d=k_clusters)
    
    print(f"✅ Identified {len(personas)} temporal personas from {len(results)} accounts")
    
    # Save results
    import pickle
    os.makedirs('cache', exist_ok=True)
    results_data = {
        'cluster_results_24d': results,
        'cluster_personas': personas,
        'account_features': clusterer.account_features
    }
    with open('cache/temporal_clustering_complete.pkl', 'wb') as f:
        pickle.dump(results_data, f)
    
    return results, personas


def generate_visualizations():
    """Generate all visualizations from cached results"""
    print("=== Generating Visualizations ===")
    
    os.makedirs('plots', exist_ok=True)
    
    # Load temporal clustering results and generate plots
    try:
        import pickle
        with open('cache/temporal_clustering_complete.pkl', 'rb') as f:
            temporal_data = pickle.load(f)
        
        # Recreate clusterer object for plotting
        analyzer = DataAnalyzer('data/sampledata_truthsocial.csv')
        analyzer.run_all()
        
        clusterer = TemporalClusterer(min_posts=5)
        clusterer.engineer_features(analyzer.df)
        clusterer.run_full_24d_analysis(optimal_k_24d=4)
        
        # Generate key plots
        clusterer.plot_24d_cluster_heatmap('plots/24d_cluster_heatmap.html')
        clusterer.plot_weekday_weekend_comparison('plots/weekday_weekend_activity_fingerprint.html')
        clusterer.plot_cluster_fingerprints('plots/24hour_fingerprint_4cluster.html')
        
        print("✅ Temporal clustering visualizations generated")
        
    except FileNotFoundError:
        print("⚠️ No temporal clustering cache found. Run temporal clustering first.")
    
    # Load burst results and generate plots if available
    try:
        with open('cache/burst_results.pkl', 'rb') as f:
            burst_list, posts_with_bursts, contributors = pickle.load(f)
        
        print("✅ Burst detection visualizations available in plots/")
        
    except FileNotFoundError:
        print("⚠️ No burst detection cache found. Run burst analysis first.")


def main():
    parser = argparse.ArgumentParser(description='Social Media Burst Detection Analysis')
    parser.add_argument('mode', choices=['burst', 'temporal', 'plots', 'all'], 
                       help='Analysis mode to run')
    parser.add_argument('--data', default='data/sampledata_truthsocial.csv',
                       help='Path to data file')
    parser.add_argument('--s', type=float, default=2.0,
                       help='Burst detection S parameter')
    parser.add_argument('--gamma', type=float, default=1.0,
                       help='Burst detection Gamma parameter')
    parser.add_argument('--min-posts', type=int, default=5,
                       help='Minimum posts for temporal clustering')
    parser.add_argument('--clusters', type=int, default=4,
                       help='Number of clusters for temporal analysis')
    
    args = parser.parse_args()
    
    print(f"🚀 Social Media Burst Detection Analysis")
    print(f"Mode: {args.mode}")
    print(f"Data: {args.data}")
    
    if args.mode == 'burst' or args.mode == 'all':
        run_burst_analysis(args.data, args.s, args.gamma)
    
    if args.mode == 'temporal' or args.mode == 'all':
        run_temporal_clustering(args.data, args.min_posts, args.clusters)
    
    if args.mode == 'plots' or args.mode == 'all':
        generate_visualizations()
    
    print("\n🎉 Analysis complete!")
    print("\nOutput locations:")
    print("  📊 Plots: plots/")
    print("  💾 Cache: cache/")
    print("  📋 Results: results/")


if __name__ == "__main__":
    main()
