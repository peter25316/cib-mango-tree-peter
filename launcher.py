#!/usr/bin/env python3
"""
CIB Mango Tree Launcher
Main entry point for running the unified analysis pipeline
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def print_header():
    """Print application header"""
    print("\n" + "="*70)
    print("🥭 CIB MANGO TREE - Social Media Coordination Analysis")
    print("="*70)

def print_menu():
    """Display the main menu options"""
    print("\n📋 MAIN MENU:")
    print("-" * 70)
    print("\n🚀 ANALYSIS:")
    print("  1. Run Complete Analysis Pipeline (with caching)")
    print("  2. Run Complete Analysis (force recompute)")
    print("  3. View Analysis Results Summary")
    print()
    print("📊 REPORTS:")
    print("  4. Generate Coordination Report")
    print("  5. Export Results to JSON")
    print()
    print("🎨 VISUALIZATIONS:")
    print("  6. Generate All Visualizations")
    print("  7. View Network Visualizations")
    print()
    print("🎭 INTERACTIVE:")
    print("  8. Launch Interactive Dashboard (Streamlit)")
    print()
    print("🛠️ UTILITIES:")
    print("  9. View Performance Stats")
    print("  10. Clear Cache")
    print("  11. System Status Check")
    print()
    print("  0. Exit")
    print("-" * 70)

def run_complete_analysis(force_rerun=False):
    """Run the complete analysis pipeline"""
    print("\n🚀 Starting Complete Analysis Pipeline...")
    print("-" * 70)

    try:
        from unified_pipeline import UnifiedAnalysisPipeline

        # Initialize pipeline
        pipeline = UnifiedAnalysisPipeline('data/sampledata_truthsocial.csv')

        # Set burst detection parameters
        burst_params = {
            's': 2.0,
            'gamma': 1.0
        }

        # Run pipeline
        success = pipeline.run_complete_pipeline(burst_params, force_rerun=force_rerun)

        if success:
            print("\n✅ Pipeline completed successfully!")
            return pipeline
        else:
            print("\n❌ Pipeline failed!")
            return None

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def view_results_summary():
    """View a summary of existing results"""
    print("\n📊 Loading Results Summary...")
    print("-" * 70)

    try:
        from unified_pipeline import UnifiedAnalysisPipeline

        pipeline = UnifiedAnalysisPipeline('data/sampledata_truthsocial.csv')

        if pipeline.load_cached_results():
            pipeline.print_summary()
            return pipeline
        else:
            print("❌ No cached results found. Run analysis first (Option 1 or 2).")
            return None

    except Exception as e:
        print(f"❌ Error: {e}")
        return None



def generate_coordination_report():
    """Generate detailed coordination analysis report"""
    print("\n📊 Generating Coordination Analysis Report...")
    print("-" * 70)

    try:
        from unified_pipeline import UnifiedAnalysisPipeline

        pipeline = UnifiedAnalysisPipeline('data/sampledata_truthsocial.csv')

        # Load cached results
        if not pipeline.load_cached_results():
            print("❌ No cached results found. Run analysis first.")
            return

        # Get coordination results
        coord_results = pipeline.results.get('coordination_analysis', {})

        if not coord_results:
            print("❌ No coordination analysis results found.")
            return

        # Display coordination summary
        print("\n🕵️ COORDINATION ANALYSIS RESULTS")
        print("="*70)

        stats = coord_results.get('summary_stats', {})
        networks = coord_results.get('coordination_networks', [])

        print(f"\n📋 OVERALL STATISTICS:")
        print(f"   Total Coordination Pairs: {stats.get('total_coordination_pairs', 0)}")
        print(f"   Identical Content Instances: {stats.get('identical_content_instances', 0)}")
        print(f"   Hashtag Coordination: {stats.get('hashtag_coordination_instances', 0)}")
        print(f"   URL Coordination: {stats.get('url_coordination_instances', 0)}")
        print(f"   RT Amplification: {stats.get('retweet_amplification_instances', 0)}")
        print(f"   Confidence Level: {coord_results.get('confidence_level', 'UNKNOWN')}")

        print(f"\n🕸️ COORDINATION NETWORKS:")
        print(f"   Total Networks: {stats.get('total_networks', 0)}")
        print(f"   Total Network Accounts: {stats.get('total_network_accounts', 0)}")
        print(f"   High Risk Networks: {stats.get('high_risk_networks', 0)}")

        if networks:
            print(f"\n🏆 TOP 5 NETWORKS:")
            for i, net in enumerate(networks[:5], 1):
                print(f"   {i}. Network {net['network_id']}: {net['size']} accounts, "
                      f"Risk={net['risk_level']}, Confidence={net['avg_confidence']:.3f}")
                print(f"      Evidence: {', '.join(net['evidence_types'])}")

        # Save detailed report
        os.makedirs('reports', exist_ok=True)
        report_file = 'reports/coordination_analysis_report.json'

        # Prepare JSON-serializable version (remove NetworkX graphs)
        exportable_results = {
            'summary_stats': stats,
            'confidence_level': coord_results.get('confidence_level', 'UNKNOWN'),
            'networks': [
                {
                    'network_id': net['network_id'],
                    'size': net['size'],
                    'accounts': net['accounts'],
                    'avg_confidence': net['avg_confidence'],
                    'risk_level': net['risk_level'],
                    'evidence_types': net['evidence_types'],
                    'network_metrics': net.get('network_metrics', {})
                }
                for net in networks
            ]
        }

        with open(report_file, 'w') as f:
            json.dump(exportable_results, f, indent=2, default=str)

        print(f"\n💾 Detailed report saved to: {report_file}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def export_results():
    """Export all results to JSON files"""
    print("\n💾 Exporting Results...")
    print("-" * 70)

    try:
        from unified_pipeline import UnifiedAnalysisPipeline

        pipeline = UnifiedAnalysisPipeline('data/sampledata_truthsocial.csv')

        if not pipeline.load_cached_results():
            print("❌ No cached results found. Run analysis first.")
            return

        os.makedirs('reports', exist_ok=True)

        # Export timing stats
        timing_file = 'reports/timing_stats.json'
        with open(timing_file, 'w') as f:
            json.dump(pipeline.timing, f, indent=2)
        print(f"   ✅ Timing stats: {timing_file}")

        # Export data analysis results
        if 'data_analysis' in pipeline.results:
            data_file = 'reports/data_analysis.json'
            with open(data_file, 'w') as f:
                json.dump(pipeline.results['data_analysis'], f, indent=2, default=str)
            print(f"   ✅ Data analysis: {data_file}")

        # Export burst detection results (without contributors detail)
        enhanced_key = [k for k in pipeline.results.keys() if 'burst_detection_enhanced' in k]
        if enhanced_key:
            burst_results = pipeline.results[enhanced_key[0]]
            burst_summary = {
                'burst_count': burst_results.get('burst_count', 0),
                'total_contributors': burst_results.get('total_contributors', 0),
                'repeated_contributors': burst_results.get('repeated_contributors', 0),
                'coordination_signals': burst_results.get('coordination_signals', 0)
            }
            burst_file = 'reports/burst_detection_summary.json'
            with open(burst_file, 'w') as f:
                json.dump(burst_summary, f, indent=2)
            print(f"   ✅ Burst detection: {burst_file}")

        print(f"\n✅ Export complete! Check the 'reports' directory.")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def generate_visualizations():
    """Generate all visualizations"""
    print("\n🎨 Generating Visualizations...")
    print("-" * 70)

    try:
        from unified_pipeline import UnifiedAnalysisPipeline

        pipeline = UnifiedAnalysisPipeline('data/sampledata_truthsocial.csv')

        if not pipeline.load_cached_results():
            print("❌ No cached results found. Running analysis first...")
            success = pipeline.run_complete_pipeline()
            if not success:
                print("❌ Analysis failed!")
                return

        # Generate visualizations
        success = pipeline.generate_visualizations(force_rerun=True)

        if success:
            print("\n✅ All visualizations generated!")
            print(f"📁 Check the 'plots' directory for all charts and graphs.")
        else:
            print("\n⚠️ Some visualizations may have failed. Check the output above.")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def view_network_visualizations():
    """Display information about network visualizations"""
    print("\n🕸️ Network Visualizations...")
    print("-" * 70)

    plots_dir = Path('plots')

    if not plots_dir.exists():
        print("❌ No plots directory found. Generate visualizations first (Option 7).")
        return

    # Find network visualization files
    network_plots = list(plots_dir.glob('network_*_visualization.png'))

    if not network_plots:
        print("❌ No network visualizations found. Generate them first (Option 7).")
        return

    print(f"\n📊 Found {len(network_plots)} network visualizations:")
    for plot in sorted(network_plots):
        print(f"   • {plot.name}")

    print(f"\n💡 Open these PNG files in your image viewer to see the network graphs.")
    print(f"📁 Location: {plots_dir.absolute()}")

def launch_streamlit_demo():
    """Launch interactive Streamlit dashboard"""
    print("\n🎭 Launching Interactive Dashboard...")
    print("-" * 70)
    print("⏳ Starting Streamlit server...")
    print("🌐 The dashboard will open in your web browser automatically.")
    print("⚠️ Press Ctrl+C in the terminal to stop the server when done.")
    print()

    import subprocess
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run',
            'demo/interactive_burst_app_v2.py',
            '--server.port', '8504'
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped.")

def view_performance_stats():
    """View performance statistics from cached results"""
    print("\n📈 Performance Statistics...")
    print("-" * 70)

    try:
        import pickle

        cache_file = 'cache/unified_pipeline_cache.pkl'
        if not os.path.exists(cache_file):
            print("❌ No cached results found. Run analysis first.")
            return

        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)

        timing = cached.get('timing', {})
        results = cached.get('results', {})

        print("\n⏱️ TIMING BREAKDOWN:")
        total_time = timing.get('total_pipeline', 0)

        if total_time > 0:
            for step, time_val in sorted(timing.items()):
                if step != 'total_pipeline' and time_val > 0:
                    percentage = (time_val / total_time * 100)
                    print(f"   {step.replace('_', ' ').title()}: {time_val:.2f}s ({percentage:.1f}%)")

            print(f"\n   {'TOTAL'}: {total_time:.2f}s (100%)")
        else:
            print("   No timing data available.")

        # Results summary
        print("\n📊 RESULTS SUMMARY:")

        if 'data_analysis' in results:
            data = results['data_analysis']
            print(f"   Posts Processed: {data.get('posts_count', 0):,}")

        enhanced_key = [k for k in results.keys() if 'burst_detection_enhanced' in k]
        if enhanced_key:
            burst = results[enhanced_key[0]]
            print(f"   Bursts Detected: {burst.get('burst_count', 0)}")
            print(f"   Contributors: {burst.get('total_contributors', 0)}")

        if 'coordination_analysis' in results:
            coord = results['coordination_analysis']
            stats = coord.get('summary_stats', {})
            print(f"   Coordination Networks: {stats.get('total_networks', 0)}")
            print(f"   Network Accounts: {stats.get('total_network_accounts', 0)}")

    except Exception as e:
        print(f"❌ Error: {e}")

def clear_cache():
    """Clear all cached results"""
    print("\n🧹 Clearing Cache...")
    print("-" * 70)

    import glob

    cache_files = glob.glob('cache/*.pkl')

    if not cache_files:
        print("   No cache files found.")
        return

    removed_count = 0
    for cache_file in cache_files:
        try:
            os.remove(cache_file)
            print(f"   ✅ Removed: {cache_file}")
            removed_count += 1
        except Exception as e:
            print(f"   ❌ Failed to remove {cache_file}: {e}")

    print(f"\n✅ Cache cleared: {removed_count} file(s) removed")

def check_system_status():
    """Check system requirements and status"""
    print("\n🔧 System Status Check...")
    print("-" * 70)

    # Python version
    print(f"\n🐍 Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    # Required packages
    required_packages = [
        'polars', 'pandas', 'numpy', 'matplotlib', 'plotly',
        'streamlit', 'statsmodels', 'scikit-learn', 'networkx'
    ]

    print("\n📦 Required Packages:")
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (MISSING)")
            missing_packages.append(package)

    # Data file
    print("\n📁 Data Files:")
    data_file = 'data/sampledata_truthsocial.csv'
    if os.path.exists(data_file):
        file_size = os.path.getsize(data_file) / (1024 * 1024)
        print(f"   ✅ {data_file} ({file_size:.1f} MB)")
    else:
        print(f"   ❌ {data_file} (MISSING)")

    # Cache
    print("\n💾 Cache Status:")
    if os.path.exists('cache'):
        cache_files = len([f for f in os.listdir('cache') if f.endswith('.pkl')])
        print(f"   📦 {cache_files} cached result(s) available")
    else:
        print(f"   📦 No cache directory")

    # Plots
    print("\n📊 Visualizations:")
    if os.path.exists('plots'):
        plot_files = len([f for f in os.listdir('plots') if f.endswith(('.png', '.html'))])
        print(f"   🎨 {plot_files} visualization(s) generated")
    else:
        print(f"   🎨 No plots directory")

    # Summary
    if missing_packages:
        print(f"\n⚠️ MISSING PACKAGES: {', '.join(missing_packages)}")
        print("   Install with: pip install -r requirements.txt")
    else:
        print(f"\n✅ All requirements satisfied!")

def main():
    """Main launcher function"""
    print_header()

    while True:
        print_menu()

        try:
            choice = input("\n👉 Enter your choice (0-11): ").strip()

            if choice == '0':
                print("\n👋 Thank you for using CIB Mango Tree!")
                print("="*70)
                break

            elif choice == '1':
                run_complete_analysis(force_rerun=False)

            elif choice == '2':
                run_complete_analysis(force_rerun=True)

            elif choice == '3':
                view_results_summary()

            elif choice == '4':
                generate_coordination_report()

            elif choice == '5':
                export_results()

            elif choice == '6':
                generate_visualizations()

            elif choice == '7':
                view_network_visualizations()

            elif choice == '8':
                launch_streamlit_demo()

            elif choice == '9':
                view_performance_stats()

            elif choice == '10':
                clear_cache()

            elif choice == '11':
                check_system_status()

            else:
                print("❌ Invalid choice. Please enter a number between 0-11.")

        except KeyboardInterrupt:
            print("\n\n👋 Thank you for using CIB Mango Tree!")
            print("="*70)
            break

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

        # Pause before showing menu again (except for Streamlit)
        if choice not in ['8', '0']:
            input("\n⏸️  Press Enter to continue...")

if __name__ == "__main__":
    main()

