#!/usr/bin/env python3
"""
Phase 1 Experiment: Content Similarity Only
Ultra-conservative baseline - only detect identical and highly similar content

This is the baseline experiment that tests ONLY content similarity detection.
All other signals (hashtags, URLs, retweets, temporal sync) are disabled.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import json
from datetime import datetime
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, List
import networkx as nx


class Phase1ContentOnlyDetector:
    """
    Phase 1: Content-only coordination detection

    Detects ONLY:
    - Identical content (100% match after normalization)
    - High similarity content (>85% similarity)

    Does NOT detect:
    - Hashtag coordination
    - URL coordination
    - Retweet amplification
    - Temporal synchronization
    """

    def __init__(self):
        self.identical_threshold = 0.95
        self.high_similarity_threshold = 0.85
        self.min_content_length = 20

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using SequenceMatcher.
        Optimized version from the pipeline.
        """
        if not text1 or not text2:
            return 0.0

        # Normalize: lowercase and remove extra spaces
        text1 = ' '.join(text1.lower().strip().split())
        text2 = ' '.join(text2.lower().strip().split())

        # Quick exact match check
        if text1 == text2:
            return 1.0

        # Quick length difference check - if too different, can't be similar
        len1, len2 = len(text1), len(text2)
        if abs(len1 - len2) / max(len1, len2) > 0.3:  # More than 30% length difference
            return 0.0

        # Use SequenceMatcher for remaining cases
        return SequenceMatcher(None, text1, text2).ratio()

    def _find_identical_content(self, burst_idx: int, burst_posts: pd.DataFrame) -> List[Dict]:
        """
        Find accounts posting identical content (after normalization).
        This is the strongest evidence of coordination.
        """
        evidence = []

        # Group posts by normalized content
        content_groups = defaultdict(list)

        for _, row in burst_posts.iterrows():
            content = row.get('content_cleaned', '') or row.get('content', '')
            account = row.get('account.username', '')

            if not content or len(content.strip()) < self.min_content_length:
                continue

            # Skip retweets - focus on original content
            if content.strip().lower().startswith('rt @'):
                continue

            # Normalize content
            normalized = ' '.join(content.lower().strip().split())

            content_groups[normalized].append({
                'account': account,
                'content': content
            })

        # Find groups with multiple accounts
        for normalized_content, posts in content_groups.items():
            unique_accounts = {post['account'] for post in posts}

            if len(unique_accounts) >= 2:
                # Create pairs of all accounts posting this identical content
                accounts_list = list(unique_accounts)
                for i, acc1 in enumerate(accounts_list):
                    for acc2 in accounts_list[i+1:]:
                        evidence.append({
                            'type': 'identical_content',
                            'burst_index': burst_idx,
                            'account1': acc1,
                            'account2': acc2,
                            'similarity': 1.0,
                            'confidence': 1.0
                        })

        return evidence

    def _find_high_similarity(self, burst_idx: int, burst_posts: pd.DataFrame) -> List[Dict]:
        """
        Find accounts posting highly similar (but not identical) content.
        """
        evidence = []

        # Get posts by account
        account_posts = defaultdict(list)

        for _, row in burst_posts.iterrows():
            content = row.get('content_cleaned', '') or row.get('content', '')
            account = row.get('account.username', '')

            if not content or len(content.strip()) < self.min_content_length:
                continue

            # Skip retweets
            if content.strip().lower().startswith('rt @'):
                continue

            account_posts[account].append(content.strip())

        # Compare posts between different accounts
        accounts = list(account_posts.keys())

        # Limit comparisons for performance
        max_accounts_to_compare = 100
        if len(accounts) > max_accounts_to_compare:
            print(f"   Burst {burst_idx}: Limiting comparison to {max_accounts_to_compare} accounts (of {len(accounts)})")
            accounts = accounts[:max_accounts_to_compare]

        for i, account1 in enumerate(accounts):
            for account2 in accounts[i+1:]:
                posts1 = account_posts[account1]
                posts2 = account_posts[account2]

                # Compare first 3 posts from each account (performance)
                for content1 in posts1[:3]:
                    for content2 in posts2[:3]:
                        similarity = self._calculate_text_similarity(content1, content2)

                        if similarity >= self.high_similarity_threshold:
                            evidence.append({
                                'type': 'high_similarity',
                                'burst_index': burst_idx,
                                'account1': account1,
                                'account2': account2,
                                'similarity': similarity,
                                'confidence': similarity
                            })
                            break  # Found similarity, move to next account pair
                    else:
                        continue
                    break  # Break outer loop if inner broke

        return evidence

    def detect_coordination(self, burst_contributors: List[Dict], full_posts_df) -> Dict:
        """
        Phase 1 detection: Content similarity only
        """
        print("\n" + "="*60)
        print("PHASE 1: CONTENT SIMILARITY ONLY")
        print("="*60)
        print("Enabled:  ✓ Identical content")
        print("          ✓ High similarity (>85%)")
        print("Disabled: ✗ Hashtag coordination")
        print("          ✗ URL coordination")
        print("          ✗ Retweet amplification")
        print("          ✗ Temporal synchronization")
        print("="*60)

        results = {
            'phase': 'Phase 1',
            'description': 'Content Similarity Only (Ultra-Conservative Baseline)',
            'enabled_features': ['identical_content', 'high_similarity'],
            'disabled_features': ['hashtag_coordination', 'url_coordination',
                                 'retweet_amplification', 'temporal_sync'],
            'coordination_pairs': [],
            'networks': [],
            'statistics': {}
        }

        # Convert to pandas
        if hasattr(full_posts_df, 'to_pandas'):
            posts_df = full_posts_df.to_pandas()
        else:
            posts_df = full_posts_df

        # Ensure datetime column and timezone compatibility
        time_col = 'created_at' if 'created_at' in posts_df.columns else 'post_timestamp'
        posts_df[time_col] = pd.to_datetime(posts_df[time_col])

        if posts_df[time_col].dt.tz is not None:
            posts_df[time_col] = posts_df[time_col].dt.tz_convert('UTC').dt.tz_localize(None)

        # Process each burst
        all_evidence = []
        total_bursts = len(burst_contributors)

        print(f"\nProcessing {total_bursts} bursts for content coordination...")

        for burst_idx, contrib in enumerate(burst_contributors):
            if burst_idx % 10 == 0:
                print(f"   Processing burst {burst_idx}/{total_bursts}...")

            start_time = contrib['start_time']
            end_time = contrib['end_time']

            # Ensure timezone compatibility
            if hasattr(start_time, 'tzinfo') and start_time.tzinfo is not None:
                start_time = start_time.replace(tzinfo=None)
            if hasattr(end_time, 'tzinfo') and end_time.tzinfo is not None:
                end_time = end_time.replace(tzinfo=None)

            start_time = pd.to_datetime(start_time)
            end_time = pd.to_datetime(end_time)

            # Get posts in this burst window
            burst_mask = ((posts_df[time_col] >= start_time) &
                         (posts_df[time_col] <= end_time))
            burst_posts = posts_df[burst_mask].copy()

            if len(burst_posts) < 2:
                continue

            # Filter to significant contributors only (from burst detection)
            significant_accounts = {acc['account.username'] for acc in contrib.get('top_accounts', [])}
            burst_posts = burst_posts[burst_posts['account.username'].isin(significant_accounts)]

            if len(burst_posts) < 2:
                continue

            # Phase 1: Find identical content
            identical_evidence = self._find_identical_content(burst_idx, burst_posts)
            all_evidence.extend(identical_evidence)

            # Phase 1: Find high similarity content
            similarity_evidence = self._find_high_similarity(burst_idx, burst_posts)
            all_evidence.extend(similarity_evidence)

        print(f"\n✓ Found {len(all_evidence)} coordination pairs")

        # Build networks from coordination pairs
        print("Building coordination networks...")

        if all_evidence:
            G = nx.Graph()

            for evidence in all_evidence:
                G.add_edge(
                    evidence['account1'],
                    evidence['account2'],
                    weight=evidence['confidence'],
                    evidence_type=evidence['type']
                )

            print(f"   Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

            # Extract networks (connected components)
            networks = []
            for i, component in enumerate(nx.connected_components(G), 1):
                if len(component) >= 2:
                    networks.append({
                        'network_id': i,
                        'accounts': sorted(list(component)),
                        'size': len(component),
                        'evidence': 'content_similarity_only'
                    })

            print(f"   Networks: {len(networks)} detected")
        else:
            networks = []
            print("   No networks detected")

        # Calculate statistics
        unique_accounts = set()
        for evidence in all_evidence:
            unique_accounts.add(evidence['account1'])
            unique_accounts.add(evidence['account2'])

        identical_count = len([e for e in all_evidence if e['type'] == 'identical_content'])
        similarity_count = len([e for e in all_evidence if e['type'] == 'high_similarity'])

        results['coordination_pairs'] = all_evidence
        results['networks'] = networks
        results['statistics'] = {
            'total_coordination_pairs': len(all_evidence),
            'unique_coordinated_accounts': len(unique_accounts),
            'networks_detected': len(networks),
            'identical_content_pairs': identical_count,
            'high_similarity_pairs': similarity_count
        }

        # Print summary
        print("\n" + "="*60)
        print("PHASE 1 RESULTS")
        print("="*60)
        stats = results['statistics']
        print(f"   Total coordination pairs: {stats['total_coordination_pairs']}")
        print(f"   └─ Identical content: {stats['identical_content_pairs']}")
        print(f"   └─ High similarity: {stats['high_similarity_pairs']}")
        print(f"   Unique coordinated accounts: {stats['unique_coordinated_accounts']}")
        print(f"   Coordination networks: {stats['networks_detected']}")

        if networks:
            print(f"\n   Top 5 Networks (by size):")
            sorted_networks = sorted(networks, key=lambda x: x['size'], reverse=True)[:5]
            for net in sorted_networks:
                print(f"      Network {net['network_id']}: {net['size']} accounts")

        print("="*60)
        print("Phase 1 Complete - Content-only baseline established ✓")
        print("="*60)

        return results


def main():
    """Run Phase 1 experiment"""

    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  PHASE 1 EXPERIMENT: CONTENT SIMILARITY ONLY".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")

    # Load data
    print("\n📊 STEP 1: Loading data...")
    from components.data_analyzer import DataAnalyzer
    from components.burst_detector_enhanced import BurstDetectorEnhanced

    data_path = os.path.join('data', 'sampledata_truthsocial.csv')
    if not os.path.exists(data_path):
        print(f"❌ ERROR: Data file not found at {data_path}")
        return None

    analyzer = DataAnalyzer(data_path)
    print("   Running data analysis...")
    analyzer.run_all()
    print("   ✓ Data analysis complete")

    # Detect bursts
    print("\n💥 STEP 2: Detecting bursts...")
    detector = BurstDetectorEnhanced(s=2.0, gamma=1.0)
    print("   Running Kleinberg burst detection...")

    burst_list, posts_with_bursts, burst_contributors = detector.detect_bursts(
        analyzer.ts_df,
        analyzer.posts_per_hour_transformed,
        analyzer.df
    )

    print(f"   ✓ Found {len(burst_list)} bursts")
    print(f"   ✓ Identified {len(burst_contributors)} burst periods with contributors")

    # Run Phase 1 detection
    print("\n🕵️ STEP 3: Running Phase 1 coordination detection...")
    phase1_detector = Phase1ContentOnlyDetector()
    results = phase1_detector.detect_coordination(burst_contributors, analyzer.df)

    # Save results
    print("\n💾 STEP 4: Saving results...")
    os.makedirs('experiments/results', exist_ok=True)

    output_file = 'experiments/results/phase1_results.json'

    # Prepare JSON-serializable output
    json_results = {
        'experiment': 'Phase 1 - Content Similarity Only',
        'timestamp': datetime.now().isoformat(),
        'phase': results['phase'],
        'description': results['description'],
        'enabled_features': results['enabled_features'],
        'disabled_features': results['disabled_features'],
        'statistics': results['statistics'],
        'networks': results['networks'],
        'coordination_pairs': results['coordination_pairs']  # Save all pairs
    }

    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"   ✓ Results saved to: {output_file}")

    # Summary
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  PHASE 1 EXPERIMENT COMPLETE".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")

    stats = results['statistics']
    print(f"\n📊 Summary:")
    print(f"   • Total pairs: {stats['total_coordination_pairs']}")
    print(f"   • Coordinated accounts: {stats['unique_coordinated_accounts']}")
    print(f"   • Networks detected: {stats['networks_detected']}")
    print(f"\n💡 This establishes the baseline. Next phases will add:")
    print(f"   → Phase 2: Hashtag & URL coordination")
    print(f"   → Phase 3: Retweet amplification")
    print(f"   → Phase 4: Temporal synchronization")

    return results


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Experiment interrupted by user")
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()

