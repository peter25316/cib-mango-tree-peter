#!/usr/bin/env python3
"""
Phase 4 Experiment: Add Temporal Synchronization
Adds ultra-conservative temporal synchronization detection to Phase 3 signals

This phase tests content + hashtag + URL + RT + TEMPORAL SYNCHRONIZATION.
Ultra-conservative: only flags obvious synchronized posting (≤30 seconds, 3+ posts).
Expected to add minimal detections (<2%) as Phase 3 already caught 98.7%.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import json
import re
from datetime import datetime
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Optional
import networkx as nx


class Phase4TemporalDetector:
    """
    Phase 4: Complete Multi-Signal Detection

    Detects:
    - Identical content (from Phase 1)
    - High similarity content (from Phase 1)
    - Hashtag coordination (from Phase 2)
    - URL coordination (from Phase 2)
    - Retweet amplification (from Phase 3)
    - Temporal synchronization ⭐ NEW (ultra-conservative)

    This is the COMPLETE coordination detection pipeline.
    """

    def __init__(self):
        self.identical_threshold = 0.95
        self.high_similarity_threshold = 0.85
        self.min_content_length = 20
        self.min_hashtag_overlap = 2
        self.hashtag_jaccard_threshold = 0.6
        self.min_retweeters = 3
        self.temporal_sync_threshold = 30  # Ultra-conservative: 30 seconds
        self.min_synchronized_posts = 3    # Require 3+ synchronized posts
        self.temporal_confidence_threshold = 0.8  # High confidence required

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity (from Phase 1)."""
        if not text1 or not text2:
            return 0.0

        text1 = ' '.join(text1.lower().strip().split())
        text2 = ' '.join(text2.lower().strip().split())

        if text1 == text2:
            return 1.0

        len1, len2 = len(text1), len(text2)
        if abs(len1 - len2) / max(len1, len2) > 0.3:
            return 0.0

        return SequenceMatcher(None, text1, text2).ratio()

    def _extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags (from Phase 2)."""
        if not text:
            return []
        return re.findall(r'#\w+', text.lower())

    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs (from Phase 2)."""
        if not text:
            return []
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.findall(url_pattern, text)

    def _extract_rt_source(self, content: str) -> Optional[str]:
        """Extract RT source (from Phase 3)."""
        rt_pattern = r'^RT\s+@(\w+)'
        match = re.match(rt_pattern, content.strip(), re.IGNORECASE)

        if match:
            return match.group(1).lower()
        return None

    def _find_identical_content(self, burst_idx: int, burst_posts: pd.DataFrame) -> List[Dict]:
        """Find identical content (from Phase 1)."""
        evidence = []
        content_groups = defaultdict(list)

        for _, row in burst_posts.iterrows():
            content = row.get('content_cleaned', '') or row.get('content', '')
            account = row.get('account.username', '')

            if not content or len(content.strip()) < self.min_content_length:
                continue

            if content.strip().lower().startswith('rt @'):
                continue

            normalized = ' '.join(content.lower().strip().split())
            content_groups[normalized].append({'account': account, 'content': content})

        for normalized_content, posts in content_groups.items():
            unique_accounts = {post['account'] for post in posts}

            if len(unique_accounts) >= 2:
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
        """Find high similarity content (from Phase 1)."""
        evidence = []
        account_posts = defaultdict(list)

        for _, row in burst_posts.iterrows():
            content = row.get('content_cleaned', '') or row.get('content', '')
            account = row.get('account.username', '')

            if not content or len(content.strip()) < self.min_content_length:
                continue

            if content.strip().lower().startswith('rt @'):
                continue

            account_posts[account].append(content.strip())

        accounts = list(account_posts.keys())

        max_accounts_to_compare = 100
        if len(accounts) > max_accounts_to_compare:
            accounts = accounts[:max_accounts_to_compare]

        for i, account1 in enumerate(accounts):
            for account2 in accounts[i+1:]:
                posts1 = account_posts[account1]
                posts2 = account_posts[account2]

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
                            break
                    else:
                        continue
                    break

        return evidence

    def _find_hashtag_coordination(self, burst_idx: int, burst_posts: pd.DataFrame) -> List[Dict]:
        """Find hashtag coordination (from Phase 2)."""
        evidence = []
        account_hashtags = defaultdict(list)

        for _, row in burst_posts.iterrows():
            content = row.get('content_cleaned', '') or row.get('content', '')
            account = row.get('account.username', '')

            if not content:
                continue

            if content.strip().lower().startswith('rt @'):
                continue

            hashtags = self._extract_hashtags(content)
            if hashtags:
                account_hashtags[account].extend(hashtags)

        accounts = list(account_hashtags.keys())

        for i, account1 in enumerate(accounts):
            for account2 in accounts[i+1:]:

                hashtags1 = set(account_hashtags[account1])
                hashtags2 = set(account_hashtags[account2])

                if len(hashtags1) >= 2 and len(hashtags2) >= 2:
                    intersection = hashtags1 & hashtags2
                    union = hashtags1 | hashtags2

                    if len(intersection) >= self.min_hashtag_overlap:
                        jaccard_similarity = len(intersection) / len(union)

                        if jaccard_similarity >= self.hashtag_jaccard_threshold:
                            evidence.append({
                                'type': 'hashtag_coordination',
                                'burst_index': burst_idx,
                                'account1': account1,
                                'account2': account2,
                                'shared_hashtags': list(intersection),
                                'jaccard_similarity': jaccard_similarity,
                                'confidence': min(jaccard_similarity * 1.5, 1.0),
                                'evidence_strength': 'HIGH' if jaccard_similarity > 0.8 else 'MEDIUM'
                            })

        return evidence

    def _find_url_coordination(self, burst_idx: int, burst_posts: pd.DataFrame) -> List[Dict]:
        """Find URL coordination (from Phase 2)."""
        evidence = []
        account_urls = defaultdict(set)

        for _, row in burst_posts.iterrows():
            content = row.get('content_cleaned', '') or row.get('content', '')
            account = row.get('account.username', '')

            if not content:
                continue

            if content.strip().lower().startswith('rt @'):
                continue

            urls = self._extract_urls(content)
            if urls:
                account_urls[account].update(urls)

        accounts = list(account_urls.keys())

        for i, account1 in enumerate(accounts):
            for account2 in accounts[i+1:]:

                urls1 = account_urls[account1]
                urls2 = account_urls[account2]

                shared_urls = urls1 & urls2

                if len(shared_urls) >= 1:
                    confidence = min(len(shared_urls) * 0.8, 1.0)

                    evidence.append({
                        'type': 'url_coordination',
                        'burst_index': burst_idx,
                        'account1': account1,
                        'account2': account2,
                        'shared_urls': list(shared_urls),
                        'confidence': confidence,
                        'evidence_strength': 'HIGH' if len(shared_urls) > 1 else 'MEDIUM'
                    })

        return evidence

    def _find_retweet_coordination(self, burst_idx: int, burst_posts: pd.DataFrame) -> List[Dict]:
        """Find retweet coordination (from Phase 3)."""
        evidence = []
        rt_sources = defaultdict(lambda: {
            'retweeters': set(),
            'content': None
        })

        for _, row in burst_posts.iterrows():
            content = row.get('content_cleaned', '') or row.get('content', '')
            retweeter = row.get('account.username', '')

            if not content:
                continue

            if content.strip().lower().startswith('rt @'):
                rt_source = self._extract_rt_source(content)
                if rt_source:
                    rt_sources[rt_source]['retweeters'].add(retweeter)
                    if rt_sources[rt_source]['content'] is None:
                        rt_sources[rt_source]['content'] = content

        for source, data in rt_sources.items():
            retweeters = data['retweeters']

            if len(retweeters) >= self.min_retweeters:
                retweeter_list = list(retweeters)
                amplification_strength = min(len(retweeters) / 10.0, 1.0)

                if len(retweeters) >= 10:
                    evidence_strength = 'VERY_HIGH'
                elif len(retweeters) >= 5:
                    evidence_strength = 'HIGH'
                else:
                    evidence_strength = 'MEDIUM'

                for i, account1 in enumerate(retweeter_list):
                    for account2 in retweeter_list[i+1:]:
                        evidence.append({
                            'type': 'retweet_amplification',
                            'burst_index': burst_idx,
                            'account1': account1,
                            'account2': account2,
                            'rt_source': source,
                            'amplification_count': len(retweeters),
                            'confidence': amplification_strength,
                            'evidence_strength': evidence_strength
                        })

        return evidence

    def _find_temporal_synchronization(self, burst_idx: int, burst_posts: pd.DataFrame, time_col: str) -> List[Dict]:
        """
        Find ultra-conservative temporal synchronization.
        Based on unified pipeline implementation.

        Only flags accounts posting within 30 seconds with 3+ synchronized posts.
        This catches only the most obvious coordinated timing patterns.
        """
        evidence = []
        sync_threshold = pd.Timedelta(seconds=self.temporal_sync_threshold)

        # Group posts by account with timestamps
        account_times = defaultdict(list)

        for _, row in burst_posts.iterrows():
            account = row.get('account.username', '')
            post_time = pd.to_datetime(row.get(time_col))
            content = row.get('content_cleaned', '') or row.get('content', '')

            if pd.notna(post_time) and content and not content.strip().lower().startswith('rt @'):
                account_times[account].append({
                    'time': post_time,
                    'content': content[:100]
                })

        # Find temporally synchronized posting pairs
        accounts = list(account_times.keys())

        for i, account1 in enumerate(accounts):
            for account2 in accounts[i+1:]:

                times1 = account_times[account1]
                times2 = account_times[account2]

                synchronized_pairs = []

                # Check each post from account1 against posts from account2
                for post1 in times1:
                    for post2 in times2:
                        time_diff = abs(post1['time'] - post2['time'])

                        if time_diff <= sync_threshold:
                            synchronized_pairs.append({
                                'time_diff_seconds': time_diff.total_seconds(),
                                'content1': post1['content'],
                                'content2': post2['content'],
                                'time1': post1['time'],
                                'time2': post2['time']
                            })

                # Ultra-conservative: require 3+ synchronized posts
                if len(synchronized_pairs) >= self.min_synchronized_posts:

                    avg_sync_time = np.mean([pair['time_diff_seconds'] for pair in synchronized_pairs])
                    sync_count = len(synchronized_pairs)

                    # Calculate confidence based on sync count and timing precision
                    timing_precision = max(0, 1 - (avg_sync_time / self.temporal_sync_threshold))
                    sync_strength = min(sync_count / 3.0, 1.0)
                    confidence = (timing_precision + sync_strength) / 2

                    # Only flag if confidence is high enough (80%+)
                    if confidence >= self.temporal_confidence_threshold:
                        evidence.append({
                            'type': 'temporal_synchronization',
                            'burst_index': burst_idx,
                            'account1': account1,
                            'account2': account2,
                            'synchronized_posts': sync_count,
                            'avg_sync_time_seconds': avg_sync_time,
                            'confidence': confidence,
                            'evidence_strength': 'VERY_HIGH' if avg_sync_time <= 10 else 'HIGH'
                        })

        return evidence

    def detect_coordination(self, burst_contributors: List[Dict], full_posts_df) -> Dict:
        """
        Phase 4 detection: ALL SIGNALS
        """
        print("\n" + "="*60)
        print("PHASE 4: COMPLETE MULTI-SIGNAL DETECTION")
        print("="*60)
        print("Enabled:  ✓ Identical content")
        print("          ✓ High similarity (>85%)")
        print("          ✓ Hashtag coordination (2+ shared, 60% Jaccard)")
        print("          ✓ URL coordination (shared URLs)")
        print("          ✓ Retweet amplification (3+ retweeters)")
        print("          ✓ Temporal synchronization (≤30s, 3+ posts) ⭐ NEW")
        print("="*60)
        print("⚠️  Ultra-conservative temporal thresholds to avoid false positives")
        print("="*60)

        results = {
            'phase': 'Phase 4',
            'description': 'Complete Multi-Signal Detection (All Signals Enabled)',
            'enabled_features': ['identical_content', 'high_similarity',
                               'hashtag_coordination', 'url_coordination',
                               'retweet_amplification', 'temporal_synchronization'],
            'disabled_features': [],
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

        print(f"\nProcessing {total_bursts} bursts for coordination...")

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

            # Filter to significant contributors only
            significant_accounts = {acc['account.username'] for acc in contrib.get('top_accounts', [])}
            burst_posts = burst_posts[burst_posts['account.username'].isin(significant_accounts)]

            if len(burst_posts) < 2:
                continue

            # Phase 1 signals: Content similarity
            identical_evidence = self._find_identical_content(burst_idx, burst_posts)
            all_evidence.extend(identical_evidence)

            similarity_evidence = self._find_high_similarity(burst_idx, burst_posts)
            all_evidence.extend(similarity_evidence)

            # Phase 2 signals: Hashtag and URL coordination
            hashtag_evidence = self._find_hashtag_coordination(burst_idx, burst_posts)
            all_evidence.extend(hashtag_evidence)

            url_evidence = self._find_url_coordination(burst_idx, burst_posts)
            all_evidence.extend(url_evidence)

            # Phase 3 signal: Retweet amplification
            retweet_evidence = self._find_retweet_coordination(burst_idx, burst_posts)
            all_evidence.extend(retweet_evidence)

            # Phase 4 NEW signal: Temporal synchronization
            temporal_evidence = self._find_temporal_synchronization(burst_idx, burst_posts, time_col)
            all_evidence.extend(temporal_evidence)

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
                        'evidence': 'all_signals_complete'
                    })

            print(f"   Networks: {len(networks)} detected")
        else:
            networks = []
            print("   No networks detected")

        # Calculate statistics by type
        unique_accounts = set()
        for evidence in all_evidence:
            unique_accounts.add(evidence['account1'])
            unique_accounts.add(evidence['account2'])

        identical_count = len([e for e in all_evidence if e['type'] == 'identical_content'])
        similarity_count = len([e for e in all_evidence if e['type'] == 'high_similarity'])
        hashtag_count = len([e for e in all_evidence if e['type'] == 'hashtag_coordination'])
        url_count = len([e for e in all_evidence if e['type'] == 'url_coordination'])
        retweet_count = len([e for e in all_evidence if e['type'] == 'retweet_amplification'])
        temporal_count = len([e for e in all_evidence if e['type'] == 'temporal_synchronization'])

        results['coordination_pairs'] = all_evidence
        results['networks'] = networks
        results['statistics'] = {
            'total_coordination_pairs': len(all_evidence),
            'unique_coordinated_accounts': len(unique_accounts),
            'networks_detected': len(networks),
            'identical_content_pairs': identical_count,
            'high_similarity_pairs': similarity_count,
            'hashtag_coordination_pairs': hashtag_count,
            'url_coordination_pairs': url_count,
            'retweet_amplification_pairs': retweet_count,
            'temporal_synchronization_pairs': temporal_count
        }

        # Print summary
        print("\n" + "="*60)
        print("PHASE 4 RESULTS - COMPLETE PIPELINE")
        print("="*60)
        stats = results['statistics']
        print(f"   Total coordination pairs: {stats['total_coordination_pairs']}")
        print(f"   └─ Identical content: {stats['identical_content_pairs']}")
        print(f"   └─ High similarity: {stats['high_similarity_pairs']}")
        print(f"   └─ Hashtag coordination: {stats['hashtag_coordination_pairs']}")
        print(f"   └─ URL coordination: {stats['url_coordination_pairs']}")
        print(f"   └─ Retweet amplification: {stats['retweet_amplification_pairs']}")
        print(f"   └─ Temporal synchronization: {stats['temporal_synchronization_pairs']} ⭐ NEW")
        print(f"   Unique coordinated accounts: {stats['unique_coordinated_accounts']}")
        print(f"   Coordination networks: {stats['networks_detected']}")

        if networks:
            print(f"\n   Top 5 Networks (by size):")
            sorted_networks = sorted(networks, key=lambda x: x['size'], reverse=True)[:5]
            for net in sorted_networks:
                print(f"      Network {net['network_id']}: {net['size']} accounts")

        # Comparison with Phase 3
        print(f"\n   📊 Improvement over Phase 3:")
        print(f"      Phase 3: 1,081 pairs, 186 accounts, 16 networks")
        print(f"      Phase 4: {stats['total_coordination_pairs']} pairs, {stats['unique_coordinated_accounts']} accounts, {stats['networks_detected']} networks")
        if stats['total_coordination_pairs'] > 1081:
            increase_pairs = ((stats['total_coordination_pairs'] - 1081) / 1081) * 100
            print(f"      Increase: +{increase_pairs:.1f}% pairs")
        elif stats['total_coordination_pairs'] == 1081:
            print(f"      No change: Temporal sync added 0 new pairs")

        # Signal contribution analysis
        total = stats['total_coordination_pairs']
        if total > 0:
            print(f"\n   📊 Signal Contribution:")
            print(f"      RT amplification: {stats['retweet_amplification_pairs']/total*100:.1f}%")
            print(f"      Hashtag coord: {stats['hashtag_coordination_pairs']/total*100:.1f}%")
            print(f"      Content similarity: {(stats['identical_content_pairs']+stats['high_similarity_pairs'])/total*100:.1f}%")
            print(f"      URL coord: {stats['url_coordination_pairs']/total*100:.1f}%")
            print(f"      Temporal sync: {stats['temporal_synchronization_pairs']/total*100:.1f}% ⭐")

        print("="*60)
        print("Phase 4 Complete - ALL SIGNALS TESTED ✓")
        print("="*60)

        return results


def main():
    """Run Phase 4 experiment"""

    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  PHASE 4: COMPLETE MULTI-SIGNAL DETECTION".center(58) + "║")
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

    # Run Phase 4 detection
    print("\n🕵️ STEP 3: Running Phase 4 coordination detection...")
    phase4_detector = Phase4TemporalDetector()
    results = phase4_detector.detect_coordination(burst_contributors, analyzer.df)

    # Save results
    print("\n💾 STEP 4: Saving results...")
    os.makedirs('experiments/results', exist_ok=True)

    output_file = 'experiments/results/phase4_results.json'

    # Prepare JSON-serializable output
    json_results = {
        'experiment': 'Phase 4 - Complete Multi-Signal Detection',
        'timestamp': datetime.now().isoformat(),
        'phase': results['phase'],
        'description': results['description'],
        'enabled_features': results['enabled_features'],
        'disabled_features': results['disabled_features'],
        'statistics': results['statistics'],
        'networks': results['networks'][:20],  # Top 20 networks
        'sample_pairs': results['coordination_pairs'][:50]  # Sample of pairs
    }

    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"   ✓ Results saved to: {output_file}")

    # Summary
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  PHASE 4 EXPERIMENT COMPLETE".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")

    stats = results['statistics']
    print(f"\n📊 Summary:")
    print(f"   • Total pairs: {stats['total_coordination_pairs']}")
    print(f"   • Coordinated accounts: {stats['unique_coordinated_accounts']}")
    print(f"   • Networks detected: {stats['networks_detected']}")
    print(f"   • NEW Temporal sync pairs: {stats['temporal_synchronization_pairs']}")

    if stats['temporal_synchronization_pairs'] == 0:
        print(f"\n💡 Temporal sync added 0 pairs - ultra-conservative thresholds working as designed!")
        print(f"   Phase 3 already caught 98.7% of coordination through RTs.")

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

