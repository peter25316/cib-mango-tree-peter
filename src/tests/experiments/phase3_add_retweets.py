#!/usr/bin/env python3
"""
Phase 3 Experiment: Add Retweet Amplification
Adds retweet coordination detection to Phase 2 signals

This phase tests content + hashtag + URL + RETWEET AMPLIFICATION.
Expected to be the biggest signal based on unified pipeline (1,234+ RT instances).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import json
import re
from datetime import datetime
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Optional
import networkx as nx


class Phase3RetweetDetector:
    """
    Phase 3: Content + Hashtag + URL + Retweet Amplification

    Detects:
    - Identical content (from Phase 1)
    - High similarity content (from Phase 1)
    - Hashtag coordination (from Phase 2)
    - URL coordination (from Phase 2)
    - Retweet amplification ⭐ NEW

    Does NOT detect:
    - Temporal synchronization (reserved for future)
    """

    def __init__(self):
        self.identical_threshold = 0.95
        self.high_similarity_threshold = 0.85
        self.min_content_length = 20
        self.min_hashtag_overlap = 2
        self.hashtag_jaccard_threshold = 0.6
        self.min_retweeters = 3  # Minimum accounts to flag RT coordination

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
        """
        Extract the source account from RT content.
        Based on unified pipeline implementation.
        """
        # Pattern to match "RT @username" at the beginning
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
        """
        Find coordinated retweet amplification patterns.
        Based on unified pipeline implementation.

        Detects accounts that coordinate to amplify specific sources through retweets.
        """
        evidence = []

        # Extract retweet patterns
        rt_sources = defaultdict(lambda: {
            'retweeters': set(),
            'content': None
        })

        for _, row in burst_posts.iterrows():
            content = row.get('content_cleaned', '') or row.get('content', '')
            retweeter = row.get('account.username', '')

            if not content:
                continue

            # Check if this is a retweet
            if content.strip().lower().startswith('rt @'):
                # Extract RT source
                rt_source = self._extract_rt_source(content)
                if rt_source:
                    rt_sources[rt_source]['retweeters'].add(retweeter)
                    # Store the original content being retweeted
                    if rt_sources[rt_source]['content'] is None:
                        rt_sources[rt_source]['content'] = content

        # Find coordinated amplification patterns
        for source, data in rt_sources.items():
            retweeters = data['retweeters']

            if len(retweeters) >= self.min_retweeters:  # At least 3 accounts retweeting same source
                retweeter_list = list(retweeters)

                # Amplification strength based on number of retweeters
                amplification_strength = min(len(retweeters) / 10.0, 1.0)

                # Determine evidence strength
                if len(retweeters) >= 10:
                    evidence_strength = 'VERY_HIGH'
                elif len(retweeters) >= 5:
                    evidence_strength = 'HIGH'
                else:
                    evidence_strength = 'MEDIUM'

                # Create pairs for all retweeters (they're coordinating by RTing same source)
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

    def detect_coordination(self, burst_contributors: List[Dict], full_posts_df) -> Dict:
        """
        Phase 3 detection: Content + Hashtags + URLs + Retweets
        """
        print("\n" + "="*60)
        print("PHASE 3: ADD RETWEET AMPLIFICATION")
        print("="*60)
        print("Enabled:  ✓ Identical content")
        print("          ✓ High similarity (>85%)")
        print("          ✓ Hashtag coordination (2+ shared, 60% Jaccard)")
        print("          ✓ URL coordination (shared URLs)")
        print("          ✓ Retweet amplification (3+ retweeters) ⭐ NEW")
        print("Disabled: ✗ Temporal synchronization")
        print("="*60)

        results = {
            'phase': 'Phase 3',
            'description': 'Content + Hashtag + URL + Retweet Amplification',
            'enabled_features': ['identical_content', 'high_similarity',
                               'hashtag_coordination', 'url_coordination',
                               'retweet_amplification'],
            'disabled_features': ['temporal_sync'],
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

            # Phase 3 NEW signal: Retweet amplification
            retweet_evidence = self._find_retweet_coordination(burst_idx, burst_posts)
            all_evidence.extend(retweet_evidence)

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
                        'evidence': 'content_hashtag_url_retweet'
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
            'retweet_amplification_pairs': retweet_count
        }

        # Print summary
        print("\n" + "="*60)
        print("PHASE 3 RESULTS")
        print("="*60)
        stats = results['statistics']
        print(f"   Total coordination pairs: {stats['total_coordination_pairs']}")
        print(f"   └─ Identical content: {stats['identical_content_pairs']}")
        print(f"   └─ High similarity: {stats['high_similarity_pairs']}")
        print(f"   └─ Hashtag coordination: {stats['hashtag_coordination_pairs']}")
        print(f"   └─ URL coordination: {stats['url_coordination_pairs']}")
        print(f"   └─ Retweet amplification: {stats['retweet_amplification_pairs']} ⭐ NEW")
        print(f"   Unique coordinated accounts: {stats['unique_coordinated_accounts']}")
        print(f"   Coordination networks: {stats['networks_detected']}")

        if networks:
            print(f"\n   Top 5 Networks (by size):")
            sorted_networks = sorted(networks, key=lambda x: x['size'], reverse=True)[:5]
            for net in sorted_networks:
                print(f"      Network {net['network_id']}: {net['size']} accounts")

        # Comparison with previous phases
        print(f"\n   📊 Improvement over Phase 2:")
        print(f"      Phase 2: 14 pairs, 19 accounts, 8 networks")
        print(f"      Phase 3: {stats['total_coordination_pairs']} pairs, {stats['unique_coordinated_accounts']} accounts, {stats['networks_detected']} networks")
        if stats['total_coordination_pairs'] > 14:
            increase_pairs = ((stats['total_coordination_pairs'] - 14) / 14) * 100
            increase_accounts = ((stats['unique_coordinated_accounts'] - 19) / 19) * 100
            print(f"      Increase: +{increase_pairs:.0f}% pairs, +{increase_accounts:.0f}% accounts")

        print("="*60)
        print("Phase 3 Complete - Retweet amplification added ✓")
        print("="*60)

        return results


def main():
    """Run Phase 3 experiment"""

    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  PHASE 3 EXPERIMENT: ADD RETWEET AMPLIFICATION".center(58) + "║")
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

    # Run Phase 3 detection
    print("\n🕵️ STEP 3: Running Phase 3 coordination detection...")
    phase3_detector = Phase3RetweetDetector()
    results = phase3_detector.detect_coordination(burst_contributors, analyzer.df)

    # Save results
    print("\n💾 STEP 4: Saving results...")
    os.makedirs('experiments/results', exist_ok=True)

    output_file = 'experiments/results/phase3_results.json'

    # Prepare JSON-serializable output
    json_results = {
        'experiment': 'Phase 3 - Add Retweet Amplification',
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
    print("║" + "  PHASE 3 EXPERIMENT COMPLETE".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")

    stats = results['statistics']
    print(f"\n📊 Summary:")
    print(f"   • Total pairs: {stats['total_coordination_pairs']}")
    print(f"   • Coordinated accounts: {stats['unique_coordinated_accounts']}")
    print(f"   • Networks detected: {stats['networks_detected']}")
    print(f"   • NEW RT pairs: {stats['retweet_amplification_pairs']}")
    print(f"\n💡 Retweet amplification is expected to be the dominant signal!")

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

