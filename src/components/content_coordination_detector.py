#!/usr/bin/env python3
"""
Content-Based Coordination Detector
Focuses on detecting coordinated accounts through content similarity analysis.
"""

import polars as pl
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional
import re
from difflib import SequenceMatcher
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import random
import os


class ContentCoordinationDetector:
    """
    Detects coordinated accounts by analyzing content similarity patterns
    within detected burst periods.
    """

    def __init__(self,
                 identical_threshold: float = 0.95,
                 high_similarity_threshold: float = 0.85,
                 min_content_length: int = 20):
        self.identical_threshold = identical_threshold
        self.high_similarity_threshold = high_similarity_threshold
        self.min_content_length = min_content_length
        self.coordination_pairs = []
        self.coordination_networks = []

    def detect_coordination(self, burst_contributors: List[Dict], full_posts_df: pl.DataFrame) -> Dict:
        """
        Main coordination detection pipeline focusing on content analysis.
        """
        print("🕵️ CONTENT-BASED COORDINATION DETECTION")
        print("-" * 50)

        results = {
            'coordination_pairs': [],
            'identical_content_groups': [],
            'hashtag_coordination': [],
            'url_coordination': [],
            'coordination_networks': [],
            'summary_stats': {},
            'confidence_level': 'NONE'
        }

        # Convert to pandas for easier processing
        if hasattr(full_posts_df, 'to_pandas'):
            posts_df = full_posts_df.to_pandas()
        else:
            posts_df = full_posts_df

        # Ensure datetime column and handle timezone compatibility
        time_col = 'created_at' if 'created_at' in posts_df.columns else 'post_timestamp'
        posts_df[time_col] = pd.to_datetime(posts_df[time_col])

        # Ensure timezone compatibility - convert all to naive datetimes
        if posts_df[time_col].dt.tz is not None:
            posts_df[time_col] = posts_df[time_col].dt.tz_convert('UTC').dt.tz_localize(None)

        print(f"📊 Analyzing {len(burst_contributors)} bursts for content coordination...")

        # Analyze each burst for content coordination
        all_coordination_evidence = []

        for burst_idx, contrib in enumerate(burst_contributors):
            burst_evidence = self._analyze_burst_content_coordination(
                burst_idx, contrib, posts_df, time_col
            )
            all_coordination_evidence.extend(burst_evidence)

        print(f"🔍 Found {len(all_coordination_evidence)} coordination evidence instances")

        # Process and categorize evidence
        results = self._process_coordination_evidence(all_coordination_evidence)

        # Build coordination networks
        results['coordination_networks'] = self._build_coordination_networks(
            results['coordination_pairs']
        )

        # Calculate summary statistics
        results['summary_stats'] = self._calculate_summary_stats(results)
        results['confidence_level'] = self._determine_confidence_level(results)

        # Add cross-burst analysis for persistent coordination detection
        cross_burst_evidence = self.add_cross_burst_analysis(all_coordination_evidence, burst_contributors)
        results['cross_burst_evidence'] = cross_burst_evidence
        results['coordination_pairs'].extend(cross_burst_evidence)

        # Step 2: Ultra-conservative mechanical posting intervals detection
        print("🤖 Analyzing mechanical posting patterns...")
        mechanical_evidence = self._find_mechanical_posting_intervals(burst_contributors, posts_df, time_col)

        # Step 3: Ultra-conservative identical activity hours detection
        print("🕐 Analyzing identical activity windows...")
        activity_evidence = self._find_identical_activity_hours(burst_contributors, posts_df, time_col)

        # Combine behavioral evidence
        all_behavioral_evidence = mechanical_evidence + activity_evidence
        results['behavioral_evidence'] = all_behavioral_evidence
        if all_behavioral_evidence:  # Only add if we find obvious patterns
            results['coordination_pairs'].extend(all_behavioral_evidence)

        print(f"🔍 Found {len(mechanical_evidence)} mechanical posting patterns")
        print(f"🔍 Found {len(activity_evidence)} identical activity hour patterns")

        self._print_results(results)

        # Generate visualizations
        print("\n🎨 GENERATING NETWORK VISUALIZATIONS...")
        print("-" * 40)
        self.generate_visualizations(results)

        return results

    def _analyze_burst_content_coordination(self, burst_idx: int, contrib: Dict,
                                          posts_df: pd.DataFrame, time_col: str) -> List[Dict]:
        """
        Analyze content coordination within a single burst.
        """
        start_time = contrib['start_time']
        end_time = contrib['end_time']

        # Ensure timezone compatibility for burst time windows
        if hasattr(start_time, 'tzinfo') and start_time.tzinfo is not None:
            start_time = start_time.replace(tzinfo=None)
        if hasattr(end_time, 'tzinfo') and end_time.tzinfo is not None:
            end_time = end_time.replace(tzinfo=None)

        # Convert to pandas datetime for compatibility
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)

        # Get posts in this burst window
        burst_mask = ((posts_df[time_col] >= start_time) &
                     (posts_df[time_col] <= end_time))
        burst_posts = posts_df[burst_mask].copy()

        if len(burst_posts) < 2:
            return []

        # Filter to significant contributors only
        significant_accounts = {acc['account.username'] for acc in contrib.get('top_accounts', [])}
        burst_posts = burst_posts[burst_posts['account.username'].isin(significant_accounts)]

        if len(burst_posts) < 2:
            return []

        coordination_evidence = []

        # 1. Identical/Near-identical content detection
        content_evidence = self._find_identical_content_coordination(burst_idx, burst_posts, time_col)
        coordination_evidence.extend(content_evidence)

        # 2. High similarity content detection
        similarity_evidence = self._find_high_similarity_coordination(burst_idx, burst_posts)
        coordination_evidence.extend(similarity_evidence)

        # 3. Hashtag coordination detection
        hashtag_evidence = self._find_hashtag_coordination(burst_idx, burst_posts)
        coordination_evidence.extend(hashtag_evidence)

        # 4. URL coordination detection
        url_evidence = self._find_url_coordination(burst_idx, burst_posts)
        coordination_evidence.extend(url_evidence)

        # 5. Retweet source coordination detection
        rt_evidence = self._find_retweet_coordination(burst_idx, burst_posts)
        coordination_evidence.extend(rt_evidence)

        # 6. Ultra-conservative temporal synchronization (Step 1: 30 seconds, 3+ posts)
        temporal_evidence = self._find_ultra_conservative_temporal_sync(burst_idx, burst_posts, time_col)
        coordination_evidence.extend(temporal_evidence)

        return coordination_evidence

    def _find_identical_content_coordination(self, burst_idx: int, burst_posts: pd.DataFrame, time_col: str) -> List[Dict]:
        """
        Find accounts posting identical or near-identical content.
        This is the strongest evidence of coordination.
        """
        evidence = []

        # Group posts by normalized content
        content_groups = defaultdict(list)

        for _, row in burst_posts.iterrows():
            content = row.get('content_cleaned', '')
            account = row['account.username']

            if not content or len(content.strip()) < self.min_content_length:
                continue

            # Skip retweets - focus on original content coordination only
            if content.strip().lower().startswith('rt @'):
                continue

            # Normalize content (remove extra spaces, convert to lowercase)
            normalized_content = ' '.join(content.lower().strip().split())

            content_groups[normalized_content].append({
                'account': account,
                'original_content': content,
                'post_time': row.get(time_col, None)
            })

        # Find groups with multiple accounts (coordination)
        for normalized_content, posts in content_groups.items():
            if len(posts) >= 2:
                # Check if these are actually different accounts
                unique_accounts = {post['account'] for post in posts}
                if len(unique_accounts) >= 2:
                    # This is identical content coordination
                    for i, post1 in enumerate(posts):
                        for post2 in posts[i+1:]:
                            if post1['account'] != post2['account']:
                                evidence.append({
                                    'type': 'identical_content',
                                    'burst_index': burst_idx,
                                    'account1': post1['account'],
                                    'account2': post2['account'],
                                    'content1': post1['original_content'],
                                    'content2': post2['original_content'],
                                    'similarity_score': 1.0,
                                    'confidence': 1.0,  # Highest confidence
                                    'evidence_strength': 'VERY_HIGH'
                                })

        return evidence

    def _find_high_similarity_coordination(self, burst_idx: int, burst_posts: pd.DataFrame) -> List[Dict]:
        """
        Find accounts posting highly similar (but not identical) content.
        """
        evidence = []

        # Get unique accounts and their posts
        account_posts = defaultdict(list)

        for _, row in burst_posts.iterrows():
            content = row.get('content_cleaned', '')
            account = row['account.username']

            if not content or len(content.strip()) < self.min_content_length:
                continue

            # Skip retweets - focus on original content coordination only
            if content.strip().lower().startswith('rt @'):
                continue

            account_posts[account].append(content.strip())

        # Compare content between different accounts
        accounts = list(account_posts.keys())

        for i, account1 in enumerate(accounts):
            for account2 in accounts[i+1:]:

                posts1 = account_posts[account1]
                posts2 = account_posts[account2]

                # Compare each post from account1 with each post from account2
                for content1 in posts1[:3]:  # Limit to first 3 posts for performance
                    for content2 in posts2[:3]:

                        similarity = self._calculate_text_similarity(content1, content2)

                        if similarity >= self.high_similarity_threshold:
                            confidence = min(similarity * 1.2, 1.0)  # Scale similarity to confidence
                            strength = 'HIGH' if similarity >= 0.9 else 'MEDIUM'

                            evidence.append({
                                'type': 'high_similarity',
                                'burst_index': burst_idx,
                                'account1': account1,
                                'account2': account2,
                                'content1': content1[:150] + "..." if len(content1) > 150 else content1,
                                'content2': content2[:150] + "..." if len(content2) > 150 else content2,
                                'similarity_score': similarity,
                                'confidence': confidence,
                                'evidence_strength': strength
                            })

        return evidence

    def _find_hashtag_coordination(self, burst_idx: int, burst_posts: pd.DataFrame) -> List[Dict]:
        """
        Find accounts using identical hashtag combinations.
        """
        evidence = []

        # Extract hashtags by account
        account_hashtags = defaultdict(list)

        for _, row in burst_posts.iterrows():
            content = row.get('content_cleaned', '')
            account = row['account.username']

            if not content:
                continue

            # Skip retweets - focus on original content hashtag coordination only
            if content.strip().lower().startswith('rt @'):
                continue

            hashtags = self._extract_hashtags(content)
            if hashtags:
                account_hashtags[account].extend(hashtags)

        # Find accounts with coordinated hashtag usage
        accounts = list(account_hashtags.keys())

        for i, account1 in enumerate(accounts):
            for account2 in accounts[i+1:]:

                hashtags1 = set(account_hashtags[account1])
                hashtags2 = set(account_hashtags[account2])

                if len(hashtags1) >= 2 and len(hashtags2) >= 2:
                    # Calculate Jaccard similarity
                    intersection = hashtags1 & hashtags2
                    union = hashtags1 | hashtags2

                    if len(intersection) >= 2:  # At least 2 shared hashtags
                        jaccard_similarity = len(intersection) / len(union)

                        if jaccard_similarity >= 0.6:  # 60% hashtag overlap
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
        """
        Find accounts sharing the same URLs.
        """
        evidence = []

        # Extract URLs by account
        account_urls = defaultdict(set)

        for _, row in burst_posts.iterrows():
            content = row.get('content_cleaned', '')
            account = row['account.username']

            if not content:
                continue

            # Skip retweets - focus on original content URL coordination only
            if content.strip().lower().startswith('rt @'):
                continue

            urls = self._extract_urls(content)
            if urls:
                account_urls[account].update(urls)

        # Find accounts sharing URLs
        accounts = list(account_urls.keys())

        for i, account1 in enumerate(accounts):
            for account2 in accounts[i+1:]:

                urls1 = account_urls[account1]
                urls2 = account_urls[account2]

                shared_urls = urls1 & urls2

                if len(shared_urls) >= 1:  # Any shared URL is suspicious in coordination context
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
        Find coordinated retweet amplification patterns with temporal synchronization.
        Detects accounts that coordinate to amplify specific sources through retweets,
        enhanced with timing analysis to catch synchronized RT networks.
        """
        evidence = []

        # Extract retweet patterns with timestamps
        rt_sources = defaultdict(lambda: {
            'retweeters': set(),
            'rt_timestamps': [],
            'content': None
        })

        for _, row in burst_posts.iterrows():
            content = row.get('content_cleaned', '')
            retweeter = row['account.username']
            post_time = pd.to_datetime(row.get('post_timestamp', row.get('created_at')))

            if not content:
                continue

            # Check if this is a retweet
            if content.strip().lower().startswith('rt @'):
                # Extract RT source
                rt_source = self._extract_rt_source(content)
                if rt_source and pd.notna(post_time):
                    rt_sources[rt_source]['retweeters'].add(retweeter)
                    rt_sources[rt_source]['rt_timestamps'].append({
                        'retweeter': retweeter,
                        'timestamp': post_time,
                        'content': content
                    })
                    # Store the original content being retweeted
                    if rt_sources[rt_source]['content'] is None:
                        rt_sources[rt_source]['content'] = content

        # Find coordinated amplification patterns with temporal analysis
        for source, data in rt_sources.items():
            retweeters = data['retweeters']
            timestamps = data['rt_timestamps']

            if len(retweeters) >= 3:  # At least 3 accounts retweeting same source
                retweeter_list = list(retweeters)

                # Basic amplification strength
                amplification_strength = min(len(retweeters) / 10.0, 1.0)

                # Enhanced: Temporal synchronization analysis
                temporal_sync_evidence = self._analyze_rt_temporal_synchronization(timestamps)

                # Calculate combined confidence
                base_confidence = amplification_strength
                temporal_boost = 0
                evidence_strength = 'MEDIUM'

                if temporal_sync_evidence['synchronized_clusters']:
                    # Boost confidence for temporal coordination
                    cluster_strength = temporal_sync_evidence['max_cluster_size'] / len(retweeters)
                    timing_precision = temporal_sync_evidence['avg_cluster_timing_precision']
                    temporal_boost = (cluster_strength * timing_precision) * 0.3  # Up to 30% boost

                    # Upgrade evidence strength
                    if temporal_sync_evidence['max_cluster_size'] >= 5:
                        evidence_strength = 'VERY_HIGH'
                    elif temporal_sync_evidence['max_cluster_size'] >= 3:
                        evidence_strength = 'HIGH'

                # Final confidence calculation
                final_confidence = min(base_confidence + temporal_boost, 1.0)

                # Determine evidence strength from both amplification and timing
                if len(retweeters) >= 10:
                    evidence_strength = 'VERY_HIGH'
                elif len(retweeters) >= 5 or temporal_sync_evidence['synchronized_clusters']:
                    evidence_strength = 'HIGH'

                evidence.append({
                    'type': 'retweet_amplification',
                    'burst_index': burst_idx,
                    'rt_source': source,
                    'retweeters': retweeter_list,
                    'amplification_count': len(retweeters),
                    'coordination_strength': amplification_strength,
                    'temporal_sync_data': temporal_sync_evidence,
                    'confidence': final_confidence,
                    'evidence_strength': evidence_strength,
                    'original_content': data['content'][:150] + "..." if len(data['content']) > 150 else data['content'],
                    'has_temporal_coordination': len(temporal_sync_evidence['synchronized_clusters']) > 0
                })

        return evidence

    def _extract_rt_source(self, content: str) -> Optional[str]:
        """Extract the source account from RT content."""
        import re

        # Pattern to match "RT @username" at the beginning
        rt_pattern = r'^RT\s+@(\w+)'
        match = re.match(rt_pattern, content.strip(), re.IGNORECASE)

        if match:
            return match.group(1).lower()
        return None

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using SequenceMatcher."""
        if not text1 or not text2:
            return 0.0

        # Normalize texts
        text1_clean = ' '.join(text1.lower().split())
        text2_clean = ' '.join(text2.lower().split())

        return SequenceMatcher(None, text1_clean, text2_clean).ratio()

    def _extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text content."""
        if not text:
            return []
        return re.findall(r'#\w+', text.lower())

    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text content."""
        if not text:
            return []
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.findall(url_pattern, text)

    def _process_coordination_evidence(self, all_evidence: List[Dict]) -> Dict:
        """Process and categorize all coordination evidence."""

        # Group evidence by type
        coordination_pairs = []
        identical_content_groups = []
        hashtag_coordination = []
        url_coordination = []
        retweet_amplification = []

        for evidence in all_evidence:
            if evidence['type'] == 'identical_content':
                identical_content_groups.append(evidence)
                coordination_pairs.append(evidence)
            elif evidence['type'] == 'high_similarity':
                coordination_pairs.append(evidence)
            elif evidence['type'] == 'hashtag_coordination':
                hashtag_coordination.append(evidence)
                coordination_pairs.append(evidence)
            elif evidence['type'] == 'url_coordination':
                url_coordination.append(evidence)
                coordination_pairs.append(evidence)
            elif evidence['type'] == 'retweet_amplification':
                retweet_amplification.append(evidence)
                # Convert RT amplification to coordination pairs for network building
                retweeters = evidence['retweeters']
                for i, retweeter1 in enumerate(retweeters):
                    for retweeter2 in retweeters[i+1:]:
                        coordination_pairs.append({
                            'type': 'retweet_coordination',
                            'account1': retweeter1,
                            'account2': retweeter2,
                            'confidence': evidence['confidence'],
                            'rt_source': evidence['rt_source'],
                            'burst_index': evidence['burst_index']
                        })
            elif evidence['type'] == 'ultra_conservative_temporal_sync':
                # Add ultra-conservative temporal evidence to coordination pairs
                coordination_pairs.append(evidence)
            elif evidence['type'] == 'mechanical_posting_intervals':
                # Add mechanical posting interval evidence to coordination pairs
                coordination_pairs.append(evidence)
            elif evidence['type'] == 'identical_activity_hours':
                # Add identical activity hours evidence to coordination pairs
                coordination_pairs.append(evidence)

        return {
            'coordination_pairs': coordination_pairs,
            'identical_content_groups': identical_content_groups,
            'hashtag_coordination': hashtag_coordination,
            'url_coordination': url_coordination,
            'retweet_amplification': retweet_amplification
        }

    def _build_coordination_networks(self, coordination_pairs: List[Dict]) -> List[Dict]:
        """Build coordination networks using NetworkX with advanced analysis."""

        # Create NetworkX graph
        G = nx.Graph()

        # Add edges with attributes
        for pair in coordination_pairs:
            account1 = pair['account1']
            account2 = pair['account2']
            confidence = pair['confidence']
            evidence_type = pair['type']
            burst_index = pair.get('burst_index', 0)

            # Add edge with weight and attributes
            if G.has_edge(account1, account2):
                # If edge exists, update with stronger confidence and add evidence types
                existing_weight = G[account1][account2]['weight']
                G[account1][account2]['weight'] = max(existing_weight, confidence)
                G[account1][account2]['evidence_types'].append(evidence_type)
                G[account1][account2]['coordination_pairs'].append(pair)
            else:
                G.add_edge(account1, account2,
                          weight=confidence,
                          evidence_types=[evidence_type],
                          coordination_pairs=[pair],
                          burst_index=burst_index)

        print(f"🕸️ Built NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # Find connected components (networks)
        networks = []
        connected_components = list(nx.connected_components(G))

        print(f"🔍 Found {len(connected_components)} connected components")

        for i, component in enumerate(connected_components):
            if len(component) >= 2:  # Network must have at least 2 accounts

                # Create subgraph for this component
                subgraph = G.subgraph(component)

                # Calculate network metrics using NetworkX
                network_metrics = self._calculate_network_metrics(subgraph)

                # Get coordination pairs for this network
                network_pairs = []
                evidence_types = set()
                for edge in subgraph.edges(data=True):
                    network_pairs.extend(edge[2]['coordination_pairs'])
                    evidence_types.update(edge[2]['evidence_types'])

                avg_confidence = np.mean([p['confidence'] for p in network_pairs])

                # Enhanced network analysis
                communities = self._detect_communities(subgraph)
                network_structure = self._analyze_network_structure(subgraph)

                networks.append({
                    'network_id': None,  # Will be assigned after sorting
                    'accounts': list(component),
                    'size': len(component),
                    'coordination_pairs': network_pairs,
                    'avg_confidence': avg_confidence,
                    'evidence_types': list(evidence_types),
                    'risk_level': 'HIGH' if avg_confidence > 0.8 else 'MEDIUM' if avg_confidence > 0.6 else 'LOW',

                    # Enhanced NetworkX metrics
                    'network_metrics': network_metrics,
                    'communities': communities,
                    'network_structure': network_structure,
                    'subgraph': subgraph  # Store for visualization
                })

        # Sort networks by size and confidence
        networks.sort(key=lambda x: (x['size'], x['avg_confidence']), reverse=True)

        # Assign network IDs based on final sorted ranking (1 = largest/best network)
        for idx, network in enumerate(networks):
            network['network_id'] = idx + 1

        # Community detection across entire graph
        if G.number_of_nodes() > 0:
            try:
                global_communities = list(nx.community.greedy_modularity_communities(G))
                print(f"🏘️ Global community detection found {len(global_communities)} communities")

                # Add global community information
                for network in networks:
                    network['global_community_info'] = self._map_to_global_communities(
                        network['accounts'], global_communities)
            except:
                print("⚠️ Global community detection failed (graph too small or disconnected)")

        return networks

    def _calculate_summary_stats(self, results: Dict) -> Dict:
        """Calculate summary statistics for the coordination analysis."""

        total_pairs = len(results['coordination_pairs'])
        identical_content = len(results['identical_content_groups'])
        hashtag_coord = len(results['hashtag_coordination'])
        url_coord = len(results['url_coordination'])
        rt_amplification = len(results.get('retweet_amplification', []))
        behavioral_patterns = len(results.get('behavioral_evidence', []))
        networks = results['coordination_networks']

        total_network_accounts = sum(net['size'] for net in networks)
        high_risk_networks = len([net for net in networks if net['risk_level'] == 'HIGH'])

        return {
            'total_coordination_pairs': total_pairs,
            'identical_content_instances': identical_content,
            'hashtag_coordination_instances': hashtag_coord,
            'url_coordination_instances': url_coord,
            'retweet_amplification_instances': rt_amplification,
            'behavioral_pattern_instances': behavioral_patterns,
            'total_networks': len(networks),
            'total_network_accounts': total_network_accounts,
            'high_risk_networks': high_risk_networks,
            'largest_network_size': max([net['size'] for net in networks]) if networks else 0
        }

    def _determine_confidence_level(self, results: Dict) -> str:
        """Determine overall confidence level in coordination detection."""

        stats = results['summary_stats']
        identical_content = stats['identical_content_instances']
        high_risk_networks = stats['high_risk_networks']
        total_pairs = stats['total_coordination_pairs']

        if identical_content >= 10 and high_risk_networks >= 2:
            return 'VERY_HIGH'
        elif identical_content >= 5 or high_risk_networks >= 1:
            return 'HIGH'
        elif total_pairs >= 10:
            return 'MEDIUM'
        elif total_pairs >= 3:
            return 'LOW'
        else:
            return 'NONE'

    def _print_results(self, results: Dict):
        """Print coordination detection results."""

        stats = results['summary_stats']
        networks = results['coordination_networks']
        confidence = results['confidence_level']

        print(f"\n🎯 CONTENT COORDINATION RESULTS")
        print("=" * 50)

        print(f"📊 Overall Confidence: {confidence}")
        print(f"🤝 Total coordination pairs: {stats['total_coordination_pairs']}")
        print(f"📋 Identical content instances: {stats['identical_content_instances']}")
        print(f"🏷️ Hashtag coordination instances: {stats['hashtag_coordination_instances']}")
        print(f"🔗 URL coordination instances: {stats['url_coordination_instances']}")
        print(f"🔄 Retweet amplification instances: {stats['retweet_amplification_instances']}")

        # Enhanced: Show RT temporal coordination stats
        rt_amplifications = results.get('retweet_amplification', [])
        temporal_rt_count = sum(1 for rt in rt_amplifications if rt.get('has_temporal_coordination', False))
        if temporal_rt_count > 0:
            print(f"⏱️ RT instances with temporal coordination: {temporal_rt_count}")

        print(f"🤖 Behavioral pattern instances: {stats.get('behavioral_pattern_instances', 0)}")

        if networks:
            print(f"\n🕸️ COORDINATION NETWORKS:")
            print("-" * 40)
            print(f"📊 Networks detected: {stats['total_networks']}")
            print(f"👥 Accounts in networks: {stats['total_network_accounts']}")
            print(f"⚠️ High-risk networks: {stats['high_risk_networks']}")
            print(f"🎯 Largest network: {stats['largest_network_size']} accounts")

            print(f"\n📋 TOP NETWORKS (Enhanced with NetworkX):")
            for i, network in enumerate(networks[:5]):
                accounts_str = ', '.join([f'@{acc}' for acc in network['accounts'][:5]])
                if len(network['accounts']) > 5:
                    accounts_str += f" ... (+{len(network['accounts'])-5} more)"

                print(f"  {i+1}. Network {i+1} ({network['risk_level']} risk)")
                print(f"     Size: {network['size']} accounts")
                print(f"     Confidence: {network['avg_confidence']:.2f}")
                print(f"     Evidence: {', '.join(network['evidence_types'])}")

                # Enhanced NetworkX metrics
                metrics = network.get('network_metrics', {})
                structure = network.get('network_structure', {})
                communities = network.get('communities', {})

                if metrics and not metrics.get('error'):
                    print(f"     📊 Network metrics: Density {metrics.get('density', 0):.2f}, Clustering {metrics.get('avg_clustering', 0):.2f}")

                    if 'most_central_account' in metrics:
                        central_account = metrics['most_central_account']
                        central_score = metrics['most_central_score']
                        print(f"     🎯 Most central: @{central_account} (centrality: {central_score:.2f})")

                if structure and not structure.get('error'):
                    struct_type = structure.get('type', 'UNKNOWN')
                    description = structure.get('description', '')
                    print(f"     🏗️ Structure: {struct_type} - {description}")

                    if 'potential_hubs' in structure and structure['potential_hubs']:
                        hub_accounts = structure['potential_hubs'][:3]  # Show top 3 hubs
                        print(f"     🌟 Hub accounts: {', '.join(['@' + acc for acc in hub_accounts])}")

                if communities.get('greedy_modularity', {}).get('count', 0) > 1:
                    comm_count = communities['greedy_modularity']['count']
                    modularity = communities['greedy_modularity'].get('modularity', 0)
                    print(f"     🏘️ Sub-communities: {comm_count} detected (modularity: {modularity:.2f})")

                print(f"     Accounts: {accounts_str}")
                print()
        else:
            print(f"\n❌ No coordination networks detected")

        # Print cross-burst analysis results
        cross_burst_evidence = results.get('cross_burst_evidence', [])
        coordination_hubs = [e for e in cross_burst_evidence if e['type'] == 'coordination_hub']

        if coordination_hubs:
            print(f"\n🎯 COORDINATION HUBS:")
            print("-" * 40)
            for i, hub in enumerate(coordination_hubs[:5], 1):
                retweeter_count = hub['total_retweeters']
                burst_count = hub['burst_appearances']
                confidence = hub['confidence']

                print(f"{i}. @{hub['hub_account']} ({hub['evidence_strength']} confidence)")
                print(f"   📊 {retweeter_count} retweeters across {burst_count} bursts")
                print(f"   🎯 Confidence: {confidence:.2f}")
                print(f"   🕸️ Network: {', '.join(['@' + acc for acc in hub['retweeter_network'][:5]])}")
                if len(hub['retweeter_network']) > 5:
                    print(f"             ... (+{len(hub['retweeter_network'])-5} more)")
                print()

        # Print behavioral patterns
        behavioral_evidence = results.get('behavioral_evidence', [])
        if behavioral_evidence:
            print(f"\n🤖 BEHAVIORAL COORDINATION PATTERNS:")
            print("-" * 40)
            for i, pattern in enumerate(behavioral_evidence[:5], 1):
                similarity = pattern['behavioral_similarity']
                types = ', '.join(pattern['similarity_types'])
                confidence = pattern['confidence']

                print(f"{i}. @{pattern['account1']} ↔ @{pattern['account2']} ({pattern['evidence_strength']} confidence)")
                print(f"   📊 Behavioral similarity: {similarity:.2f}")
                print(f"   🎯 Confidence: {confidence:.2f}")
                print(f"   🤖 Pattern types: {types}")
                print()

    def add_cross_burst_analysis(self, all_evidence: List[Dict], burst_contributors: List[Dict]) -> List[Dict]:
        """
        Add cross-burst analysis to detect accounts that coordinate across multiple bursts.
        This catches accounts like @maxjett12 that are persistent coordination hubs.
        """

        # Track RT sources across all bursts
        rt_source_activity = defaultdict(lambda: {
            'total_amplifications': 0,
            'burst_appearances': set(),
            'total_retweeters': set(),
            'burst_details': []
        })

        # Track account activity across bursts
        account_activity = defaultdict(lambda: {
            'burst_count': 0,
            'total_posts': 0,
            'rt_source_count': 0,
            'coordination_types': set()
        })

        # Analyze RT amplification evidence
        rt_amplifications = [e for e in all_evidence if e['type'] == 'retweet_amplification']

        for evidence in rt_amplifications:
            source = evidence['rt_source']
            burst_idx = evidence['burst_index']
            retweeters = evidence['retweeters']

            # Track RT source activity
            rt_source_activity[source]['total_amplifications'] += evidence['amplification_count']
            rt_source_activity[source]['burst_appearances'].add(burst_idx)
            rt_source_activity[source]['total_retweeters'].update(retweeters)
            rt_source_activity[source]['burst_details'].append({
                'burst_index': burst_idx,
                'amplification_count': evidence['amplification_count'],
                'retweeters': retweeters
            })

            # Track retweeter activity
            for retweeter in retweeters:
                account_activity[retweeter]['rt_source_count'] += 1
                account_activity[retweeter]['coordination_types'].add('rt_amplifier')

        # Track other coordination evidence
        for evidence in all_evidence:
            if evidence['type'] != 'retweet_amplification':
                for account_key in ['account1', 'account2']:
                    if account_key in evidence:
                        account = evidence[account_key]
                        account_activity[account]['coordination_types'].add(evidence['type'])

        # Generate cross-burst coordination evidence
        cross_burst_evidence = []

        # Flag persistent RT sources (coordination hubs like @maxjett12)
        for source, activity in rt_source_activity.items():
            if len(activity['burst_appearances']) >= 3 or len(activity['total_retweeters']) >= 10:
                # This is a coordination hub
                persistence_score = min(len(activity['burst_appearances']) / 10.0, 1.0)
                amplification_score = min(len(activity['total_retweeters']) / 50.0, 1.0)
                combined_score = (persistence_score + amplification_score) / 2

                cross_burst_evidence.append({
                    'type': 'coordination_hub',
                    'hub_account': source,
                    'burst_appearances': len(activity['burst_appearances']),
                    'total_retweeters': len(activity['total_retweeters']),
                    'total_amplifications': activity['total_amplifications'],
                    'persistence_score': persistence_score,
                    'amplification_score': amplification_score,
                    'confidence': combined_score,
                    'evidence_strength': 'VERY_HIGH' if combined_score > 0.8 else 'HIGH' if combined_score > 0.6 else 'MEDIUM',
                    'retweeter_network': list(activity['total_retweeters'])[:20]  # Limit for display
                })

        # Flag persistent coordination actors
        total_bursts = len(burst_contributors)
        for account, activity in account_activity.items():
            participation_rate = activity['burst_count'] / total_bursts if total_bursts > 0 else 0
            coordination_diversity = len(activity['coordination_types'])

            if participation_rate >= 0.3 or coordination_diversity >= 2:  # Active in 30%+ bursts or multiple coordination types
                cross_burst_evidence.append({
                    'type': 'persistent_actor',
                    'actor_account': account,
                    'burst_participation_rate': participation_rate,
                    'coordination_types': list(activity['coordination_types']),
                    'coordination_diversity': coordination_diversity,
                    'confidence': min(participation_rate + (coordination_diversity * 0.2), 1.0),
                    'evidence_strength': 'HIGH' if participation_rate > 0.5 else 'MEDIUM'
                })

        return cross_burst_evidence

    def _find_temporal_synchronization(self, burst_idx: int, burst_posts: pd.DataFrame, time_col: str) -> List[Dict]:
        """
        Find accounts that post within suspicious time windows (temporal synchronization).
        This catches coordinated accounts that post at nearly identical times.
        """
        evidence = []
        sync_threshold = pd.Timedelta(minutes=5)  # Posts within 5 minutes = suspicious

        # Group posts by account with timestamps
        account_times = defaultdict(list)

        for _, row in burst_posts.iterrows():
            account = row['account.username']
            post_time = pd.to_datetime(row.get(time_col))
            content = row.get('content_cleaned', '')

            if pd.notna(post_time) and content and not content.strip().lower().startswith('rt @'):
                account_times[account].append({
                    'time': post_time,
                    'content': content[:100]  # Store snippet for evidence
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

                # Flag if multiple synchronized posts or very tight timing
                if len(synchronized_pairs) >= 2 or (len(synchronized_pairs) >= 1 and
                    min(pair['time_diff_seconds'] for pair in synchronized_pairs) <= 60):

                    avg_sync_time = np.mean([pair['time_diff_seconds'] for pair in synchronized_pairs])
                    sync_count = len(synchronized_pairs)

                    # Calculate confidence based on sync count and timing precision
                    timing_precision = max(0, 1 - (avg_sync_time / 300))  # 0-1 scale based on 5min window
                    sync_strength = min(sync_count / 5.0, 1.0)  # Scale based on number of synced posts
                    confidence = (timing_precision + sync_strength) / 2

                    evidence.append({
                        'type': 'temporal_synchronization',
                        'burst_index': burst_idx,
                        'account1': account1,
                        'account2': account2,
                        'synchronized_posts': sync_count,
                        'avg_sync_time_seconds': avg_sync_time,
                        'confidence': confidence,
                        'evidence_strength': 'VERY_HIGH' if avg_sync_time <= 30 else 'HIGH' if avg_sync_time <= 120 else 'MEDIUM',
                        'sync_details': synchronized_pairs[:3]  # Store first 3 for evidence
                    })

        return evidence

    def _find_activity_fingerprinting(self, burst_contributors: List[Dict], posts_df: pd.DataFrame, time_col: str) -> List[Dict]:
        """
        Detect bot-like activity patterns and coordinated scheduling.
        This catches accounts with suspiciously similar activity fingerprints.
        """
        evidence = []

        # Extract activity patterns for each account across all bursts
        account_patterns = defaultdict(lambda: {
            'post_times': [],
            'burst_participation': [],
            'posting_intervals': [],
            'activity_hours': defaultdict(int),
            'total_posts': 0
        })

        # Collect activity data
        for burst_idx, contrib in enumerate(burst_contributors):
            start_time = pd.to_datetime(contrib['start_time'])
            end_time = pd.to_datetime(contrib['end_time'])

            # Get posts in this burst
            burst_mask = ((posts_df[time_col] >= start_time) & (posts_df[time_col] <= end_time))
            burst_posts = posts_df[burst_mask]

            for _, row in burst_posts.iterrows():
                account = row['account.username']
                post_time = pd.to_datetime(row[time_col])

                if pd.notna(post_time):
                    account_patterns[account]['post_times'].append(post_time)
                    account_patterns[account]['burst_participation'].append(burst_idx)
                    account_patterns[account]['activity_hours'][post_time.hour] += 1
                    account_patterns[account]['total_posts'] += 1

        # Calculate activity fingerprints
        for account, pattern in account_patterns.items():
            if len(pattern['post_times']) >= 5:  # Need sufficient data
                # Calculate posting intervals
                sorted_times = sorted(pattern['post_times'])
                intervals = [(sorted_times[i+1] - sorted_times[i]).total_seconds()
                           for i in range(len(sorted_times)-1)]
                pattern['posting_intervals'] = intervals

                # Calculate activity metrics
                pattern['avg_interval'] = np.mean(intervals) if intervals else 0
                pattern['interval_std'] = np.std(intervals) if intervals else 0
                pattern['interval_consistency'] = 1 - (pattern['interval_std'] / max(pattern['avg_interval'], 1))
                pattern['burst_participation_rate'] = len(set(pattern['burst_participation'])) / len(burst_contributors)
                pattern['activity_spread'] = np.std(list(pattern['activity_hours'].values()))

        # Find accounts with suspiciously similar patterns
        accounts_with_patterns = [(acc, pat) for acc, pat in account_patterns.items()
                                if len(pat['post_times']) >= 5]

        for i, (account1, pattern1) in enumerate(accounts_with_patterns):
            for account2, pattern2 in accounts_with_patterns[i+1:]:

                # Calculate pattern similarity metrics
                similarities = []

                # 1. Interval consistency similarity (both very consistent = suspicious)
                consistency1 = pattern1.get('interval_consistency', 0)
                consistency2 = pattern2.get('interval_consistency', 0)
                if consistency1 > 0.8 and consistency2 > 0.8:  # Both highly consistent
                    consistency_sim = 1 - abs(consistency1 - consistency2)
                    similarities.append(('interval_consistency', consistency_sim))

                # 2. Activity hour distribution similarity
                hours1 = pattern1['activity_hours']
                hours2 = pattern2['activity_hours']

                # Calculate hour overlap
                shared_hours = set(hours1.keys()) & set(hours2.keys())
                if len(shared_hours) >= 3:  # At least 3 shared active hours
                    hour_similarity = len(shared_hours) / len(set(hours1.keys()) | set(hours2.keys()))
                    similarities.append(('activity_hours', hour_similarity))

                # 3. Burst participation similarity
                bursts1 = set(pattern1['burst_participation'])
                bursts2 = set(pattern2['burst_participation'])
                burst_overlap = len(bursts1 & bursts2) / len(bursts1 | bursts2) if (bursts1 | bursts2) else 0
                if burst_overlap > 0.5:  # Significant overlap in burst participation
                    similarities.append(('burst_participation', burst_overlap))

                # 4. Average interval similarity (similar posting frequency)
                if pattern1['avg_interval'] > 0 and pattern2['avg_interval'] > 0:
                    interval_ratio = min(pattern1['avg_interval'], pattern2['avg_interval']) / max(pattern1['avg_interval'], pattern2['avg_interval'])
                    if interval_ratio > 0.7:  # Similar posting intervals
                        similarities.append(('posting_intervals', interval_ratio))

                # Calculate overall behavioral similarity
                if len(similarities) >= 2:  # Multiple similarity signals
                    avg_similarity = np.mean([sim for _, sim in similarities])
                    similarity_types = [sim_type for sim_type, _ in similarities]

                    # Higher confidence for multiple types of similarity
                    type_bonus = len(similarities) * 0.1
                    confidence = min(avg_similarity + type_bonus, 1.0)

                    if confidence >= 0.7:  # Threshold for suspicious similarity
                        evidence.append({
                            'type': 'behavioral_fingerprint',
                            'account1': account1,
                            'account2': account2,
                            'behavioral_similarity': avg_similarity,
                            'similarity_types': similarity_types,
                            'confidence': confidence,
                            'evidence_strength': 'VERY_HIGH' if confidence > 0.9 else 'HIGH' if confidence > 0.8 else 'MEDIUM',
                            'pattern_details': {
                                'account1_consistency': consistency1,
                                'account2_consistency': consistency2,
                                'shared_active_hours': len(shared_hours) if 'shared_hours' in locals() else 0,
                                'burst_overlap_rate': burst_overlap
                            }
                        })

        return evidence

    def _find_ultra_conservative_temporal_sync(self, burst_idx: int, burst_posts: pd.DataFrame, time_col: str) -> List[Dict]:
        """
        Ultra-conservative temporal synchronization detection (Step 1).
        Only flags accounts posting within 30 seconds with 3+ synchronized posts.
        This catches only the most obvious coordinated timing patterns.
        """
        evidence = []
        sync_threshold = pd.Timedelta(seconds=30)  # Very tight: 30 seconds (not 5 minutes)

        # Group posts by account with timestamps
        account_times = defaultdict(list)

        for _, row in burst_posts.iterrows():
            account = row['account.username']
            post_time = pd.to_datetime(row.get(time_col))
            content = row.get('content_cleaned', '')

            if pd.notna(post_time) and content and not content.strip().lower().startswith('rt @'):
                account_times[account].append({
                    'time': post_time,
                    'content': content[:100]  # Store snippet for evidence
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

                # Ultra-conservative: require 3+ synchronized posts (not 1-2)
                if len(synchronized_pairs) >= 3:

                    avg_sync_time = np.mean([pair['time_diff_seconds'] for pair in synchronized_pairs])
                    sync_count = len(synchronized_pairs)

                    # Calculate confidence based on sync count and timing precision
                    timing_precision = max(0, 1 - (avg_sync_time / 30))  # 0-1 scale based on 30sec window
                    sync_strength = min(sync_count / 3.0, 1.0)  # Scale based on number of synced posts (3+ required)
                    confidence = (timing_precision + sync_strength) / 2

                    # Only flag if confidence is high enough (80%+)
                    if confidence >= 0.8:
                        evidence.append({
                            'type': 'ultra_conservative_temporal_sync',
                            'burst_index': burst_idx,
                            'account1': account1,
                            'account2': account2,
                            'synchronized_posts': sync_count,
                            'avg_sync_time_seconds': avg_sync_time,
                            'confidence': confidence,
                            'evidence_strength': 'VERY_HIGH' if avg_sync_time <= 10 else 'HIGH',
                            'sync_details': synchronized_pairs[:3]  # Store first 3 for evidence
                        })

        return evidence

    def _find_mechanical_posting_intervals(self, burst_contributors: List[Dict], posts_df: pd.DataFrame, time_col: str) -> List[Dict]:
        """
        Ultra-conservative mechanical posting intervals detection (Step 2).
        Only flags accounts with 95%+ identical posting intervals under 1 hour.
        This catches only the most obvious bot-like mechanical behavior.
        """
        evidence = []

        # Extract posting patterns for accounts with sufficient activity
        account_patterns = defaultdict(lambda: {
            'post_times': [],
            'intervals': [],
            'interval_consistency': 0,
            'avg_interval': 0
        })

        # Collect posting times for each account across all data
        for _, row in posts_df.iterrows():
            account = row['account.username']
            post_time = pd.to_datetime(row[time_col])
            content = row.get('content_cleaned', '')

            # Skip retweets - focus on original posting patterns
            if pd.notna(post_time) and content and not content.strip().lower().startswith('rt @'):
                account_patterns[account]['post_times'].append(post_time)

        # Calculate interval consistency for each account
        for account, pattern in account_patterns.items():
            if len(pattern['post_times']) >= 10:  # Need at least 10 posts for reliable pattern
                # Sort times and calculate intervals
                sorted_times = sorted(pattern['post_times'])
                intervals = [(sorted_times[i+1] - sorted_times[i]).total_seconds()
                           for i in range(len(sorted_times)-1)]

                if intervals:
                    pattern['intervals'] = intervals
                    pattern['avg_interval'] = np.mean(intervals)
                    pattern['interval_std'] = np.std(intervals)

                    # Calculate consistency (higher = more mechanical)
                    if pattern['avg_interval'] > 0:
                        pattern['interval_consistency'] = 1 - (pattern['interval_std'] / pattern['avg_interval'])
                    else:
                        pattern['interval_consistency'] = 0

        # Find pairs of accounts with suspiciously similar mechanical patterns
        accounts_with_patterns = [(acc, pat) for acc, pat in account_patterns.items()
                                if len(pat['post_times']) >= 10 and pat['interval_consistency'] > 0.95]

        for i, (account1, pattern1) in enumerate(accounts_with_patterns):
            for account2, pattern2 in accounts_with_patterns[i+1:]:

                # Both accounts must have very high consistency (95%+)
                consistency1 = pattern1['interval_consistency']
                consistency2 = pattern2['interval_consistency']

                # Both must post at similar intervals (within 1 hour)
                interval1 = pattern1['avg_interval']
                interval2 = pattern2['avg_interval']

                # Ultra-conservative: only flag if intervals are under 1 hour (3600 seconds)
                if interval1 <= 3600 and interval2 <= 3600:

                    # Calculate interval similarity
                    interval_ratio = min(interval1, interval2) / max(interval1, interval2)

                    # Ultra-conservative: require very similar intervals (90%+ similarity)
                    if interval_ratio >= 0.9:

                        # Calculate overall confidence
                        consistency_similarity = 1 - abs(consistency1 - consistency2)
                        combined_confidence = (consistency_similarity + interval_ratio) / 2

                        # Only flag if confidence is very high (90%+)
                        if combined_confidence >= 0.9:

                            evidence.append({
                                'type': 'mechanical_posting_intervals',
                                'account1': account1,
                                'account2': account2,
                                'account1_consistency': consistency1,
                                'account2_consistency': consistency2,
                                'account1_avg_interval_minutes': interval1 / 60,
                                'account2_avg_interval_minutes': interval2 / 60,
                                'interval_similarity': interval_ratio,
                                'consistency_similarity': consistency_similarity,
                                'confidence': combined_confidence,
                                'evidence_strength': 'VERY_HIGH',
                                'pattern_details': {
                                    'account1_posts': len(pattern1['post_times']),
                                    'account2_posts': len(pattern2['post_times']),
                                    'both_highly_consistent': True,
                                    'both_sub_hourly': True
                                }
                            })

        return evidence

    def _find_identical_activity_hours(self, burst_contributors: List[Dict], posts_df: pd.DataFrame, time_col: str) -> List[Dict]:
        """
        Ultra-conservative identical activity hours detection (Step 3).
        Only flags accounts with 80%+ identical active hours AND very narrow activity windows (≤5 hours).
        This catches only the most obvious coordinated scheduling patterns.
        """
        evidence = []

        # Extract activity hour patterns for accounts with sufficient activity
        account_activity_hours = defaultdict(lambda: {
            'hourly_counts': defaultdict(int),
            'total_posts': 0,
            'active_hours': set(),
            'focused_hours': 0
        })

        # Collect hourly activity data for each account across all data
        for _, row in posts_df.iterrows():
            account = row['account.username']
            post_time = pd.to_datetime(row[time_col])
            content = row.get('content_cleaned', '')

            # Skip retweets - focus on original posting activity patterns
            if pd.notna(post_time) and content and not content.strip().lower().startswith('rt @'):
                hour = post_time.hour
                account_activity_hours[account]['hourly_counts'][hour] += 1
                account_activity_hours[account]['total_posts'] += 1
                account_activity_hours[account]['active_hours'].add(hour)

        # Calculate activity metrics for each account
        for account, activity in account_activity_hours.items():
            if activity['total_posts'] >= 20:  # Need at least 20 posts for reliable pattern
                # Count hours with significant activity (>= 5% of total posts)
                min_posts_per_hour = max(2, activity['total_posts'] * 0.05)
                focused_hours = sum(1 for count in activity['hourly_counts'].values()
                                  if count >= min_posts_per_hour)
                activity['focused_hours'] = focused_hours

        # Find pairs of accounts with suspiciously identical activity windows
        accounts_with_patterns = [(acc, act) for acc, act in account_activity_hours.items()
                                if act['total_posts'] >= 20 and act['focused_hours'] <= 5]  # Very narrow activity window

        for i, (account1, activity1) in enumerate(accounts_with_patterns):
            for account2, activity2 in accounts_with_patterns[i+1:]:

                # Get active hours for both accounts
                hours1 = activity1['active_hours']
                hours2 = activity2['active_hours']

                # Calculate hour overlap
                shared_hours = hours1 & hours2
                all_hours = hours1 | hours2

                if len(shared_hours) >= 3 and len(all_hours) >= 3:  # Need meaningful overlap

                    # Calculate Jaccard similarity for active hours
                    hour_similarity = len(shared_hours) / len(all_hours)

                    # Ultra-conservative: require very high similarity (80%+)
                    if hour_similarity >= 0.8:

                        # Both accounts must have narrow activity windows (≤5 focused hours)
                        focused1 = activity1['focused_hours']
                        focused2 = activity2['focused_hours']

                        if focused1 <= 5 and focused2 <= 5:

                            # Calculate additional similarity metrics
                            posts1 = activity1['total_posts']
                            posts2 = activity2['total_posts']

                            # Calculate intensity similarity within shared hours
                            intensity_similarity = 0
                            if shared_hours:
                                intensity_diffs = []
                                for hour in shared_hours:
                                    pct1 = activity1['hourly_counts'][hour] / posts1
                                    pct2 = activity2['hourly_counts'][hour] / posts2
                                    intensity_diffs.append(1 - abs(pct1 - pct2))
                                intensity_similarity = np.mean(intensity_diffs)

                            # Combine similarities for overall confidence
                            combined_confidence = (hour_similarity + intensity_similarity) / 2

                            # Ultra-conservative: require very high combined confidence (85%+)
                            if combined_confidence >= 0.85:

                                evidence.append({
                                    'type': 'identical_activity_hours',
                                    'account1': account1,
                                    'account2': account2,
                                    'shared_active_hours': sorted(list(shared_hours)),
                                    'hour_similarity': hour_similarity,
                                    'intensity_similarity': intensity_similarity,
                                    'account1_focused_hours': focused1,
                                    'account2_focused_hours': focused2,
                                    'account1_total_posts': posts1,
                                    'account2_total_posts': posts2,
                                    'confidence': combined_confidence,
                                    'evidence_strength': 'VERY_HIGH',
                                    'pattern_details': {
                                        'both_narrow_windows': True,
                                        'high_hour_overlap': True,
                                        'similar_intensity': intensity_similarity > 0.7
                                    }
                                })

        return evidence

    def _analyze_rt_temporal_synchronization(self, rt_timestamps: List[Dict]) -> Dict:
        """
        Analyze temporal synchronization patterns in retweet timestamps.
        Detects clusters of RTs that happen within suspicious time windows.

        Args:
            rt_timestamps: List of dicts with 'retweeter', 'timestamp', 'content'

        Returns:
            Dict with synchronization analysis results
        """
        if len(rt_timestamps) < 3:
            return {
                'synchronized_clusters': [],
                'max_cluster_size': 0,
                'avg_cluster_timing_precision': 0,
                'total_synchronized_rts': 0
            }

        # Sort by timestamp
        sorted_rts = sorted(rt_timestamps, key=lambda x: x['timestamp'])

        # Find temporal clusters (RTs within 60 seconds of each other)
        sync_threshold = pd.Timedelta(seconds=60)  # Conservative threshold for RTs
        clusters = []
        current_cluster = [sorted_rts[0]]

        for rt in sorted_rts[1:]:
            # Check if this RT is within sync threshold of the last RT in current cluster
            time_diff = rt['timestamp'] - current_cluster[-1]['timestamp']

            if time_diff <= sync_threshold:
                current_cluster.append(rt)
            else:
                # Close current cluster if it has 3+ RTs
                if len(current_cluster) >= 3:
                    clusters.append(current_cluster)
                # Start new cluster
                current_cluster = [rt]

        # Don't forget the last cluster
        if len(current_cluster) >= 3:
            clusters.append(current_cluster)

        # Analyze cluster quality
        synchronized_clusters = []
        total_synchronized_rts = 0

        for cluster in clusters:
            if len(cluster) >= 3:  # Require at least 3 synchronized RTs
                # Calculate cluster timing precision
                timestamps = [rt['timestamp'] for rt in cluster]
                time_span = (max(timestamps) - min(timestamps)).total_seconds()

                # Calculate precision score (tighter timing = higher precision)
                if time_span <= 30:  # Very tight clustering (30 seconds)
                    timing_precision = 1.0
                elif time_span <= 60:  # Good clustering (60 seconds)
                    timing_precision = 0.8
                else:  # Loose clustering (up to sync_threshold)
                    timing_precision = 0.6

                cluster_info = {
                    'size': len(cluster),
                    'time_span_seconds': time_span,
                    'timing_precision': timing_precision,
                    'retweeters': [rt['retweeter'] for rt in cluster],
                    'start_time': min(timestamps),
                    'end_time': max(timestamps)
                }

                synchronized_clusters.append(cluster_info)
                total_synchronized_rts += len(cluster)

        # Calculate summary statistics
        max_cluster_size = max([c['size'] for c in synchronized_clusters]) if synchronized_clusters else 0
        avg_timing_precision = np.mean([c['timing_precision'] for c in synchronized_clusters]) if synchronized_clusters else 0

        return {
            'synchronized_clusters': synchronized_clusters,
            'max_cluster_size': max_cluster_size,
            'avg_cluster_timing_precision': avg_timing_precision,
            'total_synchronized_rts': total_synchronized_rts,
            'cluster_count': len(synchronized_clusters)
        }

    def _calculate_network_metrics(self, subgraph: nx.Graph) -> Dict:
        """Calculate advanced network metrics using NetworkX."""

        metrics = {}

        try:
            # Basic metrics
            metrics['nodes'] = subgraph.number_of_nodes()
            metrics['edges'] = subgraph.number_of_edges()
            metrics['density'] = nx.density(subgraph)

            # Centrality measures
            if subgraph.number_of_nodes() > 1:
                metrics['avg_clustering'] = nx.average_clustering(subgraph)

                # Degree centrality
                degree_centrality = nx.degree_centrality(subgraph)
                metrics['degree_centrality'] = degree_centrality  # Store the full dict for hub account extraction
                metrics['max_degree_centrality'] = max(degree_centrality.values())
                metrics['avg_degree_centrality'] = np.mean(list(degree_centrality.values()))

                # Find most central account
                most_central = max(degree_centrality, key=degree_centrality.get)
                metrics['most_central_account'] = most_central
                metrics['most_central_score'] = degree_centrality[most_central]

                # Betweenness centrality (for networks with 3+ nodes)
                if subgraph.number_of_nodes() >= 3:
                    betweenness = nx.betweenness_centrality(subgraph)
                    metrics['max_betweenness'] = max(betweenness.values())
                    metrics['avg_betweenness'] = np.mean(list(betweenness.values()))

                    # Find bridge accounts (high betweenness)
                    bridge_accounts = {k: v for k, v in betweenness.items() if v > 0.1}
                    metrics['bridge_accounts'] = bridge_accounts

                # Clustering coefficient distribution
                clustering_coeffs = nx.clustering(subgraph)
                metrics['clustering_coefficients'] = clustering_coeffs

        except Exception as e:
            print(f"⚠️ Error calculating network metrics: {e}")
            metrics = {'error': str(e)}

        return metrics

    def _detect_communities(self, subgraph: nx.Graph) -> Dict:
        """Detect communities within the network using multiple algorithms."""

        communities = {}

        try:
            if subgraph.number_of_nodes() >= 3:
                # Greedy modularity communities
                greedy_communities = list(nx.community.greedy_modularity_communities(subgraph))
                communities['greedy_modularity'] = {
                    'communities': [list(community) for community in greedy_communities],
                    'count': len(greedy_communities),
                    'modularity': nx.community.modularity(subgraph, greedy_communities)
                }

                # Label propagation (alternative method)
                try:
                    label_prop_communities = list(nx.community.label_propagation_communities(subgraph))
                    communities['label_propagation'] = {
                        'communities': [list(community) for community in label_prop_communities],
                        'count': len(label_prop_communities)
                    }
                except:
                    pass

        except Exception as e:
            print(f"⚠️ Community detection failed: {e}")
            communities = {'error': str(e)}

        return communities

    def _analyze_network_structure(self, subgraph: nx.Graph) -> Dict:
        """Analyze whether network is hierarchical vs distributed."""

        structure = {}

        try:
            if subgraph.number_of_nodes() <= 2:
                structure['type'] = 'SIMPLE_PAIR'
                return structure

            # Calculate degree distribution
            degrees = [d for n, d in subgraph.degree()]

            if not degrees:
                return {'type': 'EMPTY'}

            structure['degree_stats'] = {
                'max': max(degrees),
                'mean': np.mean(degrees),
                'std': np.std(degrees),
                'degrees': degrees
            }

            # Calculate Gini coefficient for degree distribution
            gini = self._calculate_gini_coefficient(degrees)
            structure['gini_coefficient'] = gini

            # Classify network structure
            if gini > 0.6:
                structure['type'] = 'HIERARCHICAL'  # Few nodes with many connections
                structure['description'] = 'Hub-and-spoke structure with central coordinators'
            elif gini < 0.3:
                structure['type'] = 'DISTRIBUTED'   # Connections spread evenly
                structure['description'] = 'Peer-to-peer coordination structure'
            else:
                structure['type'] = 'MIXED'         # Hybrid structure
                structure['description'] = 'Mixed coordination structure'

            # Find potential hub accounts (high degree)
            if degrees:
                degree_threshold = np.mean(degrees) + np.std(degrees)
                hub_accounts = [node for node, degree in subgraph.degree() if degree >= degree_threshold]
                structure['potential_hubs'] = hub_accounts
                structure['hub_count'] = len(hub_accounts)

        except Exception as e:
            print(f"⚠️ Error analyzing network structure: {e}")
            structure = {'type': 'ERROR', 'error': str(e)}

        return structure

    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for measuring inequality."""
        if not values or len(values) == 0:
            return 0.0

        # Sort values
        sorted_values = sorted(values)
        n = len(values)

        if n == 1:
            return 0.0

        # Calculate Gini coefficient
        total = sum(sorted_values)
        if total == 0:
            return 0.0

        cumsum = 0
        gini_sum = 0

        for i, value in enumerate(sorted_values):
            cumsum += value
            gini_sum += (2 * (i + 1) - n - 1) * value

        return gini_sum / (n * total)

    def _map_to_global_communities(self, network_accounts: List[str], global_communities: List[Set]) -> Dict:
        """Map network accounts to global communities."""

        mapping = {}
        network_set = set(network_accounts)

        for i, community in enumerate(global_communities):
            overlap = network_set & community
            if overlap:
                mapping[f'global_community_{i + 1}'] = {
                    'overlapping_accounts': list(overlap),
                    'overlap_size': len(overlap),
                    'coverage': len(overlap) / len(network_accounts)
                }

        return mapping

    def visualize_network(self, network: Dict, show_labels: bool = True, file_path: Optional[str] = None):
        """Visualize the coordination network using NetworkX and Matplotlib."""

        # Extract NetworkX subgraph
        G = network['subgraph']

        plt.figure(figsize=(12, 12))

        # Draw the network
        pos = nx.spring_layout(G, seed=42)  # Fixed seed for reproducibility

        # Node color by risk level
        risk_color_map = {
            'VERY_HIGH': 'red',
            'HIGH': 'orange',
            'MEDIUM': 'yellow',
            'LOW': 'green'
        }

        node_colors = [risk_color_map.get(network['risk_level'], 'gray') for node in G.nodes()]

        # Edge width by confidence
        edge_widths = [G[u][v]['weight'] * 5 for u, v in G.edges()]

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.9)
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color='black')

        # Draw labels
        if show_labels:
            nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

        plt.title(f"Coordination Network Visualization (Risk Level: {network['risk_level']})", fontsize=14)
        plt.axis('off')

        # Save to file if path is provided
        if file_path:
            plt.savefig(file_path, format='PNG', dpi=300, bbox_inches='tight')
            print(f"📸 Network visualization saved to: {file_path}")

        plt.show()

    def generate_visualizations(self, results: Dict):
        """Generate comprehensive visualizations for coordination analysis results."""

        networks = results.get('coordination_networks', [])

        if not networks:
            print("❌ No networks found - skipping visualizations")
            return

        try:
            # Create plots directory
            plots_dir = "plots"
            os.makedirs(plots_dir, exist_ok=True)

            print(f"📊 Generating visualizations for {len(networks)} networks...")

            # 1. Network Overview Charts
            print("📈 Creating network metrics overview...")
            self.plot_network_metrics(results)
            self.plot_top_networks(results, top_n=min(5, len(networks)))

            # 2. Individual Network Visualizations (for smaller networks)
            print("🕸️ Creating individual network visualizations...")

            # Filter out networks that are too large to visualize clearly
            visualizable_networks = [net for net in networks if net['size'] <= 50]
            large_networks = [net for net in networks if net['size'] > 50]

            if large_networks:
                print(f"   ℹ️ Skipping {len(large_networks)} large network(s) (>50 accounts):")
                for net in large_networks:
                    print(f"      • Network {net['network_id']}: {net['size']} accounts (too large for clear visualization)")

            if visualizable_networks:
                for i, network in enumerate(visualizable_networks[:5]):  # Top 5 visualizable networks
                    try:
                        file_path = os.path.join(plots_dir, f"network_{network['network_id']}_visualization.png")
                        self.visualize_network(network, show_labels=True, file_path=file_path)
                        print(f"   ✅ Network {network['network_id']} ({network['size']} accounts) - {network['risk_level']} risk")
                    except Exception as e:
                        print(f"   ⚠️ Failed to visualize Network {network['network_id']}: {e}")
            else:
                print("   ℹ️ All networks too large for detailed visualization (>50 accounts)")

            # 3. Summary Dashboard
            print("📊 Creating summary dashboard...")
            self.create_coordination_dashboard(results)

            print(f"✅ Visualizations complete! Check the 'plots' directory for all generated files.")

        except Exception as e:
            print(f"⚠️ Visualization generation failed: {e}")

    def plot_network_metrics(self, results: Dict):
        """Plot network metrics such as size, average confidence, and risk level."""

        networks = results['coordination_networks']

        if not networks:
            print("❌ No networks to display")
            return

        # Prepare data for plotting
        data = {
            'Network ID': [],
            'Size': [],
            'Average Confidence': [],
            'Risk Level': []
        }

        for network in networks:
            data['Network ID'].append(network['network_id'])
            data['Size'].append(network['size'])
            data['Average Confidence'].append(network['avg_confidence'])
            data['Risk Level'].append(network['risk_level'])

        df = pd.DataFrame(data)

        # Plot
        fig = px.scatter(df, x='Size', y='Average Confidence', color='Risk Level',
                        title="Coordination Networks: Size vs. Average Confidence",
                        labels={"Size": "Network Size (Number of Accounts)", "Average Confidence": "Average Confidence"},
                        hover_data=['Network ID'])

        fig.update_traces(marker=dict(size=10, line=dict(width=2, color='DarkSlateGrey')),
                        selector=dict(mode='markers'))

        fig.show()

        # Save to file
        file_path = "plots/network_metrics_scatter.html"
        fig.write_html(file_path)
        print(f"📈 Network metrics plot saved to: {file_path}")

        # Also try to save PNG
        try:
            png_path = "plots/network_metrics_scatter.png"
            fig.write_image(png_path, scale=2)
            print(f"📈 Network metrics PNG saved to: {png_path}")
        except Exception as e:
            print(f"⚠️ Could not save PNG: {e}")

    def plot_top_networks(self, results: Dict, top_n: int = 5):
        """Plot the top N networks by size and average confidence."""

        networks = results['coordination_networks']

        if not networks:
            print("❌ No networks to display")
            return

        # Sort networks by size and confidence
        sorted_networks = sorted(networks, key=lambda x: (x['size'], x['avg_confidence']), reverse=True)
        top_networks = sorted_networks[:top_n]

        # Prepare data for plotting
        data = {
            'Network ID': [],
            'Size': [],
            'Average Confidence': [],
            'Risk Level': []
        }

        for network in top_networks:
            data['Network ID'].append(f"Network {network['network_id']}")
            data['Size'].append(network['size'])
            data['Average Confidence'].append(network['avg_confidence'])
            data['Risk Level'].append(network['risk_level'])

        df = pd.DataFrame(data)

        # Plot
        fig = px.bar(df, x='Network ID', y='Size',
                    title=f"Top {top_n} Coordination Networks by Size",
                    labels={"Size": "Network Size (Number of Accounts)", "Network ID": "Network ID"},
                    color='Risk Level', text='Size')

        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

        fig.show()

        # Save to file
        file_path = f"plots/top_{top_n}_networks_bar_chart.html"
        fig.write_html(file_path)
        print(f"📊 Top networks bar chart saved to: {file_path}")

        # Also try to save PNG
        try:
            png_path = f"plots/top_{top_n}_networks_bar_chart.png"
            fig.write_image(png_path, scale=2)
            print(f"📊 Top networks PNG saved to: {png_path}")
        except Exception as e:
            print(f"⚠️ Could not save PNG: {e}")

    def create_coordination_dashboard(self, results: Dict):
        """Create a comprehensive dashboard of coordination analysis results."""

        networks = results.get('coordination_networks', [])
        stats = results.get('summary_stats', {})

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Networks by Risk Level",
                "Network Size Distribution",
                "Evidence Type Distribution",
                "Confidence vs Size"
            ),
            specs=[[{"type": "pie"}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )

        # 1. Networks by Risk Level (Pie Chart)
        risk_counts = Counter([net['risk_level'] for net in networks])
        fig.add_trace(
            go.Pie(labels=list(risk_counts.keys()), values=list(risk_counts.values()),
                   name="Risk Levels", showlegend=True),
            row=1, col=1
        )

        # 2. Network Size Distribution (Histogram)
        sizes = [net['size'] for net in networks]
        fig.add_trace(
            go.Histogram(x=sizes, name="Network Sizes", showlegend=False, nbinsx=10),
            row=1, col=2
        )

        # 3. Evidence Type Distribution (Bar Chart)
        all_evidence_types = []
        for net in networks:
            all_evidence_types.extend(net['evidence_types'])
        evidence_counts = Counter(all_evidence_types)

        fig.add_trace(
            go.Bar(x=list(evidence_counts.keys()), y=list(evidence_counts.values()),
                   name="Evidence Types", showlegend=False),
            row=2, col=1
        )

        # 4. Confidence vs Size (Scatter)
        confidences = [net['avg_confidence'] for net in networks]
        colors = [net['risk_level'] for net in networks]

        fig.add_trace(
            go.Scatter(x=sizes, y=confidences, mode='markers',
                       marker=dict(size=10, opacity=0.7),
                       text=[f"Network {net['network_id']}" for net in networks],
                       name="Networks", showlegend=False),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title_text="Coordination Networks Analysis Dashboard",
            title_x=0.5,
            height=800,
            showlegend=True
        )

        # Update axes labels
        fig.update_xaxes(title_text="Network Size", row=2, col=2)
        fig.update_yaxes(title_text="Average Confidence", row=2, col=2)
        fig.update_xaxes(title_text="Evidence Type", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_xaxes(title_text="Network Size", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)

        # Save dashboard
        dashboard_path = "plots/coordination_dashboard.html"
        fig.write_html(dashboard_path)
        print(f"📊 Coordination dashboard saved to: {dashboard_path}")

        # Also save as PNG
        try:
            png_path = "plots/coordination_dashboard.png"
            fig.write_image(png_path, width=1200, height=800, scale=2)
            print(f"📊 Dashboard PNG saved to: {png_path}")
        except Exception as e:
            print(f"⚠️ Could not save PNG dashboard: {e}")
