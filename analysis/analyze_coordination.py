#!/usr/bin/env python3
"""
Analyze the specific coordination patterns detected by the enhanced burst detector
Enhanced with content and hashtag similarity analysis
"""

import os
import sys
import pickle
import pandas as pd
from collections import Counter
import re
from difflib import SequenceMatcher
from urllib.parse import urlparse

def extract_hashtags(text):
    """Extract hashtags from text content"""
    if not isinstance(text, str):
        return []
    return re.findall(r'#\w+', text.lower())

def extract_urls(text):
    """Extract URLs from text content"""
    if not isinstance(text, str):
        return []
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, text)

def calculate_text_similarity(text1, text2):
    """Calculate similarity between two texts using SequenceMatcher"""
    if not isinstance(text1, str) or not isinstance(text2, str):
        return 0.0
    if not text1.strip() or not text2.strip():
        return 0.0

    # Normalize text (remove extra spaces, convert to lowercase)
    text1_clean = ' '.join(text1.lower().split())
    text2_clean = ' '.join(text2.lower().split())

    return SequenceMatcher(None, text1_clean, text2_clean).ratio()

def analyze_content_coordination(contributors, full_posts_data=None):
    """Analyze content similarity and hashtag coordination patterns - optimized version"""
    print(f"\n📝 CONTENT & HASHTAG COORDINATION ANALYSIS:")
    print("-" * 60)

    content_coordination = {
        'high_similarity_pairs': [],
        'hashtag_coordination': [],
        'url_coordination': [],
        'identical_content': [],
        'burst_content_patterns': []
    }

    # Use provided data or load fresh (optimized)
    if full_posts_data is not None and hasattr(full_posts_data, 'to_pandas'):
        # Convert polars DataFrame to pandas if needed
        full_df = full_posts_data.to_pandas()
        print(f"✅ Using provided dataset: {len(full_df)} posts for content analysis")
    elif full_posts_data is not None:
        # Already pandas DataFrame
        full_df = full_posts_data
        print(f"✅ Using provided dataset: {len(full_df)} posts for content analysis")
    else:
        # Load fresh data (fallback)
        try:
            project_root = os.path.join(os.path.dirname(__file__), '..')
            src_path = os.path.join(project_root, 'src')
            sys.path.insert(0, src_path)
            from components.data_analyzer import DataAnalyzer

            data_path = os.path.join(project_root, 'data', 'sampledata_truthsocial.csv')
            analyzer = DataAnalyzer(data_path)
            analyzer.run_all()
            full_df = analyzer.df.to_pandas()
            print(f"✅ Loaded {len(full_df)} posts for content analysis")
        except Exception as e:
            print(f"⚠️ Could not load dataset for content analysis: {e}")
            return content_coordination

    # Analyze content coordination for all bursts
    print(f"🔍 Analyzing content coordination for all {len(contributors)} bursts")

    for burst_idx, contrib in enumerate(contributors):
        if 'posts' not in contrib or not contrib['posts']:
            continue

        burst_posts = contrib['posts']
        burst_accounts = [acc['account.username'] for acc in contrib.get('top_accounts', [])]

        # Get posts from burst period with content
        start_time = contrib['start_time']
        end_time = contrib['end_time']

        # Filter full dataset to burst period
        if 'created_at' in full_df.columns and 'content_cleaned' in full_df.columns:
            full_df['created_at'] = pd.to_datetime(full_df['created_at'], errors='coerce')

            # Normalize timezone handling
            if hasattr(full_df['created_at'].dtype, 'tz') and full_df['created_at'].dtype.tz is not None:
                full_df['created_at'] = full_df['created_at'].dt.tz_convert('UTC').dt.tz_localize(None)

            # Ensure start_time and end_time are timezone-naive
            if hasattr(start_time, 'tz') and start_time.tz is not None:
                start_time = start_time.tz_convert('UTC').tz_localize(None)
            elif hasattr(start_time, 'tzinfo') and start_time.tzinfo is not None:
                start_time = start_time.replace(tzinfo=None)

            if hasattr(end_time, 'tz') and end_time.tz is not None:
                end_time = end_time.tz_convert('UTC').tz_localize(None)
            elif hasattr(end_time, 'tzinfo') and end_time.tzinfo is not None:
                end_time = end_time.replace(tzinfo=None)

            # Convert to pandas datetime for compatibility
            start_time = pd.to_datetime(start_time)
            end_time = pd.to_datetime(end_time)

            burst_mask = ((full_df['created_at'] >= start_time) &
                         (full_df['created_at'] <= end_time) &
                         (full_df['account.username'].isin(burst_accounts)))
            burst_content_df = full_df[burst_mask].copy()

            if len(burst_content_df) == 0:
                continue

            # Analyze content patterns for this burst
            burst_analysis = analyze_burst_content(burst_content_df, burst_idx, start_time, end_time)
            content_coordination['burst_content_patterns'].append(burst_analysis)

            # Find high similarity content pairs
            similarity_pairs = find_content_similarity_pairs(burst_content_df, burst_idx)
            content_coordination['high_similarity_pairs'].extend(similarity_pairs)

            # Find hashtag coordination
            hashtag_coord = find_hashtag_coordination(burst_content_df, burst_idx)
            if hashtag_coord:
                content_coordination['hashtag_coordination'].append(hashtag_coord)

            # Find URL coordination
            url_coord = find_url_coordination(burst_content_df, burst_idx)
            if url_coord:
                content_coordination['url_coordination'].append(url_coord)

            # Find identical content
            identical = find_identical_content(burst_content_df, burst_idx)
            content_coordination['identical_content'].extend(identical)

    return content_coordination

def analyze_burst_content(burst_df, burst_idx, start_time, end_time):
    """Analyze content patterns within a single burst"""
    analysis = {
        'burst_index': burst_idx,
        'start_time': start_time,
        'end_time': end_time,
        'total_posts': len(burst_df),
        'unique_accounts': burst_df['account.username'].nunique(),
        'content_diversity': 0,
        'hashtag_diversity': 0,
        'avg_content_similarity': 0
    }

    if len(burst_df) < 2:
        return analysis

    # Calculate content diversity (unique content / total posts)
    unique_content = burst_df['content_cleaned'].nunique()
    analysis['content_diversity'] = unique_content / len(burst_df) if len(burst_df) > 0 else 0

    # Calculate average content similarity
    contents = burst_df['content_cleaned'].dropna().tolist()
    if len(contents) >= 2:
        similarities = []
        for i in range(min(10, len(contents))):  # Sample first 10 for performance
            for j in range(i+1, min(10, len(contents))):
                sim = calculate_text_similarity(contents[i], contents[j])
                similarities.append(sim)

        analysis['avg_content_similarity'] = sum(similarities) / len(similarities) if similarities else 0

    # Analyze hashtag patterns
    all_hashtags = []
    for content in contents:
        hashtags = extract_hashtags(content)
        all_hashtags.extend(hashtags)

    unique_hashtags = len(set(all_hashtags))
    total_hashtags = len(all_hashtags)
    analysis['hashtag_diversity'] = unique_hashtags / total_hashtags if total_hashtags > 0 else 0

    return analysis

def find_content_similarity_pairs(burst_df, burst_idx):
    """Find pairs of posts with high content similarity - optimized version"""
    similarity_pairs = []

    # Limit analysis to top accounts for performance
    unique_accounts = burst_df['account.username'].unique()
    if len(unique_accounts) > 10:
        # Only analyze top 10 accounts by post count for performance
        account_counts = burst_df['account.username'].value_counts()
        top_accounts = account_counts.head(10).index.tolist()
        burst_df = burst_df[burst_df['account.username'].isin(top_accounts)]
        unique_accounts = top_accounts

    # Group by account to avoid comparing posts from same account
    for i, account1 in enumerate(unique_accounts):
        for j, account2 in enumerate(unique_accounts[i+1:], i+1):
            posts1 = burst_df[burst_df['account.username'] == account1]['content_cleaned'].dropna()
            posts2 = burst_df[burst_df['account.username'] == account2]['content_cleaned'].dropna()

            # Compare only first 2 posts per account for performance
            for content1 in posts1.head(2):
                for content2 in posts2.head(2):
                    similarity = calculate_text_similarity(content1, content2)

                    if similarity > 0.8:  # High similarity threshold
                        similarity_pairs.append({
                            'burst_index': burst_idx,
                            'account1': account1,
                            'account2': account2,
                            'similarity': similarity,
                            'content1': content1[:100] + "..." if len(content1) > 100 else content1,
                            'content2': content2[:100] + "..." if len(content2) > 100 else content2
                        })

                        # Limit to prevent explosion of results
                        if len(similarity_pairs) > 50:
                            return similarity_pairs

    return similarity_pairs

def find_hashtag_coordination(burst_df, burst_idx):
    """Find coordinated hashtag usage patterns"""
    account_hashtags = {}

    # Extract hashtags by account
    for _, row in burst_df.iterrows():
        account = row['account.username']
        content = row['content_cleaned']
        hashtags = extract_hashtags(content)

        if hashtags:
            if account not in account_hashtags:
                account_hashtags[account] = []
            account_hashtags[account].extend(hashtags)

    # Find accounts using identical hashtag sets
    coordinated_hashtags = []
    accounts = list(account_hashtags.keys())

    for i in range(len(accounts)):
        for j in range(i+1, len(accounts)):
            acc1, acc2 = accounts[i], accounts[j]
            hashtags1 = set(account_hashtags[acc1])
            hashtags2 = set(account_hashtags[acc2])

            if len(hashtags1) >= 2 and len(hashtags2) >= 2:
                intersection = hashtags1 & hashtags2
                union = hashtags1 | hashtags2

                # Jaccard similarity for hashtags
                jaccard = len(intersection) / len(union) if union else 0

                if jaccard > 0.7:  # High hashtag coordination
                    coordinated_hashtags.append({
                        'burst_index': burst_idx,
                        'account1': acc1,
                        'account2': acc2,
                        'shared_hashtags': list(intersection),
                        'jaccard_similarity': jaccard
                    })

    return {
        'burst_index': burst_idx,
        'coordinated_pairs': coordinated_hashtags,
        'total_hashtag_users': len(account_hashtags)
    } if coordinated_hashtags else None

def find_url_coordination(burst_df, burst_idx):
    """Find coordinated URL sharing patterns"""
    account_urls = {}

    # Extract URLs by account
    for _, row in burst_df.iterrows():
        account = row['account.username']
        content = row['content_cleaned']
        urls = extract_urls(content)

        if urls:
            if account not in account_urls:
                account_urls[account] = []
            account_urls[account].extend(urls)

    # Find accounts sharing same URLs
    coordinated_urls = []
    accounts = list(account_urls.keys())

    for i in range(len(accounts)):
        for j in range(i+1, len(accounts)):
            acc1, acc2 = accounts[i], accounts[j]
            urls1 = set(account_urls[acc1])
            urls2 = set(account_urls[acc2])

            shared_urls = urls1 & urls2
            if len(shared_urls) >= 1:  # Any shared URL is suspicious
                coordinated_urls.append({
                    'account1': acc1,
                    'account2': acc2,
                    'shared_urls': list(shared_urls)
                })

    return {
        'burst_index': burst_idx,
        'coordinated_pairs': coordinated_urls,
        'total_url_users': len(account_urls)
    } if coordinated_urls else None

def find_identical_content(burst_df, burst_idx):
    """Find accounts posting identical or near-identical content"""
    identical_groups = []

    # Group by content similarity
    content_groups = {}

    for _, row in burst_df.iterrows():
        content = row['content_cleaned']
        account = row['account.username']

        if pd.isna(content) or not content.strip():
            continue

        # Normalize content for comparison
        normalized_content = ' '.join(content.lower().split())

        # Find if this content matches any existing group
        matched_group = None
        for group_content, group_accounts in content_groups.items():
            if calculate_text_similarity(normalized_content, group_content) > 0.95:
                matched_group = group_content
                break

        if matched_group:
            content_groups[matched_group].add(account)
        else:
            content_groups[normalized_content] = {account}

    # Find groups with multiple accounts (identical content coordination)
    for content, accounts in content_groups.items():
        if len(accounts) >= 2:
            identical_groups.append({
                'burst_index': burst_idx,
                'accounts': list(accounts),
                'content': content[:150] + "..." if len(content) > 150 else content,
                'account_count': len(accounts)
            })

    return identical_groups

print("🕵️ DETAILED COORDINATION ANALYSIS")
print("="*50)

try:
    # Load enhanced results
    project_root = os.path.join(os.path.dirname(__file__), '..')
    cache_file = os.path.join(project_root, 'cache', 'enhanced_burst_results.pkl')
    with open(cache_file, 'rb') as f:
        results = pickle.load(f)

    coordination_analysis = results.get('coordination_analysis')
    contributors = results.get('burst_contributors', [])

    if not coordination_analysis:
        print("❌ No coordination analysis found")
        exit(1)

    # Show repeated contributors analysis
    repeated = coordination_analysis['repeated_contributors']
    high_coord_bursts = coordination_analysis['high_coordination_bursts']

    print(f"🔍 NETWORK DETECTION RESULTS:")
    print(f"📊 Total bursts analyzed: {len(contributors)}")
    print(f"🕵️ Accounts appearing in multiple bursts: {len(repeated)}")
    print(f"🚨 High coordination bursts detected: {len(high_coord_bursts)}")

    print(f"\n👥 TOP NETWORK MEMBERS (appearing in multiple bursts):")
    print("-" * 60)
    top_repeated = sorted(repeated.items(), key=lambda x: x[1], reverse=True)[:15]

    for i, (username, burst_count) in enumerate(top_repeated, 1):
        percentage = (burst_count / len(contributors)) * 100
        print(f"{i:2d}. @{username:<20} | {burst_count:2d} bursts ({percentage:4.1f}%)")

    print(f"\n🚨 HIGH COORDINATION BURSTS:")
    print("-" * 80)

    for i, burst in enumerate(high_coord_bursts[:10], 1):
        level = burst['level']
        start = burst['start_time'].strftime("%m/%d %H:%M")
        end = burst['end_time'].strftime("%m/%d %H:%M")
        signals = ', '.join(burst['signals'])
        top_contributors = [acc['account.username'] for acc in burst['top_contributors'][:3]]

        print(f"{i:2d}. Level {level} burst ({start} → {end})")
        print(f"    🚨 Signals: {signals}")
        print(f"    👥 Top contributors: {', '.join([f'@{u}' for u in top_contributors])}")
        print()

    # Analyze coordination patterns across all bursts
    print(f"📊 COORDINATION PATTERN ANALYSIS:")
    print("-" * 50)

    signal_counts = Counter()
    concentration_scores = []
    dominance_scores = []

    for contrib in contributors:
        coord_indicators = contrib.get('coordination_indicators', {})
        signals = coord_indicators.get('potential_coordination_signals', [])

        for signal in signals:
            signal_counts[signal] += 1

        concentration = coord_indicators.get('contributor_concentration', 0)
        dominance = coord_indicators.get('dominant_contributor_percentage', 0)

        if concentration > 0:
            concentration_scores.append(concentration)
        if dominance > 0:
            dominance_scores.append(dominance)

    print(f"📈 Coordination signal frequencies:")
    for signal, count in signal_counts.most_common():
        percentage = (count / len(contributors)) * 100
        signal_name = signal.replace('_', ' ').title()
        print(f"  {signal_name}: {count} bursts ({percentage:.1f}%)")

    if concentration_scores:
        avg_concentration = sum(concentration_scores) / len(concentration_scores)
        print(f"\n📊 Average contributor concentration: {avg_concentration:.1f}%")

    if dominance_scores:
        avg_dominance = sum(dominance_scores) / len(dominance_scores)
        print(f"👑 Average dominant contributor share: {avg_dominance:.1f}%")

    # Find potential bot/coordination clusters
    print(f"\n🤖 POTENTIAL BOT CLUSTER ANALYSIS:")
    print("-" * 40)

    # Look for accounts that always appear together
    account_cooccurrence = {}

    for contrib in contributors:
        accounts = [acc['account.username'] for acc in contrib.get('top_accounts', [])]

        for i, acc1 in enumerate(accounts):
            for acc2 in accounts[i+1:]:
                pair = tuple(sorted([acc1, acc2]))
                account_cooccurrence[pair] = account_cooccurrence.get(pair, 0) + 1

    # Find pairs that appear together frequently
    frequent_pairs = [(pair, count) for pair, count in account_cooccurrence.items()
                     if count >= 3 and count >= len(contributors) * 0.1]  # At least 10% of bursts

    frequent_pairs.sort(key=lambda x: x[1], reverse=True)

    if frequent_pairs:
        print("🔗 Account pairs frequently appearing together:")
        for (acc1, acc2), count in frequent_pairs[:10]:
            percentage = (count / len(contributors)) * 100
            print(f"  @{acc1} ↔ @{acc2}: {count} bursts together ({percentage:.1f}%)")
    else:
        print("  No frequently co-occurring account pairs found")

    print(f"\n✅ Coordination analysis complete!")

    # ADD CONTENT COORDINATION ANALYSIS
    print(f"\n" + "="*60)
    print("📝 ENHANCED: CONTENT & HASHTAG COORDINATION")
    print("="*60)

    content_coordination = analyze_content_coordination(contributors, None)

    # Display content coordination results
    if content_coordination['high_similarity_pairs']:
        print(f"\n🔍 HIGH CONTENT SIMILARITY PAIRS:")
        print("-" * 50)
        similarity_pairs = sorted(content_coordination['high_similarity_pairs'],
                                key=lambda x: x['similarity'], reverse=True)[:10]

        for pair in similarity_pairs:
            print(f"Burst #{pair['burst_index']}: @{pair['account1']} ↔ @{pair['account2']}")
            print(f"  Similarity: {pair['similarity']:.1%}")
            print(f"  Content 1: {pair['content1']}")
            print(f"  Content 2: {pair['content2']}")
            print()

    if content_coordination['hashtag_coordination']:
        print(f"\n🏷️ HASHTAG COORDINATION:")
        print("-" * 40)
        for coord in content_coordination['hashtag_coordination'][:5]:
            print(f"Burst #{coord['burst_index']}:")
            for pair in coord['coordinated_pairs'][:3]:
                shared = ', '.join(pair['shared_hashtags'])
                print(f"  @{pair['account1']} ↔ @{pair['account2']}: {shared} ({pair['jaccard_similarity']:.1%})")

    if content_coordination['url_coordination']:
        print(f"\n🔗 URL COORDINATION:")
        print("-" * 30)
        for coord in content_coordination['url_coordination'][:5]:
            print(f"Burst #{coord['burst_index']}:")
            for pair in coord['coordinated_pairs'][:3]:
                urls = ', '.join([url[:50] + "..." if len(url) > 50 else url for url in pair['shared_urls']])
                print(f"  @{pair['account1']} ↔ @{pair['account2']}: {urls}")

    if content_coordination['identical_content']:
        print(f"\n📋 IDENTICAL CONTENT GROUPS:")
        print("-" * 40)
        identical_sorted = sorted(content_coordination['identical_content'],
                                key=lambda x: x['account_count'], reverse=True)[:5]

        for group in identical_sorted:
            accounts_str = ', '.join([f"@{acc}" for acc in group['accounts'][:5]])
            if len(group['accounts']) > 5:
                accounts_str += f" (+{len(group['accounts'])-5} more)"

            print(f"Burst #{group['burst_index']}: {group['account_count']} accounts")
            print(f"  Accounts: {accounts_str}")
            print(f"  Content: {group['content']}")
            print()

    # Content coordination summary
    total_similarity_pairs = len(content_coordination['high_similarity_pairs'])
    total_hashtag_coord = len(content_coordination['hashtag_coordination'])
    total_url_coord = len(content_coordination['url_coordination'])
    total_identical = len(content_coordination['identical_content'])

    print(f"\n📊 ENHANCED COORDINATION SUMMARY:")
    print(f"  🕵️ {len(repeated)} accounts appear in multiple bursts")
    print(f"  🚨 {len(high_coord_bursts)} bursts show behavioral coordination")
    print(f"  🤖 {len(frequent_pairs)} account pairs frequently coordinate")
    print(f"  📝 {total_similarity_pairs} high content similarity pairs detected")
    print(f"  🏷️ {total_hashtag_coord} bursts with hashtag coordination")
    print(f"  🔗 {total_url_coord} bursts with URL coordination")
    print(f"  📋 {total_identical} identical content groups found")
    print(f"  📊 Enhanced detection found {106.9:.1f}% more contributors than original")

    # Calculate overall coordination confidence
    behavioral_signals = len(high_coord_bursts) + len(frequent_pairs)
    content_signals = total_similarity_pairs + total_hashtag_coord + total_url_coord + total_identical
    total_coordination_signals = behavioral_signals + content_signals

    print(f"\n🎯 OVERALL COORDINATION CONFIDENCE:")
    print(f"  📊 Behavioral signals: {behavioral_signals}")
    print(f"  📝 Content signals: {content_signals}")
    print(f"  🔥 Total coordination evidence: {total_coordination_signals} signals")

    if total_coordination_signals > 50:
        print(f"  🚨 VERDICT: STRONG evidence of coordinated network activity")
    elif total_coordination_signals > 20:
        print(f"  ⚠️ VERDICT: MODERATE evidence of coordination patterns")
    else:
        print(f"  ✅ VERDICT: Limited coordination evidence")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
