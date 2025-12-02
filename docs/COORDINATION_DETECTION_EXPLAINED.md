# Coordination Detection System: Complete Technical Explanation

## 📋 Overview

This document explains in detail how our coordination detection system finds both **Coordination Networks** (accounts working together) and **Coordination Hubs** (accounts being amplified by coordinated networks).

## 🏗️ System Architecture

```
Raw Social Media Data
        ↓
1. Burst Detection (Kleinberg Algorithm)
        ↓
2. Content Coordination Analysis
        ↓
3. Coordination Pair Generation
        ↓
4. Network Building (Graph Theory)
        ↓
5. Cross-Burst Hub Detection
        ↓
Final Results: Networks + Hubs
```

---

## 🔗 **Key Relationship: Networks Partially Depend on Hubs**

**Important Understanding:**
- **Hub detection runs FIRST** and finds RT amplification patterns
- **Network detection uses coordination pairs generated FROM hub analysis** 
- **Networks are NOT completely independent** - they incorporate hub results
- **Hubs can appear in networks** if their retweeters coordinate with each other

**Data Flow:**
```
Raw RT Data → Hub Detection → RT Coordination Pairs → Network Building (+ other pairs)
```

---

# 🕸️ PART 1: COORDINATION NETWORKS (Account-to-Account Coordination)

## How We Find Coordination Pairs

### 1. 🎯 **IDENTICAL CONTENT COORDINATION**

**Purpose:** Find accounts posting exactly the same content (copy-paste coordination)

**Step-by-step process:**
```python
# Step 1: Normalize content for comparison
for each post in burst_posts:
    content = post['content_cleaned']
    account = post['account.username']
    
    # Skip retweets - focus on original content only
    if content.startswith('RT @'):
        continue
    
    # Normalize: remove extra spaces, convert to lowercase
    normalized_content = ' '.join(content.lower().strip().split())
    
    # Group posts by identical normalized content
    content_groups[normalized_content].append({
        'account': account,
        'original_content': content,
        'post_time': timestamp
    })

# Step 2: Find groups with multiple different accounts
for normalized_content, posts in content_groups.items():
    if len(posts) >= 2:  # Multiple posts with same content
        unique_accounts = {post['account'] for post in posts}
        if len(unique_accounts) >= 2:  # Must be different accounts
            
            # Step 3: Create coordination pairs from all combinations
            for i, post1 in enumerate(posts):
                for post2 in posts[i+1:]:
                    if post1['account'] != post2['account']:
                        # CREATE COORDINATION PAIR
                        coordination_pair = {
                            'type': 'identical_content',
                            'account1': post1['account'],
                            'account2': post2['account'],
                            'content1': post1['original_content'],
                            'content2': post2['original_content'],
                            'confidence': 1.0,  # 100% confidence for identical content
                            'evidence_strength': 'VERY_HIGH'
                        }
```

**Real Example:**
- **@PatriotAccount1** posts: "BREAKING: The Wall Street Journal reports massive voter fraud evidence!"
- **@PatriotAccount2** posts: "BREAKING: The Wall Street Journal reports massive voter fraud evidence!"
- **Result:** Coordination pair (@PatriotAccount1 ↔ @PatriotAccount2, confidence: 100%)

---

### 2. 🔗 **HIGH SIMILARITY COORDINATION**

**Purpose:** Find accounts posting very similar (but not identical) content

**Step-by-step process:**
```python
# Step 1: Collect all posts by account
account_posts = defaultdict(list)
for each post in burst_posts:
    if not is_retweet and sufficient_length:
        account_posts[account].append(content.strip())

# Step 2: Compare content between all account pairs
for i, account1 in enumerate(accounts):
    for account2 in accounts[i+1:]:
        posts1 = account_posts[account1]
        posts2 = account_posts[account2]
        
        # Step 3: Compare each post combination
        for content1 in posts1[:3]:  # Limit to 3 posts per account for performance
            for content2 in posts2[:3]:
                
                # Calculate text similarity using SequenceMatcher
                similarity = SequenceMatcher(None, 
                    content1.lower(), 
                    content2.lower()).ratio()
                
                if similarity >= 0.85:  # 85% similarity threshold
                    confidence = min(similarity * 1.2, 1.0)  # Boost confidence slightly
                    
                    # CREATE COORDINATION PAIR
                    coordination_pair = {
                        'type': 'high_similarity',
                        'account1': account1,
                        'account2': account2,
                        'content1': content1[:150] + "...",
                        'content2': content2[:150] + "...",
                        'similarity_score': similarity,
                        'confidence': confidence,
                        'evidence_strength': 'HIGH' if similarity >= 0.9 else 'MEDIUM'
                    }
```

**Real Example:**
- **@Account1:** "Patriots unite! We must fight for America and our freedoms!"
- **@Account2:** "Patriots unite! We need to fight for our America and freedoms!"
- **Similarity:** 94% → **Confidence:** 94% × 1.2 = 100% (capped)
- **Result:** Coordination pair (confidence: 100%)

---

### 3. 🏷️ **HASHTAG COORDINATION**

**Purpose:** Find accounts using identical hashtag combinations (organized campaigns)

**Step-by-step process:**
```python
# Step 1: Extract hashtags by account across the burst
account_hashtags = defaultdict(list)
for each post in burst_posts:
    if not is_retweet:
        hashtags = re.findall(r'#\w+', content.lower())
        if hashtags:
            account_hashtags[account].extend(hashtags)

# Step 2: Compare hashtag usage between all account pairs
for i, account1 in enumerate(accounts):
    for account2 in accounts[i+1:]:
        hashtags1 = set(account_hashtags[account1])  # Unique hashtags for account1
        hashtags2 = set(account_hashtags[account2])  # Unique hashtags for account2
        
        # Both accounts must use at least 2 hashtags
        if len(hashtags1) >= 2 and len(hashtags2) >= 2:
            
            # Step 3: Calculate Jaccard similarity
            intersection = hashtags1 & hashtags2  # Shared hashtags
            union = hashtags1 | hashtags2        # All unique hashtags combined
            
            if len(intersection) >= 2:  # Must share at least 2 hashtags
                jaccard_similarity = len(intersection) / len(union)
                
                if jaccard_similarity >= 0.6:  # 60% hashtag overlap threshold
                    confidence = min(jaccard_similarity * 1.5, 1.0)  # Boost confidence
                    
                    # CREATE COORDINATION PAIR
                    coordination_pair = {
                        'type': 'hashtag_coordination',
                        'account1': account1,
                        'account2': account2,
                        'shared_hashtags': list(intersection),
                        'jaccard_similarity': jaccard_similarity,
                        'confidence': confidence,
                        'evidence_strength': 'HIGH' if jaccard_similarity > 0.8 else 'MEDIUM'
                    }
```

**Real Example:**
- **@Account1** uses: #PatriotsUnite #AmericaFirst #MAGA #Trump2024 #SaveAmerica
- **@Account2** uses: #PatriotsUnite #AmericaFirst #MAGA #StopTheSteal
- **Shared hashtags:** 3 (#PatriotsUnite, #AmericaFirst, #MAGA)
- **All unique hashtags:** 6 total
- **Jaccard similarity:** 3/6 = 50% (below 60% threshold - no pair created)

- **@Account3** uses: #PatriotsUnite #AmericaFirst #MAGA #Trump2024  
- **@Account4** uses: #PatriotsUnite #AmericaFirst #MAGA #SaveAmerica #DrainTheSwamp
- **Shared hashtags:** 3 (#PatriotsUnite, #AmericaFirst, #MAGA)
- **All unique hashtags:** 5 total
- **Jaccard similarity:** 3/5 = 60% → **Confidence:** 60% × 1.5 = 90%
- **Result:** Coordination pair (@Account3 ↔ @Account4, confidence: 90%)

---

### 4. 🔗 **URL COORDINATION**

**Purpose:** Find accounts sharing the same URLs (coordinated link spreading)

**Step-by-step process:**
```python
# Step 1: Extract URLs by account
account_urls = defaultdict(set)
for each post in burst_posts:
    if not is_retweet:
        urls = re.findall(url_pattern, content)  # Extract all URLs
        if urls:
            account_urls[account].update(urls)

# Step 2: Find accounts sharing URLs
for i, account1 in enumerate(accounts):
    for account2 in accounts[i+1:]:
        urls1 = account_urls[account1]
        urls2 = account_urls[account2]
        
        shared_urls = urls1 & urls2  # Intersection of URL sets
        
        if len(shared_urls) >= 1:  # Any shared URL is suspicious
            confidence = min(len(shared_urls) * 0.8, 1.0)  # 80% per shared URL
            
            # CREATE COORDINATION PAIR
            coordination_pair = {
                'type': 'url_coordination',
                'account1': account1,
                'account2': account2,
                'shared_urls': list(shared_urls),
                'confidence': confidence,
                'evidence_strength': 'HIGH' if len(shared_urls) > 1 else 'MEDIUM'
            }
```

**Real Example:**
- **@NewsAccount1** shares: https://breitbart.com/election-fraud-story
- **@NewsAccount2** shares: https://breitbart.com/election-fraud-story, https://gateway-pundit.com/biden-scandal
- **Shared URLs:** 1 (breitbart.com story)
- **Confidence:** 1 × 80% = 80%
- **Result:** Coordination pair (@NewsAccount1 ↔ @NewsAccount2, confidence: 80%)

---

### 5. ⏱️ **ULTRA-CONSERVATIVE TEMPORAL SYNCHRONIZATION**

**Purpose:** Find accounts posting at suspiciously identical times (coordinated timing)

**Step-by-step process:**
```python
# Step 1: Group posts by account with precise timestamps
account_times = defaultdict(list)
for each post in burst_posts:
    if not is_retweet and has_valid_timestamp:
        account_times[account].append({
            'time': pd.to_datetime(post_time),
            'content': content[:100]  # Store snippet for evidence
        })

# Step 2: Find temporally synchronized posting pairs
for i, account1 in enumerate(accounts):
    for account2 in accounts[i+1:]:
        times1 = account_times[account1]
        times2 = account_times[account2]
        synchronized_pairs = []
        
        # Step 3: Check timing between all post combinations
        for post1 in times1:
            for post2 in times2:
                time_diff = abs(post1['time'] - post2['time'])
                
                # Ultra-conservative: 30 second window only
                if time_diff <= pd.Timedelta(seconds=30):
                    synchronized_pairs.append({
                        'time_diff_seconds': time_diff.total_seconds(),
                        'content1': post1['content'],
                        'content2': post2['content'],
                        'time1': post1['time'],
                        'time2': post2['time']
                    })
        
        # Step 4: Ultra-conservative thresholds
        if len(synchronized_pairs) >= 3:  # Require 3+ synchronized posts
            avg_sync_time = np.mean([pair['time_diff_seconds'] for pair in synchronized_pairs])
            
            # Calculate confidence components
            timing_precision = max(0, 1 - (avg_sync_time / 30))  # Precision within 30sec window
            sync_strength = min(len(synchronized_pairs) / 3.0, 1.0)  # Strength from count
            confidence = (timing_precision + sync_strength) / 2
            
            if confidence >= 0.8:  # Require 80%+ confidence
                # CREATE COORDINATION PAIR
                coordination_pair = {
                    'type': 'ultra_conservative_temporal_sync',
                    'account1': account1,
                    'account2': account2,
                    'synchronized_posts': len(synchronized_pairs),
                    'avg_sync_time_seconds': avg_sync_time,
                    'confidence': confidence,
                    'evidence_strength': 'VERY_HIGH' if avg_sync_time <= 10 else 'HIGH',
                    'sync_details': synchronized_pairs[:3]  # First 3 for evidence
                }
```

**Real Example:**
- **@Account1** posts at 14:32:15, 14:45:02, 15:12:44
- **@Account2** posts at 14:32:28, 14:45:19, 15:12:51  
- **Time differences:** 13 seconds, 17 seconds, 7 seconds
- **Average sync time:** 12.3 seconds
- **Timing precision:** (1 - 12.3/30) = 59%
- **Sync strength:** 3/3 = 100%  
- **Confidence:** (59% + 100%) / 2 = 79.5% (below 80% threshold - no pair created)

- **@Account3** posts at 16:15:05, 16:28:33, 16:45:12
- **@Account4** posts at 16:15:09, 16:28:37, 16:45:15
- **Time differences:** 4 seconds, 4 seconds, 3 seconds  
- **Average sync time:** 3.7 seconds
- **Timing precision:** (1 - 3.7/30) = 88%
- **Sync strength:** 3/3 = 100%
- **Confidence:** (88% + 100%) / 2 = 94%
- **Result:** Coordination pair (@Account3 ↔ @Account4, confidence: 94%)

---

## 🔗 How Coordination Pairs Become Networks

After generating individual coordination pairs, we build networks using graph theory:

```python
# Step 1: Build coordination graph
coordination_graph = defaultdict(set)
for pair in all_coordination_pairs:
    account1 = pair['account1']
    account2 = pair['account2']
    # Create bidirectional connections
    coordination_graph[account1].add(account2)
    coordination_graph[account2].add(account1)

# Step 2: Find connected components using Depth-First Search
def find_connected_component(start_account, visited, component):
    if start_account in visited:
        return
    visited.add(start_account)
    component.add(start_account)
    
    # Recursively visit all connected accounts
    for neighbor in coordination_graph[start_account]:
        find_connected_component(neighbor, visited, component)

# Step 3: Build networks from connected components
visited = set()
networks = []
for account in coordination_graph:
    if account not in visited:
        component = set()
        find_connected_component(account, visited, component)
        
        if len(component) >= 2:  # Network must have at least 2 accounts
            # Calculate network metrics
            network_pairs = [p for p in coordination_pairs 
                           if p['account1'] in component and p['account2'] in component]
            avg_confidence = np.mean([p['confidence'] for p in network_pairs])
            evidence_types = list(set([p['type'] for p in network_pairs]))
            
            networks.append({
                'accounts': list(component),
                'size': len(component),
                'avg_confidence': avg_confidence,
                'evidence_types': evidence_types,
                'risk_level': 'HIGH' if avg_confidence > 0.8 else 'MEDIUM' if avg_confidence > 0.6 else 'LOW'
            })
```

**Network Building Example:**
- **Pair 1:** @A ↔ @B (hashtag coordination, confidence: 85%)
- **Pair 2:** @B ↔ @C (temporal synchronization, confidence: 92%)  
- **Pair 3:** @C ↔ @D (URL coordination, confidence: 80%)
- **Pair 4:** @E ↔ @F (identical content, confidence: 100%)

**Resulting Networks:**
- **Network 1:** [@A, @B, @C, @D] - 4 accounts connected through chain
  - Average confidence: (85% + 92% + 80%) / 3 = 85.7%
  - Evidence types: [hashtag_coordination, ultra_conservative_temporal_sync, url_coordination]
  - Risk level: HIGH
- **Network 2:** [@E, @F] - 2 accounts with identical content
  - Average confidence: 100%  
  - Evidence types: [identical_content]
  - Risk level: HIGH

---

# 🎯 PART 2: COORDINATION HUBS (Amplification Centers)

## What Are Coordination Hubs?

**Coordination Hubs** are individual accounts that serve as **amplification centers** - they create original content that gets systematically retweeted/amplified by coordinated networks of other accounts.

**Key Difference from Networks:**
- **Hubs:** Accounts coordinate **AROUND a central source** (hub-and-spoke)
- **Networks:** Accounts coordinate **WITH each other** (peer-to-peer) 

## Hub Detection: Step-by-Step Process

### Phase 1: Retweet Amplification Analysis (Within Bursts)

```python
def _find_retweet_coordination(burst_idx, burst_posts):
    """
    Find coordinated retweet amplification patterns within a single burst.
    This identifies which accounts are being amplified by coordinated retweeting.
    """
    # Step 1: Extract retweet patterns
    rt_sources = defaultdict(set)  # source account -> set of retweeters
    rt_content = {}  # source account -> original content being RTed
    
    for each post in burst_posts:
        content = post['content_cleaned']
        retweeter = post['account.username']
        
        # Check if this is a retweet
        if content.startswith('RT @'):
            # Extract the source account being retweeted
            rt_source = extract_username_after_RT_symbol(content)  # e.g., "maxjett12"
            if rt_source:
                rt_sources[rt_source].add(retweeter)  # Track who RTed this source
                if rt_source not in rt_content:
                    rt_content[rt_source] = content  # Store what was RTed
    
    # Step 2: Find coordinated amplification patterns
    amplification_evidence = []
    for source_account, retweeters in rt_sources.items():
        if len(retweeters) >= 3:  # At least 3 accounts RTing the same source
            retweeter_list = list(retweeters)
            coordination_strength = min(len(retweeters) / 10.0, 1.0)  # Scale 0-1
            
            # This source account is being coordinated amplified!
            amplification_evidence.append({
                'type': 'retweet_amplification',
                'burst_index': burst_idx,
                'rt_source': source_account,  # The account being amplified
                'retweeters': retweeter_list,  # Accounts doing the amplifying
                'amplification_count': len(retweeters),
                'confidence': coordination_strength,
                'evidence_strength': 'VERY_HIGH' if len(retweeters) >= 10 else 'HIGH' if len(retweeters) >= 5 else 'MEDIUM',
                'original_content': rt_content[source_account][:150] + "..."
            })
    
    return amplification_evidence
```

**Real Example (Within One Burst):**
- **Burst #15** (2025-01-28 14:14:10 to 14:29:39):
  - @maxjett12 posts: "#PatriotsUnite #AmericaFirst #MAGA #Trump2024 #DrainTheSwamp..."
  - @retweeter1: "RT @maxjett12 #PatriotsUnite #AmericaFirst #MAGA..."
  - @retweeter2: "RT @maxjett12 #PatriotsUnite #AmericaFirst #MAGA..."  
  - @retweeter3: "RT @maxjett12 #PatriotsUnite #AmericaFirst #MAGA..."
  - @retweeter4: "RT @maxjett12 #PatriotsUnite #AmericaFirst #MAGA..."
  - @retweeter5: "RT @maxjett12 #PatriotsUnite #AmericaFirst #MAGA..."

**Result:** Amplification evidence created:
```json
{
  "type": "retweet_amplification",
  "burst_index": 15,
  "rt_source": "maxjett12",
  "retweeters": ["retweeter1", "retweeter2", "retweeter3", "retweeter4", "retweeter5"],
  "amplification_count": 5,
  "confidence": 0.5,  // 5 retweeters / 10 = 50%
  "evidence_strength": "MEDIUM"
}
```

### 🔗 How RT Amplification Creates Coordination Pairs for Networks

**Critical Connection:** The RT amplification evidence gets converted into coordination pairs:

```python
# In _process_coordination_evidence() method
elif evidence['type'] == 'retweet_amplification':
    retweet_amplification.append(evidence)
    
    # Convert RT amplification to coordination pairs for network building
    retweeters = evidence['retweeters']  # ["retweeter1", "retweeter2", "retweeter3", ...]
    
    for i, retweeter1 in enumerate(retweeters):
        for retweeter2 in retweeters[i+1:]:
            # CREATE COORDINATION PAIRS between all retweeter combinations
            coordination_pairs.append({
                'type': 'retweet_coordination',
                'account1': retweeter1,         # e.g., "retweeter1"
                'account2': retweeter2,         # e.g., "retweeter2"  
                'confidence': evidence['confidence'],  # 0.5
                'rt_source': evidence['rt_source'],    # "maxjett12"
                'burst_index': evidence['burst_index'] # 15
            })
```

**Result from @maxjett12 example:**
- 5 retweeters create **10 coordination pairs** between all combinations:
  - retweeter1 ↔ retweeter2 (confidence: 50%, rt_source: maxjett12)
  - retweeter1 ↔ retweeter3 (confidence: 50%, rt_source: maxjett12)
  - retweeter1 ↔ retweeter4 (confidence: 50%, rt_source: maxjett12)
  - retweeter1 ↔ retweeter5 (confidence: 50%, rt_source: maxjett12)
  - retweeter2 ↔ retweeter3 (confidence: 50%, rt_source: maxjett12)
  - ...and so on

**These coordination pairs then get used in network building!**

### Phase 2: Cross-Burst Hub Analysis (Persistence Detection)

```python
def add_cross_burst_analysis(all_evidence, burst_contributors):
    """
    Analyze RT amplification across ALL bursts to find persistent coordination hubs.
    This catches accounts like @maxjett12 that maintain amplification networks over time.
    """
    
    # Step 1: Track RT source activity across all bursts
    rt_source_activity = defaultdict(lambda: {
        'total_amplifications': 0,        # Total number of times RTed
        'burst_appearances': set(),       # Which bursts they appeared in
        'total_retweeters': set(),       # All unique accounts that RTed them
        'burst_details': []              # Detailed breakdown per burst
    })
    
    # Step 2: Aggregate amplification evidence from all bursts
    rt_amplifications = [e for e in all_evidence if e['type'] == 'retweet_amplification']
    
    for evidence in rt_amplifications:
        source = evidence['rt_source']  # e.g., "maxjett12"
        burst_idx = evidence['burst_index']
        retweeters = evidence['retweeters']
        amplification_count = evidence['amplification_count']
        
        # Accumulate cross-burst statistics
        rt_source_activity[source]['total_amplifications'] += amplification_count
        rt_source_activity[source]['burst_appearances'].add(burst_idx)
        rt_source_activity[source]['total_retweeters'].update(retweeters)
        rt_source_activity[source]['burst_details'].append({
            'burst_index': burst_idx,
            'amplification_count': amplification_count,
            'retweeters': retweeters
        })
    
    # Step 3: Flag persistent coordination hubs
    coordination_hubs = []
    for source_account, activity in rt_source_activity.items():
        
        # Hub qualification criteria (OR condition - either can qualify)
        burst_appearances = len(activity['burst_appearances'])
        total_retweeters = len(activity['total_retweeters'])
        
        if burst_appearances >= 3 OR total_retweeters >= 10:
            # This account qualifies as a coordination hub!
            
            # Step 4: Calculate hub confidence scores
            persistence_score = min(burst_appearances / 10.0, 1.0)  # Active across bursts
            amplification_score = min(total_retweeters / 50.0, 1.0)  # Size of RT network
            combined_score = (persistence_score + amplification_score) / 2
            
            # Step 5: Create coordination hub evidence
            coordination_hubs.append({
                'type': 'coordination_hub',
                'hub_account': source_account,
                'burst_appearances': burst_appearances,
                'total_retweeters': total_retweeters,
                'total_amplifications': activity['total_amplifications'],
                'persistence_score': persistence_score,
                'amplification_score': amplification_score,
                'confidence': combined_score,
                'evidence_strength': 'VERY_HIGH' if combined_score > 0.8 else 'HIGH' if combined_score > 0.6 else 'MEDIUM',
                'retweeter_network': list(activity['total_retweeters'])[:20],  # First 20 for display
                'burst_details': activity['burst_details']
            })
    
    return coordination_hubs
```

**Real Example (Cross-Burst Analysis):**

**@maxjett12 Amplification Pattern:**
- **Burst #3:** 3 retweeters (@user1, @user2, @user3)
- **Burst #7:** 4 retweeters (@user1, @user4, @user5, @user6)  
- **Burst #15:** 5 retweeters (@user2, @user7, @user8, @user9, @user10)
- **Burst #23:** 2 retweeters (@user1, @user11)
- **Burst #31:** 3 retweeters (@user12, @user13, @user14)
- **Burst #45:** 4 retweeters (@user2, @user15, @user16, @user17)
- **Burst #52:** 2 retweeters (@user18, @user19)  
- **Burst #67:** 3 retweeters (@user1, @user20, @user21)

**Cross-Burst Statistics:**
- **Burst appearances:** 8 bursts (out of 76 total)
- **Total unique retweeters:** 21 different accounts
- **Total amplifications:** 26 retweets across all bursts

**Hub Confidence Calculation:**
- **Persistence score:** min(8 / 10, 1.0) = 80% (active across 8 bursts)
- **Amplification score:** min(21 / 50, 1.0) = 42% (21-account RT network)  
- **Combined confidence:** (80% + 42%) / 2 = 61%

**Final Hub Detection Result:**
```json
{
  "type": "coordination_hub",
  "hub_account": "maxjett12",
  "burst_appearances": 8,
  "total_retweeters": 21, 
  "total_amplifications": 26,
  "confidence": 0.61,
  "evidence_strength": "MEDIUM",
  "retweeter_network": ["user1", "user2", "user3", "user4", "user5", ...]
}
```

**Result:** @maxjett12 identified as **Coordination Hub #1** with 21-account amplification network active across 8 bursts.

## Hub Qualification Criteria

### Minimum Thresholds (OR Condition):
1. **Cross-burst persistence:** Active in 3+ bursts (shows sustained coordination)
2. **Large amplification network:** 10+ unique retweeters (shows network size)

### Confidence Scoring:
- **Persistence Score:** Number of burst appearances / 10 (capped at 100%)
- **Amplification Score:** Number of unique retweeters / 50 (capped at 100%)  
- **Combined Confidence:** Average of both scores

### Evidence Strength Levels:
- **VERY_HIGH:** Combined confidence > 80%
- **HIGH:** Combined confidence > 60%  
- **MEDIUM:** Combined confidence > 40%
- **LOW:** Combined confidence ≤ 40%



---

## 🎯 **Confidence Boosting: Why and How**

### **What is Confidence Boosting?**

Confidence boosting is when we multiply a raw similarity score by a factor greater than 1.0 to increase the final confidence score. This is done strategically to reflect the **investigative significance** of different coordination patterns.

### **Confidence Boosting Factors Used:**

| Detection Method | Raw Score | Boost Factor | Reasoning |
|-----------------|-----------|-------------|-----------|
| **High Similarity** | Text similarity (0.85-1.0) | **×1.2** | Boosts 85%→100% because high text similarity is strong evidence |
| **Hashtag Coordination** | Jaccard similarity (0.6-1.0) | **×1.5** | Boosts 60%→90% because coordinated hashtag campaigns are highly indicative |
| **URL Coordination** | Count-based (1-2+ URLs) | **×0.8 per URL** | Each shared URL adds 80% confidence (can exceed 100%) |
| **Identical Content** | N/A | **1.0 (100%)** | No boosting needed - identical content is definitive evidence |
| **Temporal Sync** | Complex calculation | **No boosting** | Confidence calculated from timing precision + sync strength |

---

### **Why Do We Need Confidence Boosting?**

#### **1. Raw Similarity Scores Don't Reflect Investigative Significance**

**Problem:** A 60% hashtag overlap might seem "moderate," but in coordination detection, this is actually **very strong evidence** of organized campaigns.

**Example:**
- **Raw Jaccard similarity:** 60% (3 shared hashtags out of 5 total)
- **Investigative reality:** Using identical hashtag combinations is highly suspicious
- **Boosted confidence:** 60% × 1.5 = 90% (reflects true significance)

#### **2. Different Evidence Types Have Different Strength**

**Hashtag Coordination** gets the highest boost (×1.5) because:
- Hashtags are **deliberately chosen** for campaigns
- Identical combinations indicate **organized coordination**  
- Much stronger evidence than coincidental content similarity

**Content Similarity** gets moderate boost (×1.2) because:
- Could be coincidental or inspired by same news
- Still strong evidence but less definitive than hashtags

#### **3. Conservative Thresholds Need Confidence Adjustment**

We use conservative similarity thresholds to avoid false positives:
- **85% text similarity** threshold (high bar)
- **60% hashtag overlap** threshold (high bar)

But once these thresholds are met, the evidence is actually **stronger than the raw percentage suggests**, so we boost confidence accordingly.

---

### **Where Did These Numbers Come From?**

#### **×1.2 Factor (High Similarity):**
```python
confidence = min(similarity * 1.2, 1.0)  # 85% becomes 100%
```

**Rationale:**
- **85% text similarity** after our conservative threshold = very strong evidence
- Boost to 100% reflects that passing our strict threshold indicates definitive coordination
- **1.2 factor chosen** to push 85% threshold cases to maximum confidence
- Anything above 83% becomes 100% confidence (83% × 1.2 = 99.6% ≈ 100%)

#### **×1.5 Factor (Hashtag Coordination):**
```python
confidence = min(jaccard_similarity * 1.5, 1.0)  # 60% becomes 90%
```

**Rationale:** 
- **Hashtag coordination is highly indicative** of organized campaigns
- **60% Jaccard similarity** with our threshold requirements = strong evidence
- **1.5 factor chosen** because:
  - 60% threshold → 90% confidence (reflects high significance)
  - 67% similarity → 100% confidence (maximum for strong evidence)
  - Higher boost than content similarity because hashtags are more deliberately coordinated

#### **×0.8 Factor (URL Coordination):**
```python
confidence = min(len(shared_urls) * 0.8, 1.0)  # 1 URL = 80%, 2 URLs = 100%
```

**Rationale:**
- **Each shared URL** represents coordination evidence
- **80% per URL** chosen because:
  - 1 shared URL = 80% (strong but not definitive - could be coincidence)
  - 2 shared URLs = 160% → 100% (very strong evidence of coordination)
  - **Additive model** reflects that multiple shared URLs exponentially increase suspicion

---

### **Mathematical Justification:**

#### **Threshold-Confidence Mapping:**

The boost factors create these **threshold-to-confidence mappings**:

**High Similarity (×1.2):**
- 85% similarity → 100% confidence ✓
- 90% similarity → 100% confidence ✓  
- 95% similarity → 100% confidence ✓

**Hashtag Coordination (×1.5):**
- 60% overlap → 90% confidence ✓
- 70% overlap → 100% confidence ✓
- 80% overlap → 100% confidence ✓

**URL Coordination (×0.8 additive):**
- 1 URL → 80% confidence ✓
- 2 URLs → 100% confidence ✓
- 3 URLs → 100% confidence ✓

#### **Why These Mappings Make Sense:**

1. **Conservative thresholds** ensure we only detect strong patterns
2. **Boost factors** reflect that **meeting our thresholds** = high investigative significance  
3. **Capped at 100%** prevents overconfidence
4. **Different evidence types** get different boosts based on their coordination significance

---

### **Alternative Approaches Considered:**

#### **❌ No Boosting (Raw Scores):**
- 85% similarity → 85% confidence
- 60% hashtag overlap → 60% confidence
- **Problem:** Undervalues strong evidence that meets conservative thresholds

#### **❌ Fixed High Confidence:**
- All evidence → 90% confidence  
- **Problem:** Doesn't differentiate between evidence strength levels

#### **✅ Scaled Boosting (Our Approach):**
- Different boost factors for different evidence types
- **Advantage:** Reflects both threshold achievement AND evidence type significance
- **Advantage:** Maintains relative confidence differences while boosting strong cases

---

### **Real-World Impact:**

**Without Boosting:**
- @Account1 ↔ @Account2: 60% hashtag overlap → 60% confidence → "MEDIUM" evidence
- Might be missed in high-confidence filtering

**With Boosting:**  
- @Account1 ↔ @Account2: 60% hashtag overlap → 90% confidence → "HIGH" evidence
- Correctly flagged as strong coordination evidence

**Result:** Boosting ensures that evidence meeting our conservative thresholds gets the investigative attention it deserves, while maintaining mathematical rigor in confidence calculation.

---

