# ⏰ Temporal Clustering Analysis - Complete Guide

A comprehensive explanation of temporal clustering methodology used to identify behavioral patterns and potential coordination through posting time analysis.

---

## 📚 Table of Contents

1. [Overview](#overview)
2. [The Goal: Finding Coordinated Accounts](#the-goal-finding-coordinated-accounts)
3. [Why Temporal Clustering?](#why-temporal-clustering)
4. [The Timezone Challenge](#the-timezone-challenge)
5. [Methodology: Two Approaches](#methodology-two-approaches)
6. [2D Temporal Clustering](#2d-temporal-clustering)
7. [24D Temporal Clustering](#24d-temporal-clustering)
8. [Implementation Details](#implementation-details)
9. [Results and Interpretation](#results-and-interpretation)
10. [Limitations and Challenges](#limitations-and-challenges)
11. [Comparison with Content-Based Detection](#comparison-with-content-based-detection)
12. [Use Cases and Applications](#use-cases-and-applications)

---

## Overview

### **What is Temporal Clustering?**

Temporal clustering is a **behavioral fingerprinting technique** that groups social media accounts based on **when they post**, rather than **what they post**.

### **Core Premise**

**Hypothesis:** Coordinated accounts often show **similar posting time patterns** because:
- They're operated by the same person/organization
- They follow the same work schedule
- They're automated scripts running on the same schedule
- They're in the same timezone and follow similar routines

### **The Approach**

```
For each account:
1. Extract posting times (hour of day, day of week)
2. Create temporal "fingerprint" (activity pattern)
3. Cluster accounts with similar fingerprints
4. Analyze clusters for coordination evidence
```

### **Expected Outcome**

Accounts in the same cluster:
- ✅ Post during similar hours
- ✅ Have similar circadian rhythms
- ✅ May be coordinated (same operator/bot network)
- ❌ Could also be legitimate users in same timezone

---

## The Goal: Finding Coordinated Accounts

### **Primary Objective**

**Identify groups of accounts that coordinate together** through temporal behavior analysis.

### **Why This Matters**

**Coordinated Inauthentic Behavior (CIB) often exhibits temporal patterns:**

**Example 1: Bot Network**
```
Bot A: Posts at 9:00, 12:00, 15:00, 18:00 every day
Bot B: Posts at 9:00, 12:00, 15:00, 18:00 every day
Bot C: Posts at 9:00, 12:00, 15:00, 18:00 every day

→ Identical temporal patterns = High coordination probability
```

**Example 2: Troll Farm**
```
All accounts post primarily between 9:00-17:00 Moscow time
→ Same work schedule = Centralized operation
```

**Example 3: Automated Campaign**
```
Spike at exactly 10:00 AM every Monday
50 accounts post within 5-minute window
→ Scheduled automation
```

### **What We're Trying to Detect**

1. **Same Operator:** One person running multiple accounts
2. **Bot Networks:** Automated accounts on same schedule
3. **Coordinated Teams:** Groups working same shifts
4. **Campaign Timing:** Synchronized posting windows

---

## Why Temporal Clustering?

### **Advantages of Temporal Analysis**

#### **1. Independent of Content**

**Content-based methods miss:**
- Accounts posting different content but coordinating timing
- Multilingual coordination
- Subtle narrative alignment

**Temporal clustering catches:**
```
Account A: Posts about Topic X at 10:00, 14:00, 18:00
Account B: Posts about Topic Y at 10:00, 14:00, 18:00
Account C: Posts about Topic Z at 10:00, 14:00, 18:00

Content is different, but timing is identical
→ Suggests shared automation/operator
```

#### **2. Hard to Evade**

**Temporal patterns are difficult to mask:**
- Human operators have natural circadian rhythms
- Bots follow programmed schedules
- Work shifts create predictable patterns

**Changing posting times requires:**
- 24/7 operation (expensive)
- Random scheduling (reduces effectiveness)
- Multiple timezone coverage (complex)

#### **3. Reveals Operational Patterns**

**Temporal clustering exposes:**
```
Cluster 1 (9-5 pattern):
├─ Suggests: Paid operators
├─ Location: Likely single timezone
└─ Type: Professional operation

Cluster 2 (24/7 uniform):
├─ Suggests: Automated bots
├─ Location: No geographic bias
└─ Type: Script-based network

Cluster 3 (Evening spike):
├─ Suggests: Part-time trolls
├─ Location: After-work hours
└─ Type: Volunteer/hobbyist
```

#### **4. Complements Content Analysis**

**Multi-signal approach:**
```
Temporal Clustering finds: Accounts A, B, C have same posting schedule
        +
Content Analysis finds: Accounts A, B, C share hashtags
        ↓
Combined Evidence = HIGH confidence coordination
```

---

## The Timezone Challenge

### **The Critical Limitation**

**Problem:** Our dataset has all timestamps auto-converted to UTC.

### **What This Means**

**Original data might have been:**
```
Account A in New York (UTC-5):  Posts at 9:00 AM local = 14:00 UTC
Account B in London (UTC+0):    Posts at 9:00 AM local = 09:00 UTC
Account C in Tokyo (UTC+9):     Posts at 9:00 AM local = 00:00 UTC
```

**After conversion to UTC:**
```
Account A: 14:00 UTC
Account B: 09:00 UTC
Account C: 00:00 UTC

All three accounts post at "9 AM their time"
But in UTC, they look completely different!
```

### **Impact on Coordination Detection**

#### **False Negatives (Missing Real Coordination)**

**Scenario: Coordinated accounts in different timezones**
```
Coordinated bot network with operators in:
- New York (posts at 10 AM local)
- Moscow (posts at 10 AM local)
- Beijing (posts at 10 AM local)

In UTC: 15:00, 07:00, 02:00
→ Appear unrelated in temporal clustering
→ Miss coordination that exists
```

#### **False Positives (Finding Fake Coordination)**

**Scenario: Unrelated accounts in same timezone**
```
Random users in California:
- User A: Morning poster (9 AM)
- User B: Morning poster (9 AM)
- User C: Morning poster (9 AM)

In UTC: All post at 17:00
→ Cluster together
→ Look coordinated but aren't
```

### **Why We Can't Solve This**

**Missing information:**
```
Dataset has: Timestamp in UTC
Dataset lacks: Account timezone, location metadata

To properly detect coordination, we need:
❌ Account location
❌ Original posting timezone
❌ IP address data
❌ Timezone preference settings

We only have:
✅ UTC timestamps
```

### **Implications for Our Analysis**

**What we CAN detect:**
- ✅ Accounts in the **same timezone** with similar patterns
- ✅ Bot networks operating on **UTC schedule**
- ✅ Campaigns with **absolute time** coordination

**What we CANNOT detect:**
- ❌ Cross-timezone coordination (same local time)
- ❌ Distributed networks (synchronized local time)
- ❌ Global campaigns (multi-timezone operators)

**Our approach becomes:**
```
Not: "Find coordinated accounts"
But: "Find accounts with similar UTC posting patterns"

Then investigate:
- Are they in same timezone? (might be coordinated)
- Are they in different timezones? (less likely coordinated by timing alone)
```

---

## Methodology: Two Approaches

### **Approach 1: 2D Temporal Clustering**

**Features (2 dimensions):**
1. **Hour of Day** (0-23): When do they typically post?
2. **Weekend Activity Ratio** (0-1): Weekday vs weekend posting

**Purpose:** Broad behavioral categorization

**Example clusters:**
```
Cluster 0: Morning posters (6-9 AM peak)
Cluster 1: Evening posters (6-10 PM peak)
Cluster 2: Night owls (10 PM - 2 AM peak)
```

### **Approach 2: 24D Temporal Clustering**

**Features (24 dimensions):**
- One dimension per hour (0:00, 1:00, 2:00, ..., 23:00)
- Value = proportion of posts in that hour

**Purpose:** Fine-grained activity fingerprinting

**Example fingerprint:**
```
Account A: [0.02, 0.01, 0.00, 0.00, 0.05, 0.10, 0.15, 0.20, ...]
           │─────────────────│  │──────────────────────│
           Sleeping          Morning rush

Account B: [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.05, ...]
           │─────────────────────────────────────│  │────────
           No activity until 8 AM                 Morning start
```

### **Why Both?**

| Aspect | 2D Clustering | 24D Clustering |
|--------|---------------|----------------|
| **Granularity** | Coarse (general patterns) | Fine (hour-by-hour) |
| **Interpretability** | Easy (morning/evening) | Complex (heatmaps needed) |
| **Noise** | Less affected | More sensitive |
| **Coordination Detection** | Broad groups | Precise matching |
| **Computational Cost** | Low | Higher |

**Strategy:** Use both for comprehensive analysis
- 2D: Quick overview of behavioral personas
- 24D: Detailed fingerprinting for coordination

---

## 2D Temporal Clustering

### **Feature Engineering**

#### **Feature 1: Mean Hour of Day**

**Calculation:**
```python
# For each account
posts_per_hour = count_posts_by_hour(account)
mean_hour = Σ(hour × posts[hour]) / total_posts

Example:
Posts at: [9:00, 10:00, 14:00, 15:00, 16:00]
Mean: (9 + 10 + 14 + 15 + 16) / 5 = 12.8
→ Midday poster
```

**Interpretation:**
- Mean = 6-10: Morning person
- Mean = 10-14: Midday poster
- Mean = 14-18: Afternoon active
- Mean = 18-22: Evening poster
- Mean = 22-6: Night owl

#### **Feature 2: Weekend Activity Ratio**

**Calculation:**
```python
weekend_posts = posts on Saturday + Sunday
weekday_posts = posts on Monday-Friday
weekend_ratio = weekend_posts / total_posts

Example:
Total posts: 100
Weekend: 40
Weekday: 60
Ratio: 40/100 = 0.40
→ Active on weekends
```

**Interpretation:**
- Ratio < 0.2: Weekday focused (work hours?)
- Ratio = 0.3-0.4: Balanced
- Ratio > 0.5: Weekend heavy (personal use?)

### **Clustering Process**

**1. Feature Extraction:**
```python
for each account:
    features[account] = [mean_hour, weekend_ratio]
```

**2. Normalization:**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)
```

**3. K-means Clustering:**
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)  # or use elbow method
clusters = kmeans.fit_predict(features_normalized)
```

**4. Cluster Assignment:**
```python
cluster_labels = {
    account: cluster_id
    for account, cluster_id in zip(accounts, clusters)
}
```

### **Example Results**

**Optimal k = 3 (from elbow method)**

**Cluster 0 (n=400): "9-to-5 Workers"**
- Mean hour: 12.5
- Weekend ratio: 0.15
- **Pattern:** Business hours, weekday focused
- **Interpretation:** Professional accounts or bots mimicking workers

**Cluster 1 (n=350): "Evening Users"**
- Mean hour: 19.2
- Weekend ratio: 0.42
- **Pattern:** Evening spike, weekend active
- **Interpretation:** Personal accounts, after-work users

**Cluster 2 (n=250): "Night Owls"**
- Mean hour: 1.8 (early morning)
- Weekend ratio: 0.38
- **Pattern:** Late night/early morning posts
- **Interpretation:** Different timezone OR shift workers OR bots

### **Visualization**

**2D Scatter Plot:**
```
Weekend Ratio
│
1.0│                    ○ ○
   │                  ○   ○
0.5│        ● ●     ○       ← Cluster 1 (Evening)
   │      ● ● ●
   │    ▲ ▲ ▲             
0.0│  ▲ ▲ ▲               ← Cluster 0 (9-to-5)
   │                  ■ ■  ← Cluster 2 (Night)
   └─────────────────────────→ Mean Hour
    0  6  12  18  24
```

### **Coordination Detection**

**Within each cluster, look for:**

```python
for cluster in clusters:
    accounts_in_cluster = get_accounts(cluster)
    
    # Check for additional coordination signals:
    content_similarity = analyze_content(accounts_in_cluster)
    hashtag_overlap = check_hashtags(accounts_in_cluster)
    burst_participation = check_bursts(accounts_in_cluster)
    
    if multiple_signals_present:
        flag_as_potential_coordination(accounts_in_cluster)
```

---

## 24D Temporal Clustering

### **The 24-Hour Fingerprint**

**Concept:** Each account gets a 24-dimensional vector representing hourly activity distribution.

**Construction:**
```python
fingerprint = [
    posts_in_hour_0 / total_posts,   # Midnight-1AM
    posts_in_hour_1 / total_posts,   # 1-2 AM
    posts_in_hour_2 / total_posts,   # 2-3 AM
    ...
    posts_in_hour_23 / total_posts   # 11PM-Midnight
]

Example:
Account with 100 total posts:
- 5 posts at 9 AM → fingerprint[9] = 0.05
- 10 posts at 10 AM → fingerprint[10] = 0.10
- 15 posts at 2 PM → fingerprint[14] = 0.15
...
```

### **Normalized Fingerprint**

**Why normalize:**
- Accounts have different posting volumes
- Focus on **pattern** not **volume**
- Makes small and large accounts comparable

**Normalization:**
```python
fingerprint_normalized = fingerprint / sum(fingerprint)
# Now sum = 1.0 (probability distribution)
```

### **Feature Engineering Details**

**Per account extraction:**
```python
def create_24d_fingerprint(account_posts):
    """Create 24-dimensional hourly activity fingerprint"""
    
    # Initialize 24 bins (one per hour)
    hourly_counts = np.zeros(24)
    
    # Count posts per hour
    for post in account_posts:
        hour = post.timestamp.hour  # 0-23
        hourly_counts[hour] += 1
    
    # Normalize to probability distribution
    total_posts = sum(hourly_counts)
    fingerprint = hourly_counts / total_posts if total_posts > 0 else hourly_counts
    
    return fingerprint
```

### **Clustering Process**

**1. Feature Matrix Construction:**
```python
# Matrix: (num_accounts × 24)
feature_matrix = np.array([
    create_24d_fingerprint(account)
    for account in accounts
])
```

**2. Standardization:**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(feature_matrix)
```

**3. Optimal k Selection (Elbow Method):**
```python
inertias = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features_scaled)
    inertias.append(kmeans.inertia_)

# Find elbow point
optimal_k = find_elbow(inertias)  # e.g., k=4
```

**4. Final Clustering:**
```python
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(features_scaled)
```

### **Example Results**

**Optimal k = 4**

**Cluster 0 (n=412): "Business Hours Pattern"**
```
Hourly Distribution:
[0.00, 0.00, 0.00, 0.00, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.12, 0.10, 0.08, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00]
 │────────────────────│  │───────────────────────────────────────────────│  │──────────────────────│
 Sleeping (0-6 AM)      Active 8AM-5PM                                      Evening decline
```
**Interpretation:** Professional accounts, work schedule, possibly troll farm

**Cluster 1 (n=289): "Evening Heavy"**
```
Hourly Distribution:
[0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.02, 0.03, 0.05, 0.05, 0.04, 0.03, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.10, 0.05, 0.03, 0.02, 0.01, 0.00]
 │────────────────────────────────│  │────────────────────────────────────────│  │───────────────────────│
 Low morning activity               Gradual increase                          6-8 PM spike
```
**Interpretation:** Personal users, after-work posting, organic behavior

**Cluster 2 (n=534): "Multi-Peak"**
```
Hourly Distribution:
[0.01, 0.00, 0.00, 0.00, 0.02, 0.05, 0.08, 0.10, 0.08, 0.05, 0.03, 0.04, 0.06, 0.08, 0.06, 0.04, 0.05, 0.08, 0.10, 0.05, 0.03, 0.02, 0.01, 0.01]
 │                      │────────────────│              │────────────────│              │───────────────│
                         Morning peak (8-9AM)           Lunch peak          Evening peak (6-8PM)
```
**Interpretation:** Active users, multiple posting sessions, possibly mixed timezones

**Cluster 3 (n=242): "Night Owls / Different Timezone"**
```
Hourly Distribution:
[0.08, 0.10, 0.12, 0.15, 0.10, 0.05, 0.02, 0.01, 0.01, 0.01, 0.02, 0.03, 0.04, 0.03, 0.02, 0.01, 0.01, 0.02, 0.03, 0.05, 0.06, 0.08, 0.10, 0.10]
 │───────────────────────────│  │───────────────────────────────────────────────│  │────────────────────│
 Midnight-4AM spike          Very low daytime activity                          Evening increase
```
**Interpretation:** Different timezone OR night shift workers OR automated bots

### **Cluster Validation**

**Silhouette Score:**
```python
from sklearn.metrics import silhouette_score
score = silhouette_score(features_scaled, cluster_labels)
# Score: 0.51 (good separation)
```

**Interpretation:**
- Score > 0.5: Good cluster separation
- Clusters are distinct and meaningful

### **Visualization: Heatmap**

**24D Cluster Heatmap:**
```
Cluster │ Hour: 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23
────────┼────────────────────────────────────────────────────────────────────────────
   0    │ ░░░░░░░░░░░░░░░░░░░░░░██████████████████████░░░░░░░░░░░░░░  Business hours
   1    │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████████████░░░░░░  Evening heavy
   2    │ ░░░░░░░░░░░░░░████░░░░░░░░░░████░░░░░░░░░░████░░░░░░░░░░  Multi-peak
   3    │ ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████  Night owls

Legend: ░ Low activity, ▒ Medium activity, ▓ High activity, █ Very high activity
```

---

## Implementation Details

### **Code Structure**

**Located in:** `src/components/temporal_clusterer.py`

### **Key Components**

**1. Feature Engineering:**
```python
def engineer_features(self, df: pl.DataFrame) -> pl.DataFrame:
    """
    Create 2D temporal features for each account
    
    Returns:
    - account_features: DataFrame with [account, mean_hour, weekend_ratio]
    """
    
    # Extract hour from timestamp
    df = df.with_columns([
        pl.col('post_timestamp').dt.hour().alias('hour'),
        pl.col('post_timestamp').dt.weekday().alias('weekday')
    ])
    
    # Calculate mean hour
    mean_hours = df.group_by('account.username').agg([
        (pl.col('hour').mean()).alias('mean_hour')
    ])
    
    # Calculate weekend ratio
    weekend_ratios = df.group_by('account.username').agg([
        (pl.col('weekday').is_in([5, 6]).mean()).alias('weekend_ratio')
    ])
    
    # Combine features
    features = mean_hours.join(weekend_ratios, on='account.username')
    
    return features
```

**2. 24D Fingerprint Creation:**
```python
def create_24d_features(self, df: pl.DataFrame) -> np.ndarray:
    """
    Create 24-dimensional hourly fingerprints
    
    Returns:
    - feature_matrix: (num_accounts, 24) array
    """
    
    # Get unique accounts
    accounts = df['account.username'].unique().to_list()
    
    # Initialize feature matrix
    feature_matrix = np.zeros((len(accounts), 24))
    
    # For each account
    for i, account in enumerate(accounts):
        account_posts = df.filter(pl.col('account.username') == account)
        
        # Count posts per hour
        for hour in range(24):
            count = account_posts.filter(pl.col('hour') == hour).height
            feature_matrix[i, hour] = count
        
        # Normalize to probability distribution
        total = feature_matrix[i].sum()
        if total > 0:
            feature_matrix[i] /= total
    
    return feature_matrix
```

**3. Clustering:**
```python
def run_clustering(self, n_clusters=None):
    """
    Run K-means clustering on features
    
    If n_clusters is None, use elbow method to find optimal k
    """
    
    # Standardize features
    features_scaled = self.scaler.fit_transform(self.account_features)
    
    # Find optimal k if not specified
    if n_clusters is None:
        n_clusters = self._find_optimal_k(features_scaled)
    
    # Run K-means
    self.kmeans_model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    
    cluster_labels = self.kmeans_model.fit_predict(features_scaled)
    
    return cluster_labels
```

**4. Elbow Method:**
```python
def _find_optimal_k(self, features, max_k=10):
    """
    Use elbow method to find optimal number of clusters
    """
    
    inertias = []
    k_range = range(2, min(max_k, len(features) // 10))
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features)
        inertias.append(kmeans.inertia_)
    
    # Find elbow (point of maximum curvature)
    optimal_k = self._detect_elbow(k_range, inertias)
    
    return optimal_k
```

### **Performance Optimizations**

**1. Polars for Speed:**
```python
# Use Polars instead of Pandas for faster groupby operations
# 5-10x faster on large datasets
```

**2. Vectorized Operations:**
```python
# Use NumPy vectorization instead of loops
# Batch normalize all accounts at once
```

**3. Efficient Memory:**
```python
# Store only normalized fingerprints (24 floats per account)
# ~200 bytes per account vs full post history
```

---

## Results and Interpretation

### **Our Dataset Results**

**2D Clustering:**
- Optimal k: 3
- Silhouette score: 0.43
- Clusters: Morning, Evening, Night

**24D Clustering:**
- Optimal k: 4
- Silhouette score: 0.51 (18% improvement!)
- Clusters: Business hours, Evening heavy, Multi-peak, Night owls

### **Cluster Personas**

**Persona 1: "Professional/Bot Network" (Business Hours)**
- **Size:** 412 accounts (27.9%)
- **Pattern:** Clear 9-5 pattern, weekday focused
- **Suspicion Level:** Medium-High
- **Reasoning:** Could be professional accounts OR coordinated troll farm

**Persona 2: "After-Work Users" (Evening Heavy)**
- **Size:** 289 accounts (19.6%)
- **Pattern:** Evening spike (6-10 PM), weekend active
- **Suspicion Level:** Low
- **Reasoning:** Consistent with organic personal use

**Persona 3: "Active Users" (Multi-Peak)**
- **Size:** 534 accounts (36.1%)
- **Pattern:** Multiple posting sessions throughout day
- **Suspicion Level:** Low-Medium
- **Reasoning:** Could be power users OR accounts in mixed timezones

**Persona 4: "Night Owls / Foreign Timezone" (Night Pattern)**
- **Size:** 242 accounts (16.4%)
- **Pattern:** Midnight-4 AM spike in UTC
- **Suspicion Level:** Medium
- **Reasoning:** Either different timezone OR unusual bot schedule

### **Coordination Indicators**

**Strong indicators within a cluster:**
```
Cluster has:
✅ Very similar hourly patterns (correlation > 0.9)
✅ Identical posting times (same minute)
✅ Mechanical intervals (exactly every N hours)
✅ Plus content similarity
→ High probability of coordination
```

**Weak indicators:**
```
Cluster has:
⚠️ General time similarity (same period)
⚠️ Normal variation in exact times
⚠️ No content overlap
→ Likely just same timezone/schedule
```

### **Integration with Burst Detection**

**Enhanced workflow:**
```python
# 1. Detect bursts (Kleinberg)
bursts = detect_bursts(posts)

# 2. Find burst contributors
contributors = get_burst_contributors(bursts)

# 3. Get their cluster assignments
for contributor in contributors:
    cluster = get_cluster(contributor)
    
    # 4. Adaptive selection: prioritize accounts from clusters
    #    that show high burst participation
    if cluster_burst_rate[cluster] > threshold:
        high_priority_contributors.add(contributor)
```

**Result:** Focus on accounts that BOTH:
- Participate in bursts
- Belong to temporally similar clusters

---

## Limitations and Challenges

### **1. The Timezone Problem (Primary Limitation)**

**Impact: SEVERE**

**What we can't detect:**
```
Coordinated Campaign Example:
- Operator 1 in NYC (posts at 9 AM local)
- Operator 2 in Moscow (posts at 9 AM local)
- Operator 3 in Beijing (posts at 9 AM local)

In UTC:
- NYC: 14:00
- Moscow: 06:00
- Beijing: 01:00

Our clustering: Puts them in DIFFERENT clusters
Reality: They're coordinated (same local time)

MISSED COORDINATION ❌
```

**What we falsely detect:**
```
Uncoordinated Accounts Example:
- User A in California (morning poster, 9 AM)
- User B in California (morning poster, 9 AM)
- User C in California (morning poster, 9 AM)

In UTC: All post at 17:00

Our clustering: Groups them together
Reality: Just friends/random users in same timezone

FALSE POSITIVE ⚠️
```

### **2. Timezone Distribution Unknown**

**We don't know:**
- How many users are in each timezone
- If detected patterns are timezone artifacts
- Whether clustering captures coordination or geography

**Example uncertainty:**
```
Cluster of 200 accounts posting 14:00-17:00 UTC

Could be:
A) 200 coordinated bots on same schedule ✓
B) 200 random US users (morning posters) ✗
C) Mix of both ⚠️

We cannot distinguish without timezone data
```

### **3. Cultural and Behavioral Confounds**

**Work schedules vary by culture:**
```
US: 9 AM - 5 PM common
Spain: 10 AM - 2 PM, 4 PM - 8 PM (siesta)
Middle East: Different work week (Sunday-Thursday)

Same cluster ≠ Same coordination
Could just be same cultural norm
```

**Weekend patterns:**
```
Weekend ratio might indicate:
- Personal vs professional account ✓
- Religious observances (e.g., Friday holy day)
- Different work cultures
```

### **4. Temporal Drift**

**Accounts may change patterns over time:**
```
Account joins in January:
- Posts during business hours (Cluster 0)

Account in June:
- Posts during evenings (Cluster 1)

Static clustering misses this evolution
```

**Solution:** Re-run clustering on time windows

### **5. Low Post Volume Accounts**

**Accounts with few posts:**
```
Account with only 5 posts:
- 2 at 9 AM
- 3 at 2 PM

Fingerprint is unreliable (high variance)
May cluster randomly
```

**Our mitigation:**
```python
min_posts = 5  # Filter out low-volume accounts
```

### **6. Automated Posting Tools**

**Legitimate users using schedulers:**
```
User schedules posts for 9 AM, 12 PM, 3 PM every day
→ Looks like bot network
→ Actually just Buffer/Hootsuite user

False positive for coordination
```

### **7. Event-Driven Posting**

**Bursts may override normal patterns:**
```
Normal pattern: Evening poster (7-9 PM)

During breaking news at 2 PM:
- Posts immediately

Temporal clustering based on all posts:
- May misclassify due to events
```

---

## Comparison with Content-Based Detection

### **Temporal Clustering vs Content Analysis**

| Aspect | Temporal Clustering | Content Similarity |
|--------|--------------------|--------------------|
| **What it detects** | WHEN accounts post | WHAT accounts post |
| **Strength** | Hard to evade timing patterns | Direct evidence of coordination |
| **Weakness** | Timezone confusion | Easy to vary content slightly |
| **False positives** | Same timezone users | Trending topics |
| **False negatives** | Cross-timezone coordination | Paraphrased content |
| **Computational cost** | Low (just timestamps) | High (text comparison) |
| **Interpretability** | Medium | High |

### **Complementary Strengths**

**Temporal clustering finds:**
```
Accounts A, B, C post at same times
→ Operational pattern evidence
```

**Content analysis finds:**
```
Accounts A, B, C post similar content
→ Message coordination evidence
```

**Combined (strongest evidence):**
```
Accounts A, B, C:
- Post at same times ✓
- Post similar content ✓
- Use same hashtags ✓
- Retweet each other ✓
→ HIGH CONFIDENCE coordination
```

### **When to Use Each**

**Use temporal clustering when:**
- ✅ Looking for bot networks (scheduled posting)
- ✅ Identifying work-shift patterns
- ✅ Fast preliminary screening
- ✅ Timezone-homogeneous dataset

**Use content analysis when:**
- ✅ Looking for narrative coordination
- ✅ Cross-timezone campaigns
- ✅ High-confidence attribution needed
- ✅ Platform has timezone metadata

**Use both when:**
- ✅ Maximum detection coverage needed
- ✅ Multi-signal confidence scoring
- ✅ Academic rigor required

---

## Use Cases and Applications

### **1. Bot Network Detection**

**Scenario:** Identifying automated bot accounts

**Application:**
```
1. Run 24D temporal clustering
2. Look for clusters with:
   - Very tight temporal patterns (low variance)
   - Mechanical posting intervals
   - 24/7 uniform distribution (no sleep)
   
3. Flag Cluster 3 (Night owls):
   - Could be different timezone
   - OR automated bots (no human pattern)
   
4. Cross-reference with content:
   - If also posting identical content → HIGH confidence bots
```

### **2. Troll Farm Identification**

**Scenario:** Finding paid operators working shifts

**Application:**
```
1. Run 2D clustering
2. Look for Cluster 0 (Business hours):
   - Strong 9-5 pattern
   - Weekday focused
   - All in same timezone
   
3. Check for:
   - Multiple accounts with identical schedule
   - Content similarity
   - Burst participation
   
4. Conclusion: Likely coordinated troll farm
```

### **3. Cross-Platform Coordination**

**Scenario:** Same operators on multiple platforms

**Application:**
```
1. Run temporal clustering on Platform A (Twitter)
2. Run temporal clustering on Platform B (Facebook)
3. Match cluster patterns across platforms:
   - Similar hourly fingerprints
   - Same personas emerge
   
4. Accounts in matching clusters likely same operators
```

### **4. Campaign Attribution**

**Scenario:** Linking accounts to specific campaigns

**Application:**
```
1. Known campaign operates 9 AM - 5 PM EST
2. In UTC: 14:00 - 22:00
3. Cluster accounts, find those active 14:00-22:00
4. Within that cluster:
   - Check for campaign hashtags
   - Check for message similarity
   
5. Attribute accounts to campaign
```

### **5. Persona Validation**

**Scenario:** Verifying account authenticity

**Application:**
```
Claimed persona: "California tech worker"
Expected pattern: Posts 9-5 PST (17:00-01:00 UTC)

Actual pattern: Posts 2-10 AM UTC

Mismatch → Persona likely fake
Account may be operated from different location
```

---

## Practical Recommendations

### **When Temporal Clustering Works Best**

✅ **Good conditions:**
- Dataset from single timezone region
- Platform with timezone metadata available
- Looking for bot networks (automated schedules)
- Preliminary screening for coordination
- Combined with other detection methods

❌ **Poor conditions:**
- Global dataset with mixed timezones
- Looking for cross-timezone coordination
- No timezone metadata
- Used as sole detection method
- High-precision attribution needed

### **Best Practices**

**1. Always combine with other signals:**
```python
coordination_score = (
    temporal_similarity * 0.3 +
    content_similarity * 0.4 +
    network_centrality * 0.3
)
```

**2. Account for timezone uncertainty:**
```python
# Don't claim "coordination" from temporal alone
# Say: "Similar posting patterns (may indicate coordination)"
```

**3. Filter low-volume accounts:**
```python
min_posts = 5  # or higher
filtered_accounts = accounts.filter(post_count >= min_posts)
```

**4. Use 24D for precision, 2D for speed:**
```python
# Quick scan: 2D clustering
quick_personas = cluster_2d(accounts)

# Detailed analysis: 24D clustering
detailed_fingerprints = cluster_24d(high_priority_accounts)
```

**5. Validate with manual inspection:**
```python
# For each cluster
sample_accounts = random.sample(cluster, 10)
manually_review(sample_accounts)
# Check: Do patterns make sense? Real coordination?
```

---

## Summary

### **Key Takeaways**

1. **Purpose:** Temporal clustering groups accounts by **posting time patterns** to find potential coordination

2. **Two Approaches:**
   - **2D:** Broad personas (morning/evening/night)
   - **24D:** Fine-grained hourly fingerprints

3. **Primary Limitation:** **Timezone auto-conversion** prevents detecting cross-timezone coordination

4. **What We Can Detect:**
   - ✅ Same-timezone coordination
   - ✅ Bot network schedules
   - ✅ Work-shift patterns

5. **What We Cannot Detect:**
   - ❌ Cross-timezone coordination
   - ❌ Distributed networks
   - ❌ Same-local-time campaigns

6. **Best Use:** As **complementary signal** combined with content analysis, not standalone

7. **Strength:** Hard to evade (temporal patterns are intrinsic)

8. **Weakness:** Timezone confounds create false positives/negatives

### **Integration in Our Pipeline**

```
1. Burst Detection (Kleinberg)
   ↓ Identifies when activity spikes
   
2. Temporal Clustering
   ↓ Groups accounts by posting patterns
   
3. Adaptive Contributor Selection
   ↓ Prioritizes accounts in high-burst clusters
   
4. Content Coordination Detection
   ↓ Analyzes what coordinated accounts post
   
5. Network Analysis
   ↓ Maps coordination relationships
   
= Multi-Signal Coordination Detection
```

### **The Fundamental Tradeoff**

**We attempted to use temporal clustering to find coordination**

**But discovered:**
- ✅ It CAN identify behavioral patterns
- ✅ It CAN group similar accounts
- ✅ It CAN complement other methods
- ❌ It CANNOT reliably detect coordination alone (timezone limitation)

**The lesson:**
> Temporal clustering is a **useful tool in the toolkit**, but not a **magic bullet**. Like all detection methods, it has strengths and weaknesses. The timezone limitation is fundamental and cannot be overcome without additional metadata.

**Our approach:**
> Use temporal clustering as **one signal among many**, weight it appropriately, and acknowledge its limitations transparently.

---

## References

**Academic Papers:**
- Varol et al. (2017). "Online Human-Bot Interactions: Detection, Estimation, and Characterization"
- Ferrara et al. (2016). "The Rise of Social Bots"
- Golder & Macy (2011). "Diurnal and Seasonal Mood Vary with Work, Sleep, and Daylength"

**Implementation:**
- `src/components/temporal_clusterer.py` - Our implementation
- Scikit-learn K-means documentation
- Polars DataFrame operations

**Related Concepts:**
- Circadian rhythm analysis
- Behavioral fingerprinting
- Time series clustering
- Timezone normalization challenges

---

*This guide provides a comprehensive explanation of temporal clustering methodology as implemented in the CIB Mango Tree project, including its purpose, implementation, and fundamental limitations.*

**Key Message:** Temporal clustering is a **different methodology** attempting to achieve the same goal (finding coordinated accounts) as content analysis, but **limited by timezone auto-conversion** in the dataset.

