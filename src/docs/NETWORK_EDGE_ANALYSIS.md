# 🕸️ Network Edge Analysis: What Connects Accounts in Coordination Networks

## Overview
In the NetworkX coordination networks, **edges** represent different types of coordinated behaviors between accounts. Each edge has a **weight** (confidence score) and **evidence types** that justify the connection.

## 🔗 Edge Types (What Creates Connections)

### **Understanding Edge Weights**

**Edge Weight = Confidence Score** (0.0 to 1.0)

Every edge in the NetworkX graph has a **weight** property that stores the confidence score. This represents the strength of evidence for coordination between two accounts.

**Maximum Rule for Duplicate Edges:**
When two accounts are connected by multiple evidence types, we use the **MAXIMUM confidence** (not sum):
```python
if G.has_edge(account1, account2):
    # Keep the strongest evidence
    G[account1][account2]['weight'] = max(existing_weight, new_confidence)
    # But accumulate all evidence types
    G[account1][account2]['evidence_types'].append(new_type)
```

**Why Maximum?** Evidence types are not independent—accounts coordinating through content similarity may also coordinate through hashtags. Maximum confidence reflects the strongest evidence, avoiding inflation from correlated signals.

---

### 1. **RETWEET COORDINATION** 
- **Most Common Edge Type** (~96% of connections in production)
- **Connection Logic**: Accounts that retweet the same source within the same burst period
- **Confidence Formula**: `min(retweeter_count / 10.0, 1.0) + temporal_boost`
  - Base: Number of retweeters ÷ 10 (capped at 1.0)
  - Temporal boost: Up to +0.30 if RTs happen within 60 seconds
- **Example**: 15 accounts all RT @maxjett12 → creates C(15,2) = 105 edges between all pairs
- **Edge Weight Range**: 0.3 (3 retweeters) to 1.0 (10+ retweeters)

```
@account1 ←--RT coordination--→ @account2
    ↓                           ↓
Both RT @maxjett12 within Burst #5
Base confidence: 0.60 (6 retweeters)
Temporal boost: +0.15 (clustered RTs within 45 seconds)
Final Edge Weight: 0.75
```

### 2. **HASHTAG COORDINATION**
- **Connection Logic**: Accounts using identical hashtag combinations (60%+ Jaccard overlap)
- **Confidence Formula**: `min(jaccard_similarity × 1.5, 1.0)`
- **Example**: Both accounts use #PatriotsUnite #AmericaFirst #DrainTheSwamp
- **Threshold**: Minimum 2 shared hashtags required
- **Edge Weight Range**: 0.6 (minimal overlap) to 1.0 (high overlap)

```
@account1 ←--hashtag coordination--→ @account2
    ↓                                ↓
#PatriotsUnite #MAGA #Trump2024  vs  #PatriotsUnite #MAGA #AmericaFirst
Shared: {#PatriotsUnite, #MAGA} = 2 hashtags
Union: 4 unique hashtags
Jaccard Similarity: 2/4 = 0.50
Confidence: min(0.50 × 1.5, 1.0) = 0.75
Edge Weight: 0.75
```

### 3. **URL COORDINATION**
- **Connection Logic**: Accounts sharing identical URLs (normalized, tracking params stripped)
- **Confidence Formula**: `min(shared_url_count × 0.8, 1.0)`
- **Example**: Multiple accounts sharing same news article link
- **Detection**: Exact URL matching within burst periods
- **Edge Weight Range**: 0.8 (1 shared URL) to 1.0 (2+ shared URLs)

```
@account1 ←--URL coordination--→ @account2
Both share: https://example.com/article
Confidence: min(1 × 0.8, 1.0) = 0.8
Edge Weight: 0.8
```

### 4. **IDENTICAL CONTENT**
- **Strongest Evidence** (Edge Weight: 1.0)
- **Connection Logic**: Accounts posting exactly the same text content
- **Confidence Formula**: `1.0` (maximum confidence)
- **Requirements**: 95%+ text similarity using SequenceMatcher, minimum 20 characters
- **Rare but High-Impact**: Creates strongest network connections
- **Edge Weight**: Always 1.0

```
@account1 ←--identical content--→ @account2
Both post: "Patriots unite! Stand strong against the radical left agenda!"
Similarity: 100%
Edge Weight: 1.0
```

### 5. **HIGH SIMILARITY CONTENT**
- **Connection Logic**: Accounts posting very similar content (85-94% similarity)
- **Confidence Formula**: `min(similarity × 1.2, 1.0)`
- **Detection**: Uses difflib.SequenceMatcher for character-level text comparison
- **Example**: Similar talking points with slight variations
- **Edge Weight Range**: 0.85 (85% similar × 1.2 = 1.0) to 1.0

```
@account1: "Patriots unite against radical agenda"
@account2: "Patriots unite to fight the radical agenda"
Similarity: 0.88
Confidence: min(0.88 × 1.2, 1.0) = 1.0
Edge Weight: 1.0
```

### 6. **ULTRA-CONSERVATIVE TEMPORAL SYNC**
- **Connection Logic**: Accounts posting within ≤30 seconds, ≥3 times, ≥80% confidence
- **Confidence Formula**: `(timing_precision + sync_strength) / 2`
  - timing_precision = `max(0, 1 - (avg_sync_time / 30))`
  - sync_strength = `min(sync_count / 3.0, 1.0)`
- **Ultra-Conservative**: Only catches most obvious coordinated timing (80%+ threshold)
- **Example**: Both accounts post at 14:32:15 and 14:32:43 multiple times
- **Edge Weight Range**: 0.8 to 1.0 (only high-confidence temporal patterns)

```
@account1 posts: 14:32:15, 14:45:02, 15:12:44
@account2 posts: 14:32:19, 14:45:07, 15:12:47
Time diffs: 4s, 5s, 3s (avg: 4s)
timing_precision: 1 - (4/30) = 0.87
sync_strength: 3/3 = 1.0
Confidence: (0.87 + 1.0) / 2 = 0.935
Edge Weight: 0.935 ✅ (passes 80% threshold)
```

### 7. **MECHANICAL POSTING INTERVALS** (DEPRECATED - Not in Production)
- **Status**: Removed in final implementation (Phase 5 rejected)
- **Reason**: 50-70% false positive rate from timezone effects and organic patterns
- This signal is NOT used in the production system

### 8. **IDENTICAL ACTIVITY HOURS** (DEPRECATED - Not in Production)
- **Status**: Removed in final implementation (Phase 5 rejected)
- **Reason**: High false positive rate from legitimate regional/timezone patterns
- This signal is NOT used in the production system

---

## 🎯 Network Structure Analysis

### Edge Weight Calculation (Maximum Rule)
```python
# When building NetworkX graph from coordination pairs
G = nx.Graph()

for pair in coordination_pairs:
    account1 = pair['account1']
    account2 = pair['account2']
    confidence = pair['confidence']  # 0.0 to 1.0
    evidence_type = pair['type']
    
    if G.has_edge(account1, account2):
        # MAXIMUM RULE: Keep strongest evidence
        existing_weight = G[account1][account2]['weight']
        G[account1][account2]['weight'] = max(existing_weight, confidence)
        
        # Accumulate all evidence types
        G[account1][account2]['evidence_types'].append(evidence_type)
        G[account1][account2]['coordination_pairs'].append(pair)
    else:
        # New edge: confidence becomes weight
        G.add_edge(account1, account2,
                  weight=confidence,
                  evidence_types=[evidence_type],
                  coordination_pairs=[pair])
```

**Why Maximum Instead of Sum?**
- Evidence types are often **correlated** (not independent)
- Accounts posting identical content often also share hashtags
- Summing would artificially inflate confidence
- Maximum reflects the **strongest evidence** without inflation

**Example:**
```python
# Account pair detected by 3 signals:
# 1. Retweet coordination: confidence = 0.70
# 2. Hashtag coordination: confidence = 0.85
# 3. URL coordination: confidence = 0.80

# Final edge weight = max(0.70, 0.85, 0.80) = 0.85
# NOT sum (0.70 + 0.85 + 0.80 = 2.35 ❌)
```

### Confidence Score → Network Risk Level

After calculating average confidence across all pairs in a network:

```python
avg_confidence = np.mean([pair['confidence'] for pair in network_pairs])

if avg_confidence > 0.8:
    risk_level = 'HIGH'      # >80% confidence
elif avg_confidence > 0.6:
    risk_level = 'MEDIUM'    # 60-80% confidence
else:
    risk_level = 'LOW'       # <60% confidence
```

### Network Formation
1. **Coordination Pairs** → Individual connections between accounts
2. **NetworkX Graph** → All pairs become weighted edges
3. **Connected Components** → Groups of interconnected accounts
4. **Community Detection** → Sub-groups within large networks

## 📊 Real Network Example

### Network 1 (153 accounts, MEDIUM risk):
- **Primary Connection**: Retweet coordination (660+ edges)
- **Secondary**: Hashtag coordination, URL sharing
- **Structure**: MIXED (hub-and-spoke + peer-to-peer)
- **Central Hub**: @Ginger102 (35% centrality)
- **Edge Density**: 0.05 (sparse but significant connections)

```
@Ginger102 ←--RT coordination--→ @PatriotRN7
     ↕                              ↕
RT @maxjett12                  RT @maxjett12
     ↕                              ↕  
@Sullivan82 ←--hashtag coord--→ @jg410c
```

## 🔍 Edge Analysis from Visualizations

Looking at the generated network visualizations:

1. **Edge Thickness** = Confidence/Weight
2. **Edge Color** = Evidence type (if multiple types, strongest shown)
3. **Node Size** = Degree centrality
4. **Node Color** = Risk level

### Most Common Patterns:
- **Star Patterns**: Hub accounts (like @maxjett12) with many RT coordinators
- **Clique Patterns**: Small groups with multiple connection types
- **Chain Patterns**: Sequential connections through RT networks

## ⚡ Key Insights

### **Production System (Phase 4 Configuration):**

1. **Retweet coordination dominates** - Creates **96.1%** of all coordination pairs detected
2. **Content + Hashtag + URL signals** - Combined contribute only **1.3%** (but HIGH confidence)
3. **Temporal synchronization** - Contributes **2.7%** (ultra-conservative thresholds)
4. **@maxjett12 is a coordination hub** - 15+ retweeters across 8 bursts (amplification target)
5. **Multiple evidence types** create stronger edges (maximum confidence rule)
6. **Phase 5 signals REJECTED** - Behavioral patterns (mechanical intervals, activity hours) removed due to 50-70% false positive rates

### **Detection Statistics (20-day Truth Social Dataset):**
- **Total coordination pairs**: 1,110
- **Unique accounts involved**: 211
- **Networks formed**: 20
- **Multi-signal pairs**: 89 (8.0% detected by ≥2 evidence types)
- **Largest network**: 153 accounts
- **High-risk networks**: 5

### **Signal Contributions (Phased Evaluation):**
- **Phase 1 (Content only)**: 4 pairs (0.36%)
- **Phase 2 (+ Hashtag/URL)**: +11 pairs (+1.0%) → 15 total (1.4%)
- **Phase 3 (+ Retweets)**: +1,066 pairs (+96.1%) → 1,081 total (97.4%)
- **Phase 4 (+ Temporal)**: +29 pairs (+2.7%) → 1,110 total (100%) ✅ **PRODUCTION**
- **Phase 5 (+ Behavioral)**: REJECTED due to high false positives

## 🎭 Edge Interpretation for Investigators

### **Confidence-Based Prioritization:**

- **Edge Weight 1.0**: Identical content or optimal retweet amplification → **HIGHEST PRIORITY**
- **Edge Weight 0.8-0.99**: Strong evidence (high hashtag overlap, temporal sync) → **HIGH PRIORITY**
- **Edge Weight 0.6-0.79**: Medium evidence (moderate retweets, URL sharing) → **MEDIUM PRIORITY**
- **Edge Weight < 0.6**: Weak evidence → **LOW PRIORITY** (may be coincidental)

### **Multi-Signal Edges (Strongest Evidence):**
When an edge has multiple evidence types, it indicates **sophisticated coordination**:
```python
# Example multi-signal edge:
Edge: @account1 ↔ @account2
Evidence types: ['retweet_amplification', 'hashtag_coordination', 'temporal_sync']
Confidence: 0.95 (maximum across all 3 signals)
Interpretation: Highly coordinated pair using multiple tactics
```

### **Investigation Strategy:**

1. **Start with identical content edges** (confidence = 1.0) → Likely copy-paste operations
2. **Examine temporal sync edges** (confidence ≥ 0.8) → Possible automation/scheduling
3. **Trace retweet amplification hubs** → Identify whose content is being amplified
4. **Map multi-signal accounts** → Find sophisticated coordinators
5. **Analyze network structure** → Determine if hierarchical (hub-based) or distributed (peer-to-peer)

### **Red Flags:**
- ✅ **High edge weights (0.8+)**: Strong coordination evidence
- ✅ **Multiple evidence types**: Sophisticated, multi-tactic coordination
- ✅ **Temporal synchronization**: Automation or manual scheduling
- ✅ **Identical content**: Copy-paste operations or centralized script distribution
- ✅ **Hub accounts with many edges**: Key coordinators or bot controllers
- ⚠️ **Single low-confidence edge**: May be coincidental (requires context)

The edges represent **quantified evidence** of coordinated behavior, not just associations!

---

## 🔄 Complete Edge Lifecycle: From Detection to Visualization

### **Phase 1: Coordination Pair Detection**

```python
# Step 1: Detect coordination within bursts
detector = ContentCoordinationDetector()
results = detector.detect_coordination(df)

# Example pair created:
pair = {
    'type': 'retweet_amplification',
    'account1': '@user1',
    'account2': '@user2',
    'burst_index': 5,
    'confidence': 0.75,  # ← CONFIDENCE SCORE ASSIGNED
    'evidence_strength': 'HIGH',
    'rt_source': '@maxjett12'
}
```

### **Phase 2: NetworkX Graph Construction**

```python
# Step 2: Pairs become edges
G = nx.Graph()

for pair in coordination_pairs:
    account1 = pair['account1']
    account2 = pair['account2']
    confidence = pair['confidence']  # ← BECOMES EDGE WEIGHT
    
    if G.has_edge(account1, account2):
        # Maximum rule for duplicate edges
        G[account1][account2]['weight'] = max(
            G[account1][account2]['weight'],
            confidence
        )
        G[account1][account2]['evidence_types'].append(pair['type'])
    else:
        # Create new edge
        G.add_edge(account1, account2,
                  weight=confidence,  # ← EDGE WEIGHT = CONFIDENCE
                  evidence_types=[pair['type']],
                  coordination_pairs=[pair])

# Result: Graph with weighted edges
print(f"Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
# Output: Built graph: 211 nodes, 1110 edges
```

### **Phase 3: Network Formation (Connected Components)**

```python
# Step 3: Find networks using NetworkX
networks = []
for component in nx.connected_components(G):
    if len(component) >= 2:
        subgraph = G.subgraph(component)
        
        # Extract all pairs in this network
        network_pairs = []
        for edge in subgraph.edges(data=True):
            network_pairs.extend(edge[2]['coordination_pairs'])
        
        # Calculate average confidence
        avg_confidence = np.mean([p['confidence'] for p in network_pairs])
        
        networks.append({
            'accounts': list(component),
            'size': len(component),
            'avg_confidence': avg_confidence,  # ← NETWORK-LEVEL CONFIDENCE
            'subgraph': subgraph
        })

# Result: 20 networks identified
```

### **Phase 4: Risk Classification**

```python
# Step 4: Classify network risk based on average confidence
for network in networks:
    avg_conf = network['avg_confidence']
    
    if avg_conf > 0.8:
        network['risk_level'] = 'HIGH'      # Strong evidence
    elif avg_conf > 0.6:
        network['risk_level'] = 'MEDIUM'    # Moderate evidence
    else:
        network['risk_level'] = 'LOW'       # Weak evidence

# Example output:
# Network 1: 153 accounts, avg_confidence=0.70 → MEDIUM risk
# Network 2: 12 accounts, avg_confidence=0.95 → HIGH risk
```

### **Phase 5: Visualization**

```python
# Step 5: Visualize networks with edge weights reflected
import matplotlib.pyplot as plt

# Edge width scaled by confidence (weight)
edge_widths = [G[u][v]['weight'] * 3 for u, v in G.edges()]

# Edge colors by evidence type
edge_colors = []
for u, v, data in G.edges(data=True):
    if 'identical_content' in data['evidence_types']:
        edge_colors.append('red')      # Strongest
    elif 'temporal_sync' in data['evidence_types']:
        edge_colors.append('orange')   # Strong
    elif 'retweet_amplification' in data['evidence_types']:
        edge_colors.append('blue')     # Common
    else:
        edge_colors.append('gray')     # Other

# Draw network
pos = nx.spring_layout(G)
nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.5)
nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue')
nx.draw_networkx_labels(G, pos, font_size=8)

plt.title('Coordination Network (Edge Width = Confidence)')
plt.savefig('network_visualization.png')
```

### **Phase 6: Reporting & Analysis**

```python
# Step 6: Generate human-readable reports
print(f"\n📊 COORDINATION NETWORK SUMMARY")
print(f"Total networks: {len(networks)}")
print(f"Total accounts involved: {sum(n['size'] for n in networks)}")
print(f"High-risk networks: {sum(1 for n in networks if n['risk_level'] == 'HIGH')}")

for network in sorted(networks, key=lambda x: x['size'], reverse=True)[:5]:
    print(f"\n🕸️ Network {network['network_id']}: {network['size']} accounts")
    print(f"   Average confidence: {network['avg_confidence']:.2f}")
    print(f"   Risk level: {network['risk_level']}")
    print(f"   Evidence types: {', '.join(set(network['evidence_types']))}")
```

---

## 📋 Edge Lifecycle Summary

| Phase | Input | Output | Confidence Usage |
|-------|-------|--------|------------------|
| **1. Detection** | Burst posts | Coordination pairs | Confidence score calculated (0.0-1.0) |
| **2. Graph Building** | Coordination pairs | NetworkX graph | Confidence → edge weight (max rule) |
| **3. Network Formation** | NetworkX graph | Connected components | Edges define connectivity |
| **4. Network Analysis** | Connected components | Network metrics | Avg confidence calculated |
| **5. Risk Classification** | Average confidence | Risk level | HIGH/MEDIUM/LOW assignment |
| **6. Visualization** | NetworkX graph | Visual plots | Edge width/color from confidence |
| **7. Reporting** | All data | Human reports | Confidence displayed in summaries |

---

## 🎯 Key Takeaways for Investigators

### **What Edges Tell You:**

1. **Edge Exists** → Two accounts coordinated (at least once)
2. **Edge Weight (0.0-1.0)** → Strength of evidence
3. **Multiple Evidence Types** → Sophisticated, multi-tactic coordination
4. **Edge in Large Network** → Part of broader campaign
5. **High-Weight Edges** → Priority investigation targets

### **What to Look For:**

- 🔴 **Weight = 1.0**: Identical content (copy-paste operations)
- 🟠 **Weight ≥ 0.8**: Strong evidence (temporal sync, high hashtag overlap)
- 🔵 **Weight 0.6-0.79**: Moderate evidence (retweet coordination)
- ⚪ **Weight < 0.6**: Weak evidence (may be organic)

### **Investigation Workflow:**

```
1. Filter to high-weight edges (≥0.8)
   ↓
2. Identify multi-signal edges (multiple evidence types)
   ↓
3. Map accounts involved in high-confidence edges
   ↓
4. Examine network structure (hub vs distributed)
   ↓
5. Trace content sources (coordination hubs)
   ↓
6. Assess operational sophistication
```

**Remember:** Edges are **not just connections**—they are **quantified, evidence-based detections** with confidence scores that reflect the strength of coordination evidence!

