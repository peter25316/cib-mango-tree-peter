# 🕸️ Network Edge Analysis: What Connects Accounts in Coordination Networks

## Overview
In the NetworkX coordination networks, **edges** represent different types of coordinated behaviors between accounts. Each edge has a **weight** (confidence score) and **evidence types** that justify the connection.

## 🔗 Edge Types (What Creates Connections)

### 1. **RETWEET COORDINATION** 
- **Most Common Edge Type** (~95% of connections)
- **Connection Logic**: Accounts that retweet the same source within the same burst period
- **Edge Weight**: Based on amplification strength + temporal synchronization
- **Example**: 15 accounts all RT @maxjett12 → creates 105 edges between all pairs
- **Enhanced Feature**: Temporal synchronization boosts confidence if RTs happen within 60 seconds

```
@account1 ←--RT coordination--→ @account2
    ↓                           ↓
Both RT @maxjett12 within Burst #5
Edge Weight: 0.67 (medium confidence)
```

### 2. **HASHTAG COORDINATION**
- **Connection Logic**: Accounts using identical hashtag combinations (60%+ overlap)
- **Edge Weight**: Based on Jaccard similarity coefficient
- **Example**: Both accounts use #PatriotsUnite #AmericaFirst #DrainTheSwamp
- **Threshold**: Minimum 2 shared hashtags required

```
@account1 ←--hashtag coordination--→ @account2
    ↓                                ↓
#PatriotsUnite #MAGA #Trump2024  vs  #PatriotsUnite #MAGA #AmericaFirst
Jaccard Similarity: 66% → Edge Weight: 0.99
```

### 3. **URL COORDINATION**
- **Connection Logic**: Accounts sharing identical URLs
- **Edge Weight**: 0.8 per shared URL
- **Example**: Multiple accounts sharing same news article link
- **Detection**: Exact URL matching within burst periods

### 4. **IDENTICAL CONTENT**
- **Strongest Evidence** (Edge Weight: 1.0)
- **Connection Logic**: Accounts posting exactly the same text content
- **Requirements**: 95%+ text similarity, minimum 20 characters
- **Rare but High-Impact**: Creates strongest network connections

### 5. **HIGH SIMILARITY CONTENT**
- **Connection Logic**: Accounts posting very similar content (85%+ similarity)
- **Edge Weight**: Scales with similarity score (0.85-1.0)
- **Detection**: Uses SequenceMatcher for text comparison
- **Example**: Similar talking points with slight variations

### 6. **ULTRA-CONSERVATIVE TEMPORAL SYNC**
- **Connection Logic**: Accounts posting within 30 seconds, 3+ times
- **Edge Weight**: 0.8+ based on timing precision
- **Ultra-Conservative**: Only catches most obvious coordinated timing
- **Example**: Both accounts post at 14:32:15 and 14:32:43 multiple times

### 7. **MECHANICAL POSTING INTERVALS** 
- **Connection Logic**: 95%+ identical posting intervals under 1 hour
- **Edge Weight**: 0.9+ (indicates bot-like behavior)
- **Detection**: Statistical analysis of posting time patterns
- **Example**: Both accounts post exactly every 37 minutes

### 8. **IDENTICAL ACTIVITY HOURS**
- **Connection Logic**: 80%+ identical active hours with narrow windows (≤5 hours)
- **Edge Weight**: 0.85+ based on schedule coordination
- **Example**: Both accounts only active 2-4 PM and 8-9 PM daily

## 🎯 Network Structure Analysis

### Edge Weight Calculation
```python
# Multiple evidence types boost confidence
if edge_exists:
    edge_weight = max(existing_weight, new_confidence)
    evidence_types.append(new_type)
else:
    edge_weight = base_confidence
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

1. **Retweet coordination dominates** - Creates 95% of all edges
2. **@maxjett12 is a major hub** - 15 retweeters across 8 bursts
3. **Temporal synchronization enhances** RT coordination confidence
4. **Multiple evidence types** create stronger edges (higher weights)
5. **Ultra-conservative thresholds** prevent false positives

## 🎭 Edge Interpretation for Investigators

- **High-weight edges (0.8+)**: Strong evidence of coordination
- **Multiple evidence types**: More sophisticated coordination
- **Temporal sync edges**: Possible automation/scheduling
- **Identical content edges**: Copy-paste operations or bot networks
- **Hub connections**: Key coordinators worth investigating

The edges represent **quantified evidence** of coordinated behavior, not just associations!
