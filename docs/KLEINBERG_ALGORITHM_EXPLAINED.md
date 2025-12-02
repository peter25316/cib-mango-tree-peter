# 📊 Kleinberg Burst Detection Algorithm - Complete Guide

A comprehensive explanation of Kleinberg's burst detection algorithm used in temporal coordination analysis.

---

## 📚 Table of Contents

1. [Overview](#overview)
2. [What is a "Burst"?](#what-is-a-burst)
3. [Why Detect Bursts?](#why-detect-bursts)
4. [The Intuition](#the-intuition)
5. [How the Algorithm Works](#how-the-algorithm-works)
6. [Mathematical Foundation](#mathematical-foundation)
7. [Parameters Explained](#parameters-explained)
8. [Step-by-Step Example](#step-by-step-example)
9. [Implementation Details](#implementation-details)
10. [Interpretation Guide](#interpretation-guide)
11. [Real-World Applications](#real-world-applications)
12. [Advantages and Limitations](#advantages-and-limitations)

---

## Overview

### **What is Kleinberg's Algorithm?**

Kleinberg's burst detection algorithm (2003) is a **temporal pattern detection method** that identifies periods of unusually high activity in time-stamped data streams.

### **Original Paper**
*"Bursty and Hierarchical Structure in Streams"* by Jon Kleinberg (2003)  
Published in: Data Mining and Knowledge Discovery, 7(4), 373-397

### **Core Idea**
The algorithm models document/event arrival as an **infinite-state automaton** where different states represent different activity levels. It identifies "bursts" as transitions to higher activity states.

### **Key Applications**
- 📧 Email burst detection (sudden increase in emails about a topic)
- 📰 News event detection (spike in news articles)
- 🐦 Social media trending (surge in posts/hashtags)
- 🤖 **Coordinated bot activity** (synchronized posting campaigns)

---

## What is a "Burst"?

### **Simple Definition**
A **burst** is a time period where activity is **significantly higher than baseline**.

### **Visual Example**

```
Posts per Hour:
│
40│           ╭──────╮  ← BURST (2x baseline)
  │           │      │
20│───────────┤      ├────  ← Baseline activity
  │           │      │
10│═══════════════════════
  │
  └─────────────────────────→ Time
   0    5   10   15   20   25
```

### **Everyday Analogy**

**Normal coffee shop:**
- 8-10 customers per hour (baseline)

**Burst:**
- 30 customers in one hour (morning rush!)
- 3x normal activity = burst detected

### **In Social Media**

**Normal posting:**
- Account posts 2-3 times per day

**Burst:**
- Account posts 20 times in one hour
- Likely automated or coordinated behavior

---

## Why Detect Bursts?

### **1. Identify Coordinated Campaigns**

**Problem:** Bot networks coordinate to amplify messages

**Solution:** Detect when many accounts suddenly post together
```
Hour 1-10:  5-8 posts per hour (normal)
Hour 11:    87 posts (BURST!) ← Coordinated campaign activated
Hour 12-20: 6-9 posts per hour (back to normal)
```

### **2. Find Important Events**

**Real-world example:**
```
Breaking News Event:
├─ Before: 100 tweets/hour about "election"
├─ Event happens (scandal breaks)
└─ After: 5,000 tweets/hour ← BURST indicates important event
```

### **3. Track Information Diffusion**

**How messages spread:**
```
Message posted → Small group shares → BURST as it goes viral → Decay
```

### **4. Detect Anomalies**

**Normal vs Abnormal:**
- Normal: Gradual changes in posting frequency
- Abnormal: Sudden spike suggests external intervention (bots, coordination)

---

## The Intuition

### **The "Two-State" Mental Model**

Imagine the system has two modes:

**State 0 (Normal/Quiet):**
- 🐢 Slow posting rate
- Background activity
- Organic behavior

**State 1 (Burst/Active):**
- 🚀 Fast posting rate
- Elevated activity
- Something interesting happening

### **State Transition Model**

```
       ┌─────────┐
       │ State 0 │  (Quiet: 5 posts/hour)
       │ Normal  │
       └────┬────┘
            │
      Cost to │ switch
            ↓
       ┌─────────┐
       │ State 1 │  (Active: 20 posts/hour)
       │ Burst   │
       └─────────┘
```

**The algorithm asks:**
- Is the activity increase **significant enough** to justify switching states?
- What **sequence of state transitions** best explains the observed data?

### **Real-World Analogy: Traffic Light**

Think of it like a traffic light controller:

**Green (State 0):** Normal traffic flow (5 cars/minute)
- Stay green as long as traffic is light

**Red (State 1):** Heavy traffic (20 cars/minute)
- Switch to red when traffic spikes
- But don't switch too frequently (costly)

**The algorithm decides:**
- When to switch from green → red (burst starts)
- When to switch from red → green (burst ends)
- Balances: Accuracy vs Sensitivity

---

## How the Algorithm Works

### **Step 1: Model as Automaton**

The algorithm uses an **infinite-state automaton** with states *q = 0, 1, 2, 3, ...*

**Each state represents an activity level:**
```
State 0: Baseline (e.g., 5 posts/hour)
State 1: 2× baseline (10 posts/hour)
State 2: 4× baseline (20 posts/hour)
State 3: 8× baseline (40 posts/hour)
...
```

**State transition rate:**
```
rq = r0 × s^q

Where:
- r0 = baseline rate
- s = scaling factor (typically s=2)
- q = state number
```

### **Step 2: Calculate Costs**

**Two types of costs:**

**1. Emission Cost (Cost of observing data in a state):**
```
If in state q, and we observe gap g between posts:
Cost = -log(P(gap g | state q))

Interpretation:
- Small gaps in high state = low cost (expected)
- Large gaps in high state = high cost (unexpected)
```

**2. Transition Cost (Cost of switching states):**
```
τ(i, j) = γ × ln(s) × |i - j|

Where:
- γ (gamma) = controls burst granularity
- s = scaling factor
- |i - j| = how many states we jump

Interpretation:
- Jumping 1 state: Small cost
- Jumping 5 states: Large cost (penalizes big jumps)
```

### **Step 3: Find Optimal State Sequence**

Use **dynamic programming** (Viterbi algorithm) to find the state sequence that **minimizes total cost**.

**Total cost = Emission costs + Transition costs**

**Example:**
```
Observed posts at times: [1, 2, 2.5, 3, 10, 15, 16, 16.5, 17]

Algorithm finds best state sequence:
Times:  [1, 2, 2.5, 3] [10, 15, 16, 16.5, 17]
States: [0, 0, 0,   0] [2,  2,  2,  2,    2]
         ↑              ↑
      Normal        Burst (state 2 = 4× faster)
```

### **Step 4: Extract Bursts**

**A burst is defined as:**
```
Burst = {
    start_time: When state > 0 begins,
    end_time: When state returns to 0,
    state: Maximum state reached,
    weight: Burst intensity
}
```

---

## Mathematical Foundation

### **Formal Problem Setup**

**Input:**
- Sequence of timestamps: *t₁, t₂, ..., tₙ*
- Gaps between events: *gᵢ = tᵢ₊₁ - tᵢ*

**Model:**
- Infinite-state automaton with states *q ∈ {0, 1, 2, ...}*
- Each state has exponential inter-arrival rate *rq*

### **State Transition Rates**

```
rq = r0 × s^q

Where:
- r0 = baseline rate = 1 / (average gap)
- s = scaling factor (s > 1)
- q = state index
```

**Example calculation:**
```
Average gap = 10 minutes
r0 = 1/10 = 0.1 events/minute

With s=2:
- State 0: r0 = 0.1 events/min (10 min gaps)
- State 1: r1 = 0.2 events/min (5 min gaps)
- State 2: r2 = 0.4 events/min (2.5 min gaps)
- State 3: r3 = 0.8 events/min (1.25 min gaps)
```

### **Cost Functions**

**Emission Cost (observing gap g in state q):**
```
c(q, g) = -log(rq × exp(-rq × g))
        = -log(rq) + rq × g
        = rq × g - log(rq)
```

**Transition Cost (moving from state i to state j):**
```
τ(i, j) = (j - i) × γ × log(s)    if j ≥ i
        = 0                        if j < i

Where:
- γ = granularity parameter
- log(s) = natural log of scaling factor
```

### **Optimization Problem**

**Find state sequence Q = (q₁, q₂, ..., qₙ) that minimizes:**
```
Cost(Q) = Σ c(qᵢ, gᵢ) + Σ τ(qᵢ, qᵢ₊₁)
          i=1         i=1

Subject to: qᵢ ∈ {0, 1, 2, ...}
```

**Solution:** Dynamic programming (Viterbi algorithm)

### **Dynamic Programming Recurrence**

```
Let C[i][q] = minimum cost to reach state q at time i

C[i][q] = min over all previous states p {
    C[i-1][p] + τ(p, q) + c(q, gᵢ)
}

Base case: C[0][0] = 0 (start in state 0)

Final answer: min over q { C[n][q] }
```

### **Complexity**

- **Time:** *O(n × Q²)* where n = number of events, Q = max state considered
- **Space:** *O(n × Q)*
- **In practice:** Q is bounded (rarely exceeds 10), so ~O(n)

---

## Parameters Explained

### **Parameter 1: s (Scaling Factor)**

**Definition:** How much faster each state is compared to the previous one.

**Formula:** *rq = r0 × s^q*

**Typical values:** s = 1.5 to 2.5

**Effects:**

#### **s = 2.0 (Standard)**
```
State 0: 1× baseline
State 1: 2× baseline
State 2: 4× baseline
State 3: 8× baseline

Result: Detects clear doubling of activity
```

#### **s = 1.5 (Sensitive)**
```
State 0: 1× baseline
State 1: 1.5× baseline
State 2: 2.25× baseline
State 3: 3.375× baseline

Result: More sensitive, detects smaller increases
```

#### **s = 3.0 (Conservative)**
```
State 0: 1× baseline
State 1: 3× baseline
State 2: 9× baseline
State 3: 27× baseline

Result: Less sensitive, only detects major spikes
```

**Recommendation:** s = 2.0 is a good default

---

### **Parameter 2: γ (Gamma - Granularity)**

**Definition:** Controls the "cost" of state transitions.

**Formula:** *τ(i, j) = γ × ln(s) × |i - j|*

**Typical values:** γ = 0.5 to 2.0

**Effects:**

#### **γ = 0.5 (Many small bursts)**
```
Low transition cost → Algorithm changes states frequently

Result:
├─ Burst 1 (short)
├─ Normal
├─ Burst 2 (short)
├─ Normal
├─ Burst 3 (short)
└─ ...

Many short, fine-grained bursts
```

#### **γ = 1.0 (Balanced, Standard)**
```
Moderate transition cost → Balanced sensitivity

Result:
├─ Normal
├─ Burst 1 (moderate length)
├─ Normal
└─ Burst 2 (moderate length)

Balanced number of bursts
```

#### **γ = 2.0 (Few long bursts)**
```
High transition cost → Algorithm stays in states longer

Result:
├─ Normal
├─ Burst 1 (very long, spans multiple peaks)
└─ Normal

Fewer, longer bursts
```

**Recommendation:** γ = 1.0 is a good default

---

### **Parameter Selection Guide**

| Goal | s | γ |
|------|---|---|
| **Detect subtle changes** | 1.5 | 0.5 |
| **Standard detection (balanced)** | 2.0 | 1.0 |
| **Only major spikes** | 2.5 | 1.5 |
| **Fine-grained bursts** | 2.0 | 0.5 |
| **Long-duration bursts** | 2.0 | 2.0 |

**Our Project Uses:** s = 2.0, γ = 1.0 (standard balanced approach)

---

## Step-by-Step Example

### **Sample Data: Social Media Posts**

```
Timestamps (in hours):
0.5, 1.0, 1.8, 2.5, 3.2, 10.1, 10.2, 10.3, 10.5, 10.7, 11.0, 15.5, 16.2, 17.0

Posts per hour:
Hours 0-5:   7 posts (normal)
Hours 10-11: 6 posts in 1 hour (BURST!)
Hours 15-17: 3 posts (normal)
```

### **Step 1: Calculate Gaps**

```
Gaps between consecutive posts:
0.5, 0.8, 0.7, 0.7, 6.9, 0.1, 0.1, 0.2, 0.2, 0.3, 4.5, 0.7, 0.8

Average gap: 1.33 hours
r0 = 1/1.33 = 0.75 events/hour
```

### **Step 2: Define States (s=2)**

```
State 0: r0 = 0.75 events/hour (1.33 hr gaps)
State 1: r1 = 1.50 events/hour (0.67 hr gaps)
State 2: r2 = 3.00 events/hour (0.33 hr gaps)
State 3: r3 = 6.00 events/hour (0.17 hr gaps)
```

### **Step 3: Calculate Costs**

**For each gap, in each state:**

**Gap 1 (0.5 hours):**
```
State 0: c(0, 0.5) = 0.75×0.5 - ln(0.75) = 0.375 + 0.288 = 0.663
State 1: c(1, 0.5) = 1.50×0.5 - ln(1.50) = 0.750 - 0.405 = 0.345 ← Lower
State 2: c(2, 0.5) = 3.00×0.5 - ln(3.00) = 1.500 - 1.099 = 0.401
```

**Gap 7 (0.1 hours - during burst):**
```
State 0: c(0, 0.1) = 0.75×0.1 - ln(0.75) = 0.075 + 0.288 = 0.363
State 1: c(1, 0.1) = 1.50×0.1 - ln(1.50) = 0.150 - 0.405 = -0.255
State 2: c(2, 0.1) = 3.00×0.1 - ln(3.00) = 0.300 - 1.099 = -0.799 ← Lower!
State 3: c(3, 0.1) = 6.00×0.1 - ln(6.00) = 0.600 - 1.792 = -1.192 ← Even lower!
```

**Interpretation:** Short gaps (0.1 hr) have lower cost in high states (bursts)

### **Step 4: Dynamic Programming**

**For each position, find optimal state:**

```
Position 1-6 (normal gaps):
Optimal states: 0, 0, 0, 0, 0, 0

Position 7-11 (short gaps during burst):
Optimal states: 2, 2, 2, 2, 2

Position 12-14 (normal gaps):
Optimal states: 0, 0, 0
```

### **Step 5: Extract Bursts**

```
Burst detected:
├─ Start: Position 7 (time 10.1 hours)
├─ End: Position 11 (time 11.0 hours)
├─ State: 2 (4× normal activity)
└─ Duration: 0.9 hours

Interpretation: 
6 posts in ~1 hour vs normal 0.75 posts/hour
Clear coordinated posting activity
```

### **Visual Output**

```
State Sequence:
│
2│        ╭────────╮
 │        │        │
1│        │        │
 │        │        │
0│════════╯        ╰════════
 │
 └─────────────────────────→ Time
  0   5   10   15   20
      
      ↑            ↑
   Normal       Burst!
```

---

## Implementation Details

### **Our Implementation**

Located in: `src/components/kleinberg_utils.py`

### **Key Function**

```python
def kleinberg(offsets, s=2, gamma=1):
    """
    Kleinberg's burst detection algorithm.
    
    Parameters:
    -----------
    offsets : array-like
        Timestamps of events (in seconds or any time unit)
    s : float
        Scaling factor for state rates (default: 2.0)
    gamma : float
        Transition cost parameter (default: 1.0)
    
    Returns:
    --------
    bursts : ndarray
        Array of bursts, each row: [state, start_time, end_time]
    """
```

### **Algorithm Steps in Code**

**1. Prepare data:**
```python
offsets = np.sort(offsets)
gaps = np.diff(offsets)
n = len(gaps)
```

**2. Calculate baseline rate:**
```python
r0 = 1.0 / np.mean(gaps)
```

**3. Initialize state costs:**
```python
# For each possible state q
for q in range(max_state):
    rq = r0 * (s ** q)
    cost[q] = rq * gaps - np.log(rq)
```

**4. Dynamic programming:**
```python
# dp[i][q] = minimum cost to be in state q at position i
for i in range(1, n):
    for q in range(max_state):
        min_cost = infinity
        for p in range(max_state):
            transition_cost = gamma * log(s) * max(0, q - p)
            total = dp[i-1][p] + transition_cost + cost[q][i]
            min_cost = min(min_cost, total)
        dp[i][q] = min_cost
```

**5. Backtrack to find optimal path:**
```python
# Find best ending state
best_state = argmin(dp[n-1])

# Backtrack to reconstruct state sequence
states = backtrack(dp, best_state)
```

**6. Extract bursts:**
```python
bursts = []
in_burst = False
for i, state in enumerate(states):
    if state > 0 and not in_burst:
        burst_start = offsets[i]
        burst_state = state
        in_burst = True
    elif state == 0 and in_burst:
        burst_end = offsets[i]
        bursts.append([burst_state, burst_start, burst_end])
        in_burst = False
```

### **Performance Optimizations**

**1. Maximum State Limiting:**
```python
# Don't consider unrealistically high states
max_state = min(50, int(np.log(n) / np.log(s)) + 3)
```

**2. Sparse Cost Matrix:**
```python
# Only compute costs for reachable states
```

**3. Early Termination:**
```python
# If cost becomes too high, prune search
```

---

## Interpretation Guide

### **Burst State Meanings**

| State | Activity Level | Interpretation |
|-------|---------------|----------------|
| **0** | Baseline | Normal, organic activity |
| **1** | 2× baseline | Slight increase, possibly organic |
| **2** | 4× baseline | Clear spike, investigate |
| **3** | 8× baseline | Major spike, likely coordinated |
| **4+** | 16×+ baseline | Extreme spike, very suspicious |

### **Burst Duration**

**Short bursts (< 1 hour):**
- Flash mob style coordination
- Automated burst
- Event-driven spike

**Medium bursts (1-6 hours):**
- Coordinated campaign window
- Work shift pattern
- Event coverage

**Long bursts (6+ hours):**
- Sustained campaign
- News cycle
- Multiple coordinated groups

### **Burst Frequency**

**Many small bursts:**
- Automated posting patterns
- Multiple small campaigns
- May need higher γ to consolidate

**Few large bursts:**
- Major coordinated events
- Campaign phases
- Natural event-driven activity

### **Contributor Analysis**

**After burst detection, analyze WHO posted:**

```python
# For each burst
for burst in bursts:
    # Find accounts that posted during burst
    contributors = find_accounts_in_burst(burst)
    
    # Look for patterns:
    # - Same accounts in multiple bursts?
    # - Similar posting times?
    # - Content similarity?
```

---

## Real-World Applications

### **1. Bot Network Detection**

**Problem:** Coordinated bot accounts post synchronized messages

**Application:**
```
1. Detect bursts in overall posting activity
2. Identify accounts posting during bursts
3. Analyze if same accounts appear in multiple bursts
4. Check for content similarity within bursts
→ Strong evidence of coordination
```

### **2. Election Monitoring**

**Problem:** Coordinated campaigns to influence elections

**Application:**
```
Normal:   200 election posts/hour
Burst 1:  5,000 posts/hour (debate starts)
Burst 2:  8,000 posts/hour (coordinated campaign)
Burst 3:  3,000 posts/hour (debate ends)

Analysis:
- Burst 1 & 3: Natural (event-driven)
- Burst 2: Suspicious (no event trigger, different accounts)
```

### **3. Trending Topic Analysis**

**Problem:** Distinguish organic trends from manufactured ones

**Application:**
```
Organic trend:
├─ Gradual increase (no clear burst start)
├─ Diverse accounts
└─ Varied content

Manufactured trend:
├─ Sharp burst (clear start time)
├─ Repeated accounts across bursts
└─ Similar/identical content
```

### **4. Crisis Response**

**Problem:** Identify when critical events occur

**Application:**
```
Monitoring keyword: "earthquake"

Normal:   5 posts/hour
Burst:    500 posts/hour at 3:15 PM

Action: Earthquake likely occurred at 3:15 PM
→ Trigger emergency response protocols
```

---

## Advantages and Limitations

### **✅ Advantages**

**1. No Training Data Needed**
- Unsupervised algorithm
- Works on unlabeled data
- No need for historical examples

**2. Adaptive to Baseline**
- Automatically learns "normal" activity
- Works across different domains
- No manual threshold setting

**3. Hierarchical States**
- Captures different burst intensities
- State 2 vs State 3 = different severity
- More nuanced than binary burst/no-burst

**4. Theoretically Grounded**
- Based on automaton theory
- Optimal solution (dynamic programming)
- Formal cost minimization

**5. Interpretable Results**
- Clear burst start/end times
- Burst intensity (state number)
- Easy to explain to non-technical audience

### **❌ Limitations**

**1. Assumes Exponential Inter-Arrival**
- Real data may not follow exponential distribution
- Can lead to suboptimal burst detection
- Workaround: Pre-process data to stabilize variance

**2. Sensitive to Parameters**
- Different s and γ give different results
- Requires parameter tuning
- No universal "best" parameters

**3. Global Baseline**
- Uses single baseline for entire stream
- Doesn't adapt to long-term trends
- Workaround: Re-run algorithm on windows

**4. No Account-Level Analysis**
- Detects bursts in aggregate
- Doesn't inherently identify coordinating accounts
- Workaround: Our enhancement (adaptive contributor selection)

**5. Computationally Intensive**
- O(n × Q²) time complexity
- Can be slow for very long streams
- Workaround: Process in batches

**6. Memory Requirements**
- Needs to store full timestamp sequence
- Can be problematic for very large datasets
- Workaround: Streaming approximations

### **When to Use Kleinberg**

**✅ Good for:**
- Temporal anomaly detection
- Event identification
- Coordination campaign detection
- Unsupervised analysis

**❌ Not ideal for:**
- Real-time streaming (use approximations)
- Account-level attribution (need additional analysis)
- Very large scale (billions of events)
- Non-temporal data

---

## Comparison with Other Methods

### **Kleinberg vs Simple Threshold**

**Simple Threshold:**
```python
burst = posts_per_hour > 50  # Fixed threshold
```

**Problems:**
- Doesn't adapt to baseline
- Same threshold for 10 vs 10,000 followers
- Binary (burst/no-burst)

**Kleinberg:**
- ✅ Adaptive baseline
- ✅ Account for natural variation
- ✅ Hierarchical burst levels

---

### **Kleinberg vs Statistical Process Control**

**SPC (Control Charts):**
```python
burst = posts_per_hour > (mean + 3*std)
```

**Similarities:**
- Both detect deviations from normal

**Differences:**
- SPC: Statistical thresholds
- Kleinberg: Optimal state sequence
- Kleinberg captures temporal structure better

---

### **Kleinberg vs Machine Learning**

**ML Approach:**
```python
model = train_classifier(labeled_bursts)
burst = model.predict(new_data)
```

**ML Advantages:**
- Can learn complex patterns
- Incorporate multiple features

**Kleinberg Advantages:**
- ✅ No training data needed
- ✅ Interpretable (state sequence)
- ✅ Theoretically grounded

**Best approach:** Combine both (use Kleinberg features in ML model)

---

## Extensions and Enhancements

### **Our Enhancement: Adaptive Contributor Selection**

**Problem:** Standard Kleinberg only detects burst times, not WHO coordinated

**Our Solution:**
```python
def select_burst_contributors(burst, all_accounts, clusters):
    """
    After detecting burst, find accounts that:
    1. Posted during the burst
    2. Belong to clusters with high burst participation
    3. Have posting patterns consistent with coordination
    """
    
    contributors = []
    for account in all_accounts:
        if posted_in_burst(account, burst):
            cluster = get_cluster(account)
            participation = cluster_burst_participation(cluster)
            
            if participation > threshold:
                contributors.append(account)
    
    return contributors
```

**Benefit:** Focus coordination analysis on truly coordinated accounts, not casual participants

---

### **Other Possible Extensions**

**1. Multi-Scale Bursts**
- Detect bursts at different time scales (hourly, daily, weekly)
- Hierarchical burst structure

**2. Content-Aware Bursts**
- Incorporate content similarity into cost function
- Detect bursts of specific topics/hashtags

**3. Network-Aware Bursts**
- Consider account relationships
- Bursts in connected subgraphs more suspicious

**4. Seasonal Decomposition**
- Remove periodic patterns before burst detection
- Better baseline for recurring patterns

---

## Quick Reference

### **Algorithm Cheat Sheet**

```python
# Basic usage
from kleinberg_utils import kleinberg

# Timestamps (in seconds, minutes, hours, etc.)
timestamps = [1, 5, 7, 9, 50, 51, 52, 53, 100]

# Run algorithm
bursts = kleinberg(timestamps, s=2.0, gamma=1.0)

# Each burst: [state, start_time, end_time]
for state, start, end in bursts:
    print(f"Burst: State {state}, {start} to {end}")
```

### **Parameter Quick Guide**

| Want to... | Adjust |
|------------|--------|
| Detect smaller increases | Decrease s (e.g., 1.5) |
| Only major spikes | Increase s (e.g., 2.5) |
| More bursts (fine-grained) | Decrease γ (e.g., 0.5) |
| Fewer bursts (coarse) | Increase γ (e.g., 2.0) |
| Standard balanced | s=2.0, γ=1.0 |

### **Interpretation Quick Guide**

| Observation | Interpretation |
|-------------|----------------|
| State 0 | Normal activity |
| State 1-2 | Mild burst (investigate) |
| State 3-4 | Strong burst (likely coordinated) |
| State 5+ | Extreme burst (very suspicious) |
| Many short bursts | Automated pattern or high γ needed |
| Few long bursts | Major campaigns or low γ needed |
| Same accounts across bursts | Coordination evidence |

---

## Additional Resources

### **Academic Papers**

1. **Original Paper:**
   - Kleinberg, J. (2003). "Bursty and hierarchical structure in streams"
   - Data Mining and Knowledge Discovery, 7(4), 373-397

2. **Applications:**
   - Mathioudakis & Koudas (2010). "TwitterMonitor: Trend detection over the Twitter stream"
   - Lehmann et al. (2012). "Dynamical classes of collective attention in twitter"

### **Implementations**

- **Python (ours):** `src/components/kleinberg_utils.py`
- **R package:** `bursts` package
- **MATLAB:** Available on MathWorks File Exchange

### **Related Algorithms**

- **EWMA (Exponential Weighted Moving Average):** For streaming detection
- **CUSUM (Cumulative Sum):** For change point detection
- **Bayesian Change Point Detection:** For probabilistic burst detection

---

## Summary

### **Key Takeaways**

1. **Kleinberg detects bursts** = periods of elevated activity
2. **Works by modeling** activity as state automaton
3. **Two parameters:** s (sensitivity) and γ (granularity)
4. **Optimal solution** via dynamic programming
5. **Unsupervised** - no training data needed
6. **Interpretable** - clear burst periods and intensities
7. **Best for** temporal anomaly detection in streams
8. **Enhanced in our project** with adaptive contributor selection

### **When to Use**

- ✅ Detecting coordinated posting campaigns
- ✅ Identifying important events
- ✅ Finding temporal anomalies
- ✅ Unsupervised analysis scenarios

### **Integration in Our Pipeline**

```
1. Kleinberg detects burst periods
   ↓
2. Adaptive contributor selection identifies coordinating accounts
   ↓
3. Content analysis examines what they posted
   ↓
4. Network analysis reveals coordination structure
   ↓
5. Combined evidence = coordination detection
```

---

*This guide provides a comprehensive explanation of Kleinberg's burst detection algorithm as implemented in the CIB Mango Tree project. For implementation details, see `src/components/kleinberg_utils.py` and `src/components/burst_detector_enhanced.py`.*

**References:**
- Kleinberg, J. (2003). Bursty and hierarchical structure in streams. *Data Mining and Knowledge Discovery*, 7(4), 373-397.
- Project implementation: `src/components/kleinberg_utils.py`

