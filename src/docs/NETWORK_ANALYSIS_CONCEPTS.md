# 🕸️ Network Coordination Analysis - Complete Guide

A comprehensive explanation of all network analysis concepts used in coordination detection.

---

## 📚 Table of Contents

1. [Confidence Scores & Edge Weights](#confidence-scores--edge-weights)
2. [Network Density](#network-density)
3. [Clustering Coefficient](#clustering-coefficient)
4. [Sub-Communities](#sub-communities)
5. [Hub Accounts (Network Centrality)](#hub-accounts-network-centrality)
6. [Most Central Account](#most-central-account)
7. [Coordination Hubs (Retweet Amplification)](#coordination-hubs-retweet-amplification)
8. [Centrality Measures](#centrality-measures)
9. [How Networks Are Built](#how-networks-are-built)
10. [Putting It All Together](#putting-it-all-together)

---

## Confidence Scores & Edge Weights

### **Definition**
Every coordination pair has a **confidence score** (0.0 to 1.0) that quantifies the strength of evidence. When pairs become NetworkX edges, the confidence score becomes the **edge weight**.

### **Confidence Score = Edge Weight**

```python
# Coordination pair
pair = {
    'account1': '@user1',
    'account2': '@user2',
    'type': 'hashtag_coordination',
    'confidence': 0.85  # ← This becomes the edge weight
}

# NetworkX edge
G.add_edge('@user1', '@user2', weight=0.85)
#                                      ↑
#                            Edge weight = confidence
```

### **Confidence Formulas by Evidence Type**

| Evidence Type | Confidence Formula | Range | Example |
|---------------|-------------------|-------|---------|
| **Identical Content** | `1.0` | 1.0 | Two accounts post exact same text → confidence = 1.0 |
| **High Similarity** | `min(similarity × 1.2, 1.0)` | 0.85-1.0 | 88% similar content → min(0.88×1.2, 1.0) = 1.0 |
| **Hashtag Coordination** | `min(jaccard × 1.5, 1.0)` | 0.6-1.0 | 66% hashtag overlap → min(0.66×1.5, 1.0) = 0.99 |
| **URL Coordination** | `min(url_count × 0.8, 1.0)` | 0.8-1.0 | 1 shared URL → 0.8; 2+ shared → 1.0 |
| **Retweet Amplification** | `min(retweeters / 10, 1.0) + boost` | 0.3-1.0 | 6 retweeters → 0.6; +temporal boost up to +0.3 |
| **Temporal Sync** | `(timing_precision + sync_strength) / 2` | 0.8-1.0 | Only ≥80% confidence pairs are kept |

### **Maximum Rule for Multi-Signal Pairs**

When two accounts are connected by **multiple evidence types**, we use **MAXIMUM confidence** (not sum):

```python
# Account pair detected by 3 different signals:
@user1 ↔ @user2

Signal 1: Retweet coordination → confidence = 0.70
Signal 2: Hashtag coordination → confidence = 0.85
Signal 3: URL coordination → confidence = 0.80

# Edge weight = max(0.70, 0.85, 0.80) = 0.85
# NOT sum (0.70 + 0.85 + 0.80 = 2.35 ❌)
```

**Why Maximum?**
- Evidence types are **correlated** (not independent)
- Accounts posting identical content often also share hashtags
- Maximum reflects the **strongest evidence** without inflation
- Prevents over-counting coordinated signals

### **Confidence → Average Network Confidence**

For each network, we calculate the **average confidence** across all coordination pairs:

```python
# Network with 6 pairs:
pair_confidences = [0.85, 0.92, 0.80, 1.0, 0.88, 0.95]

avg_confidence = mean(pair_confidences) = 0.90  # 90% average
```

### **Average Confidence → Risk Level**

Networks are classified by their average confidence:

| Avg Confidence | Risk Level | Interpretation |
|----------------|------------|----------------|
| **> 0.8** | 🔴 **HIGH** | Strong evidence of coordination (likely bot network or organized campaign) |
| **0.6 - 0.8** | 🟡 **MEDIUM** | Moderate evidence (mixed signals, possible coordination) |
| **< 0.6** | 🟢 **LOW** | Weak evidence (may be organic overlap or coincidence) |

**Real Examples:**

```
Network 1: 12 accounts, avg_confidence = 0.95
Risk: HIGH (tight bot cluster posting identical content)

Network 2: 153 accounts, avg_confidence = 0.70
Risk: MEDIUM (large network with moderate retweet coordination)

Network 3: 8 accounts, avg_confidence = 0.55
Risk: LOW (weak hashtag overlap, possibly organic)
```

**Key Insight:** Network size ≠ risk level!
- Small network with high confidence = higher priority than large network with low confidence
- A cluster of 10 accounts posting identical content (confidence = 1.0) is more suspicious than 100 accounts with weak hashtag overlap (confidence = 0.5)

### **Confidence Flows Through Entire Pipeline**

```
1. Detection: Calculate confidence for each pair
   ↓
2. Graph Building: Confidence → Edge weight (max rule for duplicates)
   ↓
3. Network Analysis: Average confidence per network
   ↓
4. Risk Classification: Avg confidence → HIGH/MEDIUM/LOW
   ↓
5. Visualization: Edge thickness scaled by confidence
```

---

## Network Density

### **Definition**
Network density measures **how interconnected the accounts are** in a coordination network.

### **Formula**
```
Density = (Actual Edges) / (Maximum Possible Edges)

Where:
- Actual Edges = Number of coordination relationships that exist
- Maximum Possible Edges = n × (n-1) / 2 (for n accounts)
```

### **Range**
- **0.0 to 1.0**

### **Interpretation**

#### **High Density (0.7 - 1.0)**
- **Meaning:** Most accounts coordinate with most other accounts
- **Pattern:** Tight, fully-connected network
- **Suspicion:** Very high - suggests organized coordination or bot network
- **Example:** 85% of all possible connections exist

**Visual:**
```
A --- B --- C
|  X  |  X  |
D --- E --- F
|  X  |  X  |
G --- H --- I

(Almost everyone connects to everyone)
```

#### **Medium Density (0.3 - 0.7)**
- **Meaning:** Moderate interconnection
- **Pattern:** Organized but not complete
- **Suspicion:** Medium - could be coordinated campaign
- **Example:** 45% of possible connections exist

**Visual:**
```
A --- B --- C
|     |     
D --- E --- F
      |     
G --- H     I

(Some connections, not complete)
```

#### **Low Density (0.0 - 0.3)**
- **Meaning:** Sparse connections
- **Pattern:** Chain-like or hub-and-spoke
- **Suspicion:** Lower - might be organic overlap
- **Example:** Only 5-15% of possible connections exist

**Visual:**
```
A --- B --- C --- D
      |
      E --- F --- G
            |
            H --- I

(Linear chains, few cross-connections)
```

### **Real-World Examples**

**Bot Network:**
- 50 bots, all controlled by same operator
- Density: 0.95 (almost everyone coordinates with everyone)
- **Red flag!**

**Organic Activist Network:**
- 100 activists, coordinating around common cause
- Density: 0.15 (loose coordination, many small groups)
- **Less suspicious**

**Organized Campaign:**
- 75 accounts, coordinated by leaders
- Density: 0.45 (moderate coordination)
- **Investigate further**

### **NetworkX Code**
```python
density = nx.density(subgraph)
```

---

## Clustering Coefficient

### **Definition**
Clustering coefficient measures **how "cliquish" the network is** - do your friends know each other?

### **Formula**
For each account:
```
Local Clustering = (Actual triangles involving the account) / (Possible triangles)

Average Clustering = Mean of all local clustering coefficients
```

### **Range**
- **0.0 to 1.0**

### **Interpretation**

#### **High Clustering (0.7 - 1.0)**
- **Meaning:** Accounts form tight groups where everyone knows everyone
- **Pattern:** Many triangles and cliques
- **Suspicion:** Very high - suggests organized echo chambers
- **Example:** If A coordinates with B and C, then B and C also coordinate

**Visual:**
```
High Clustering (0.9):
    A --- B
    |  X  |    (Triangle: A-B-C all coordinate)
    C --- D
    |  X  |    (Triangle: C-D-E all coordinate)
    E --- F

(Many closed triangles = tight groups)
```

#### **Medium Clustering (0.3 - 0.7)**
- **Meaning:** Some group formation, some loose connections
- **Pattern:** Mix of cliques and chains
- **Suspicion:** Moderate
- **Example:** Some friends know each other, some don't

**Visual:**
```
Medium Clustering (0.5):
    A --- B --- C
    |           |
    D     E --- F

(Some triangles, some not)
```

#### **Low Clustering (0.0 - 0.3)**
- **Meaning:** Star-like pattern, central hub with spokes
- **Pattern:** Hub-and-spoke topology
- **Suspicion:** Lower - could be legitimate influencer
- **Example:** Everyone coordinates with hub, but not with each other

**Visual:**
```
Low Clustering (0.1):
    B     C     D
     \    |    /
      \   |   /
         HUB
      /   |   \
     /    |    \
    E     F     G

(No triangles, all connections go through hub)
```

### **Real-World Examples**

**Tight Coordination Group:**
- Bot network with peer-to-peer coordination
- Clustering: 0.85 (everyone coordinates with everyone in their group)
- **Very suspicious!**

**Influencer Network:**
- Celebrity with many followers retweeting
- Clustering: 0.05 (followers don't coordinate with each other)
- **Less suspicious - organic pattern**

**Campaign with Sub-teams:**
- Different messaging teams that coordinate internally
- Clustering: 0.65 (high within teams, low across teams)
- **Moderately suspicious**

### **Why It Matters**

**High Clustering + High Density = Bot Network**
- Everyone coordinates with everyone
- Tight, organized structure
- Very suspicious

**Low Clustering + Low Density = Influencer**
- Star pattern, one central node
- Followers don't coordinate with each other
- More organic

**High Clustering + Low Density = Sub-Communities**
- Tight groups that don't connect to each other much
- See next section!

### **NetworkX Code**
```python
clustering_coefficient = nx.average_clustering(subgraph)
```

---

## Sub-Communities

### **Definition**
Sub-communities are **smaller groups within a larger network** that are more tightly connected to each other than to the rest of the network.

### **Analogy**
Think of a high school:
- **The network:** The entire school
- **Sub-communities:** Different friend groups (jocks, nerds, artists, etc.)
- **Pattern:** People within each group hang out more with each other than with people from other groups

### **How They're Detected**

#### **Community Detection Algorithms:**

**1. Greedy Modularity Communities** (What we use)
```python
communities = nx.community.greedy_modularity_communities(G)
```

**How it works:**
- Looks for groups where **internal connections are stronger** than external connections
- Maximizes "modularity" - a measure of how well-separated communities are
- Returns list of sub-communities

**Modularity Score:**
- **Range:** -1 to +1
- **>0.4:** Strong community structure - distinct groups exist
- **0.3-0.4:** Moderate community structure
- **<0.3:** Weak community structure - network is well-mixed
- **Negative:** Worse than random (rare)

**2. Label Propagation** (Alternative)
```python
communities = nx.community.label_propagation_communities(G)
```

**How it works:**
- Each account starts with unique label
- Accounts iteratively adopt most common label among neighbors
- When stable, accounts with same label = same community
- Faster but less deterministic

### **Visual Example**

#### **Network WITHOUT Sub-Communities:**
```
Fully mixed - everyone coordinates equally:

A --- B --- C
|  X  |  X  |
D --- E --- F
|  X  |  X  |
G --- H --- I

(One homogeneous network)
```

#### **Network WITH Sub-Communities:**
```
Three distinct groups:

Community 1:          Community 2:          Community 3:
A --- B --- C         D --- E --- F         G --- H --- I
|  X  |  X  |         |  X  |  X  |         |  X  |  X  |
(tight)               (tight)               (tight)

With few connections between communities:
C --- D (weak bridge)
F --- G (weak bridge)
```

### **Indicators of Sub-Communities**

**High Clustering + Low Density:**
- **High clustering:** Tight groups exist
- **Low density:** Groups aren't all connected to each other
- **Conclusion:** Multiple sub-communities present

**Example from your data:**
- Network 1: 153 accounts
- Density: 0.049 (very low)
- Clustering: 0.707 (very high)
- **Interpretation:** Strong sub-community structure!

### **What Sub-Communities Reveal**

#### **In Bot Networks:**
- **Different bot clusters** operated by different controllers
- **Different campaigns** or narratives being pushed
- **Geographic or temporal** divisions in bot activity

**Example:**
```
Bot Network (100 bots):
├─ Sub-Community 1 (30 bots): Pro-Candidate A messaging
├─ Sub-Community 2 (40 bots): Anti-Candidate B messaging
└─ Sub-Community 3 (30 bots): Generic engagement boosting
```

#### **In Organic Networks:**
- Different **interest groups** within a movement
- **Geographic clusters** (local vs national activists)
- **Roles** (content creators vs amplifiers)

**Example:**
```
Activist Network (150 accounts):
├─ Sub-Community 1 (50 accounts): Core organizers (highly coordinated)
├─ Sub-Community 2 (60 accounts): Active supporters (moderate coordination)
└─ Sub-Community 3 (40 accounts): Casual participants (loose coordination)
```

### **Investigation Questions**

When you find sub-communities, ask:

1. **Topic Analysis:**
   - Do different sub-communities post about different topics?
   - Different hashtag patterns per sub-community?

2. **Temporal Patterns:**
   - Do sub-communities activate at different times?
   - Synchronized or independent?

3. **Account Characteristics:**
   - Similar profiles within sub-communities?
   - Different account ages, follower counts?

4. **Evidence Types:**
   - Different coordination methods per sub-community?
   - One uses identical content, another uses retweets?

### **Mathematical Details**

**Modularity Formula:**
```
Q = (1/2m) × Σ[Aij - (kikj/2m)] × δ(ci, cj)

Where:
- m = total edges in network
- Aij = 1 if edge exists between i and j, else 0
- ki, kj = degree of nodes i and j
- δ(ci, cj) = 1 if i and j in same community, else 0
```

**Plain English:**
- For each pair of accounts in the same community
- Check if they're connected MORE than random chance predicts
- Sum across all pairs
- Higher Q = better community division

---

## Hub Accounts (Network Centrality)

### **Definition**
Accounts with the **highest degree centrality** within a coordination network - they have the most coordination connections.

### **How They're Identified**

**Degree Centrality Formula:**
```
Degree Centrality = (Number of connections) / (Maximum possible connections)

Example in a 10-account network:
- Account A connects to 7 accounts → Centrality = 7/9 = 0.78
- Account B connects to 2 accounts → Centrality = 2/9 = 0.22

Account A is a "hub account"
```

**Selection:**
- Sort all accounts by degree centrality
- Top 3 = "Hub Accounts"

### **What They Represent**

- **Most connected accounts** in the network
- **Key coordinators** - they coordinate with many others
- **Network organizers** - likely central to coordination effort
- **Critical nodes** - removing them would fragment the network

### **Visual Example**

```
Hub Account Pattern:

         @hub_account (40 connections)
        /  /  |  |  \  \  \
       /  /   |  |   \  \  \
      A  B    C  D    E  F  G  ... (connects to 40 accounts)
      
Not a hub:
      @peripheral (2 connections)
      |
      X --- Y
```

### **Role in Coordination**

**Hub Accounts are:**
- ✅ **Organizers** - They coordinate with many others
- ✅ **Network architects** - They structure the coordination
- ✅ **Controllers** - Potential bot controllers in malicious networks
- ✅ **Influencers** - They spread coordination patterns

**Hub Accounts are NOT:**
- ❌ Necessarily content creators
- ❌ Always the most amplified accounts
- ❌ Always malicious (could be legitimate organizers)

### **Detection Implications**

**In Enforcement:**
- **Target hub accounts** to disrupt network
- Removing 1-3 hub accounts can collapse entire network
- **Force multiplier** for intervention

**In Attribution:**
- Hub accounts likely **know who's coordinating**
- Links back to potential **organizers or controllers**
- Investigation starting points

### **Example from Your Data**

```
Network 1 (153 accounts):
Hub Accounts:
1. @Ginger182 (top connections)
2. @fray64
3. @MagaconfeA2028

These are the KEY COORDINATORS in the network.
```

### **NetworkX Code**
```python
degree_centrality = nx.degree_centrality(subgraph)
sorted_hubs = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
hub_accounts = [acc for acc, score in sorted_hubs[:3]]
```

---

## Most Central Account

### **Definition**
The **single account** with the **highest degree centrality** in the network.

### **Relationship to Hub Accounts**

```
Hub Accounts = Top 3 by degree centrality
Most Central = Hub Account #1 (the very top one)
```

**It's the SAME data, different presentation:**
- **Most Central** shows the #1 account
- **Hub Accounts** shows top 3 accounts

### **Example**

```
Network 1 (153 accounts):

Most Central Account: @Ginger182 (degree centrality: 0.78)

Hub Accounts:
1. @Ginger182 (0.78) ← Same as "Most Central"
2. @fray64 (0.65)
3. @MagaconfeA2028 (0.52)
```

### **Why Show Both?**

- **Most Central** - Quick reference to the TOP account
- **Hub Accounts** - Shows leadership structure (top 3)

### **What It Tells You**

- The **primary coordinator** in the network
- The account with **most influence** in coordination
- **Single point of failure** - if removed, network might collapse
- Likely the **organizer or controller** of the coordination effort

### **Investigation Priority**

**Most Central Account is:**
- 🎯 **Highest priority** for investigation
- 🎯 **Most likely organizer** or controller
- 🎯 **Best starting point** for attribution
- 🎯 **Maximum impact** if removed

---

## Coordination Hubs (Retweet Amplification)

### **Definition**
Accounts that are **frequently retweeted** by multiple other accounts during bursts. They are **amplification targets**, not necessarily coordinators.

### **How They're Identified**

```
For each account:
1. Count how many DIFFERENT accounts retweeted them
2. Count in how many DIFFERENT bursts they were retweeted
3. If ≥2 different retweeters → Coordination Hub

Example:
@politician was retweeted by:
- @bot1, @bot2, @bot3, @bot4, @bot5 (5 retweeters)
- In burst #1, burst #5, burst #8 (3 bursts)
→ Coordination Hub with confidence 0.5 (5/10 scale)
```

### **What They Represent**

- **Content originators** being amplified
- **Influencers** whose content gets coordinated retweets
- Potential **targets** of amplification campaigns
- **Message sources** being pushed by coordination network

### **KEY DIFFERENCE from Hub Accounts**

| Aspect | Hub Accounts | Coordination Hubs |
|--------|-------------|-------------------|
| **Measured by** | Coordination connections | Being retweeted |
| **Role** | Coordinator WITH others | Amplified BY others |
| **Direction** | Bidirectional coordination | Unidirectional (source→retweeters) |
| **Detection** | NetworkX centrality | Retweet pattern analysis |
| **Indicates** | Who coordinates | Who gets amplified |

### **Visual Comparison**

**Hub Account (Coordinator):**
```
@organizer coordinates WITH:
├─ @supporter1 (similar content)
├─ @supporter2 (same hashtags)
├─ @supporter3 (shared URLs)
└─ @supporter4 (identical posts)

Role: ORGANIZES coordination
```

**Coordination Hub (Amplified):**
```
@politician posts original content
         ↓
    (retweeted by)
         ↓
├─ @bot1
├─ @bot2
├─ @bot3
├─ @bot4
└─ @bot5

Role: TARGET of amplification
```

### **Why BOTH Are Important**

**Hub Accounts answer:**
- ❓ WHO is organizing the coordination?
- ❓ HOW is the network structured?
- ❓ WHERE to intervene to disrupt?

**Coordination Hubs answer:**
- ❓ WHAT content is being amplified?
- ❓ WHO benefits from the amplification?
- ❓ WHAT narratives are being pushed?

### **Combined Analysis**

**Account in BOTH lists:**
```
@example is:
├─ Hub Account (high degree centrality)
└─ Coordination Hub (frequently retweeted)

Interpretation: DUAL ROLE - Both coordinator AND content source
Suspicion: VERY HIGH - central figure in coordination
```

**Account ONLY in Hub Accounts:**
```
@coordinator is Hub Account but NOT Coordination Hub

Interpretation: ORGANIZER role - coordinates but doesn't create viral content
Role: Behind-the-scenes coordinator
```

**Account ONLY in Coordination Hubs:**
```
@celebrity is Coordination Hub but NOT Hub Account

Interpretation: AMPLIFICATION TARGET - being amplified but not coordinating
Could be: Legitimate influencer being exploited
```

### **Example from Your Data**

```
Coordination Hubs (if detected):
1. @some_account - 15 retweeters (HIGH confidence)
2. @another_account - 12 retweeters (MEDIUM confidence)
3. @third_account - 8 retweeters (MEDIUM confidence)

These are WHAT is being amplified by the network.
```

---

## Network Structure Types

### **Overview**
Network structure type describes the **overall topology** of how coordination is organized - whether it's centralized around leaders, distributed among peers, or a mix of both.

### **The Three Structure Types**

#### **1. Hierarchical (Centralized / Hub-and-Spoke)**

**Definition:**
- Network organized around **one or few central hubs**
- Most accounts connect through the hub(s)
- Star-like topology

**Visual:**
```
Hierarchical Structure:

         LEADER
        /  |  |  \
       /   |  |   \
      A    B  C    D
     /|    |  |    |\
    E F    G  H    I J

- One central hub (LEADER)
- Followers connect to hub
- Followers DON'T connect to each other much
```

**Characteristics:**
- **Low clustering coefficient** (0.1-0.3) - Few triangles, mostly star pattern
- **High centralization** - One or few accounts dominate connections
- **Low density** (0.05-0.2) - Not everyone connects to everyone
- **Clear hub account(s)** - High degree centrality concentration

**Indicators:**
- Top hub account has degree centrality **>2x** the second hub
- Removing top hub would **fragment network** significantly
- **Betweenness centrality** concentrated in hub(s)

**Real-World Examples:**
- **Influencer amplification network:** Celebrity with many followers retweeting
- **Bot master controller:** Single operator controlling many bot accounts
- **Campaign with clear leader:** One person organizing, others following

**Suspicion Assessment:**
- **If organic influencer:** Low suspicion (natural pattern)
- **If bot network:** High suspicion (centralized control)
- **Depends on context and other signals**

---

#### **2. Distributed (Peer-to-Peer / Fully Connected)**

**Definition:**
- Network organized as **peer-to-peer** coordination
- Most accounts coordinate with most other accounts
- Mesh-like topology

**Visual:**
```
Distributed Structure:

A --- B --- C --- D
|  X  |  X  |  X  |
E --- F --- G --- H
|  X  |  X  |  X  |
I --- J --- K --- L

- Everyone coordinates with almost everyone
- No single central hub
- Many cross-connections
```

**Characteristics:**
- **High clustering coefficient** (0.7-1.0) - Many triangles and cliques
- **High density** (0.6-1.0) - Most possible connections exist
- **Low centralization** - Connections evenly distributed
- **Multiple hub accounts** with similar centrality scores

**Indicators:**
- Top 5 hub accounts have similar degree centrality (within 20%)
- Many **triangles** (A-B-C all coordinate)
- Removing any single account has minimal impact

**Real-World Examples:**
- **Bot peer network:** Bots programmed to coordinate with each other
- **Tight activist group:** Small coordinated team where everyone knows everyone
- **Echo chamber:** Tight community amplifying each other

**Suspicion Assessment:**
- **Very high suspicion** - Suggests organized, automated coordination
- Unnatural pattern for organic groups (humans don't all coordinate equally)
- Typical of **bot networks** or **coordinated inauthentic behavior**

---

#### **3. Mixed (Hybrid / Clustered)**

**Definition:**
- Network has **both hierarchical and distributed** elements
- Contains **sub-communities** with different internal structures
- Most common in real-world networks

**Visual:**
```
Mixed Structure:

Sub-Community 1:        Bridge        Sub-Community 2:
    LEADER1                               LEADER2
   /   |   \                             /   |   \
  A    B    C --------<connection>------ D    E    F
 /|\  /|\  /|\                          /|\  /|\  /|\
G H I J K L M                          N O P Q R S T

Sub-Community 3:
U --- V --- W
|  X  |  X  |
X --- Y --- Z

- Multiple sub-communities
- Some hierarchical (Leader1, Leader2)
- Some distributed (Sub-Community 3)
- Bridges between communities
```

**Characteristics:**
- **Medium clustering coefficient** (0.4-0.7) - Some cliques, some stars
- **Low to medium density** (0.1-0.5) - Depends on community connectivity
- **Variable centralization** - Different patterns in different parts
- **Clear sub-communities** - Modularity >0.4

**Indicators:**
- **Modularity** >0.4 (strong community structure)
- Different **density within** vs **between** communities
- Some hub accounts **within communities**, some **bridging communities**
- **Betweenness centrality** identifies bridge accounts

**Real-World Examples:**
- **Multi-team campaign:** Different messaging teams, each with leaders
- **Geographic clusters:** Local organizers coordinating regional groups
- **Specialized roles:** Content creators + amplifiers + engagement bots
- **Evolving network:** Organic groups that become more organized

**Suspicion Assessment:**
- **Medium to high suspicion** - Organized coordination with structure
- More sophisticated than simple bot network
- Could be legitimate organized activism OR coordinated campaign
- **Investigate further** - Look at sub-community behaviors

---

### **How to Determine Structure Type**

#### **Method 1: Clustering + Density Analysis (Simple)**

```python
def determine_structure_simple(G):
    """Simple structure type determination"""
    density = nx.density(G)
    clustering = nx.average_clustering(G)
    
    # Hierarchical: Low clustering, low-medium density
    if clustering < 0.4 and density < 0.4:
        return "HIERARCHICAL"
    
    # Distributed: High clustering, high density
    elif clustering >= 0.6 and density >= 0.5:
        return "DISTRIBUTED"
    
    # Mixed: Everything else
    else:
        return "MIXED"
```

**Decision Matrix:**

| Clustering | Density | Structure Type |
|------------|---------|----------------|
| Low (0-0.4) | Low (0-0.3) | **HIERARCHICAL** (Star pattern) |
| Low (0-0.4) | High (0.6-1.0) | **HIERARCHICAL** (Few hubs, many spokes) |
| High (0.6-1.0) | High (0.6-1.0) | **DISTRIBUTED** (Mesh pattern) |
| Medium (0.4-0.6) | Low-Medium (0.1-0.5) | **MIXED** (Sub-communities) |
| High (0.6-1.0) | Low (0-0.3) | **MIXED** (Tight groups, loose links) |

---

#### **Method 2: Hub Concentration Analysis (Better)**

```python
def determine_structure_advanced(G):
    """Advanced structure type with hub analysis"""
    density = nx.density(G)
    clustering = nx.average_clustering(G)
    
    # Calculate degree centrality
    degree_cent = nx.degree_centrality(G)
    sorted_cent = sorted(degree_cent.values(), reverse=True)
    
    if len(sorted_cent) < 2:
        return "INSUFFICIENT_DATA"
    
    # Hub concentration ratio
    top_hub = sorted_cent[0]
    second_hub = sorted_cent[1]
    hub_ratio = top_hub / second_hub if second_hub > 0 else float('inf')
    
    # Average degree centrality
    avg_centrality = sum(sorted_cent) / len(sorted_cent)
    
    # Decision logic
    # HIERARCHICAL: One dominant hub
    if hub_ratio > 2.0 and clustering < 0.5:
        return "HIERARCHICAL"
    
    # DISTRIBUTED: Even distribution, high connectivity
    elif hub_ratio < 1.3 and clustering > 0.6 and density > 0.5:
        return "DISTRIBUTED"
    
    # MIXED: Everything else (default for complex networks)
    else:
        return "MIXED"
```

**Hub Concentration Indicators:**

| Hub Ratio (Top/Second) | Interpretation |
|------------------------|----------------|
| **>3.0** | Very hierarchical - one dominant leader |
| **2.0-3.0** | Hierarchical - clear leader |
| **1.5-2.0** | Moderately hierarchical - some leaders |
| **1.0-1.5** | Distributed - similar influence |

---

#### **Method 3: Modularity + Sub-Communities (Most Accurate)**

```python
def determine_structure_comprehensive(G):
    """Most comprehensive structure analysis"""
    density = nx.density(G)
    clustering = nx.average_clustering(G)
    
    # Detect sub-communities
    communities = list(nx.community.greedy_modularity_communities(G))
    modularity = nx.community.modularity(G, communities)
    num_communities = len(communities)
    
    # Degree centrality analysis
    degree_cent = nx.degree_centrality(G)
    sorted_cent = sorted(degree_cent.values(), reverse=True)
    top_hub = sorted_cent[0] if sorted_cent else 0
    hub_ratio = sorted_cent[0] / sorted_cent[1] if len(sorted_cent) > 1 and sorted_cent[1] > 0 else float('inf')
    
    # Decision logic
    # HIERARCHICAL: Centralized, low modularity
    if hub_ratio > 2.0 and modularity < 0.3 and clustering < 0.5:
        return {
            'type': 'HIERARCHICAL',
            'confidence': 'HIGH',
            'reason': 'Single dominant hub with centralized structure'
        }
    
    # DISTRIBUTED: High connectivity, no communities
    elif density > 0.5 and clustering > 0.6 and modularity < 0.3:
        return {
            'type': 'DISTRIBUTED',
            'confidence': 'HIGH',
            'reason': 'High density and clustering without community structure'
        }
    
    # MIXED: Strong community structure
    elif modularity >= 0.4 or (num_communities > 2 and clustering > 0.5):
        return {
            'type': 'MIXED',
            'confidence': 'HIGH',
            'reason': f'{num_communities} distinct sub-communities detected'
        }
    
    # MIXED (default): Can't clearly classify
    else:
        return {
            'type': 'MIXED',
            'confidence': 'MEDIUM',
            'reason': 'Complex structure with mixed characteristics'
        }
```

**Modularity-Based Classification:**

| Modularity | Communities | Interpretation |
|-----------|-------------|----------------|
| **<0.3** | 1-2 | Homogeneous network (likely hierarchical or distributed) |
| **0.3-0.4** | 2-3 | Moderate community structure (likely mixed) |
| **>0.4** | 3+ | Strong community structure (definitely mixed) |

---

### **Visual Classification Guide**

#### **How to Spot Each Type Visually**

**HIERARCHICAL (Star Pattern):**
```
      HUB
    / | | \
   A  B C  D
  /|  | |  |\
 E F  G H  I J

Look for:
✓ One central node with many connections
✓ Periphery nodes with few connections
✓ Few or no connections between periphery nodes
```

**DISTRIBUTED (Mesh Pattern):**
```
A --- B --- C
|\ X /|\ X /|
| X  |  X  |
D --- E --- F
|\ X /|\ X /|
| X  |  X  |
G --- H --- I

Look for:
✓ Many cross-connections
✓ Every node connects to many others
✓ No single dominant center
```

**MIXED (Clustered Pattern):**
```
Cluster 1:     Bridge      Cluster 2:
  HUB1                        HUB2
 / | \                       / | \
A  B  C ----bridge----- D  E  F

Cluster 3:
G --- H --- I
|\ X /|\ X /|
J --- K --- L

Look for:
✓ Distinct groups/clusters
✓ Different patterns within groups
✓ Bridge connections between groups
```

---

### **Labeling Networks: Complete Example**

```python
def analyze_and_label_network(network_accounts, coordination_pairs):
    """Complete network structure analysis"""
    
    # Build graph
    G = nx.Graph()
    for pair in coordination_pairs:
        G.add_edge(pair['account1'], pair['account2'])
    
    # Calculate metrics
    density = nx.density(G)
    clustering = nx.average_clustering(G)
    
    # Detect communities
    communities = list(nx.community.greedy_modularity_communities(G))
    modularity = nx.community.modularity(G, communities)
    
    # Hub analysis
    degree_cent = nx.degree_centrality(G)
    sorted_cent = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)
    hub_accounts = [acc for acc, _ in sorted_cent[:3]]
    
    # Calculate hub ratio
    if len(sorted_cent) >= 2:
        hub_ratio = sorted_cent[0][1] / sorted_cent[1][1]
    else:
        hub_ratio = 1.0
    
    # Determine structure type
    if hub_ratio > 2.0 and modularity < 0.3 and clustering < 0.5:
        structure_type = "HIERARCHICAL"
        structure_detail = "Centralized around single hub"
    elif density > 0.5 and clustering > 0.6 and modularity < 0.3:
        structure_type = "DISTRIBUTED"
        structure_detail = "Peer-to-peer mesh network"
    elif modularity >= 0.4:
        structure_type = "MIXED"
        structure_detail = f"Multiple sub-communities (n={len(communities)})"
    else:
        structure_type = "MIXED"
        structure_detail = "Complex mixed structure"
    
    return {
        'structure_type': structure_type,
        'structure_detail': structure_detail,
        'metrics': {
            'density': density,
            'clustering': clustering,
            'modularity': modularity,
            'hub_ratio': hub_ratio,
            'num_communities': len(communities)
        },
        'hub_accounts': hub_accounts,
        'communities': [list(c) for c in communities]
    }
```

---

### **Interpretation Guide by Structure Type**

#### **If Network is HIERARCHICAL:**

**What it means:**
- Centralized control or influence
- Single point of failure
- Clear leader(s) directing others

**Investigation priorities:**
1. Focus on the **hub account** (likely organizer/controller)
2. Check if hub is legitimate influencer or bot master
3. Analyze hub's content and behavior patterns
4. Determine if followers are organic or automated

**Enforcement strategy:**
- **High impact:** Removing hub account disrupts entire network
- **Efficient:** Target one account to affect many

---

#### **If Network is DISTRIBUTED:**

**What it means:**
- Peer-to-peer coordination
- Highly organized and synchronized
- No single point of failure

**Investigation priorities:**
1. **Very suspicious** - Unnatural pattern for humans
2. Look for automated/bot behavior across all accounts
3. Check account creation dates (batch created?)
4. Analyze posting patterns (synchronized?)

**Enforcement strategy:**
- **Challenging:** No single hub to target
- **Required:** Broad action against multiple accounts
- **Pattern-based:** Look for common characteristics (IP, device, etc.)

---

#### **If Network is MIXED:**

**What it means:**
- Organized campaign with structure
- Multiple teams or roles
- More sophisticated coordination

**Investigation priorities:**
1. Analyze each **sub-community** separately
2. Identify **bridge accounts** connecting communities
3. Determine roles (content creators, amplifiers, engagers)
4. Look for campaign coordination timeline

**Enforcement strategy:**
- **Strategic:** Target hub accounts in each sub-community
- **Nuanced:** Different sub-communities may need different approaches
- **Bridge accounts:** Removing these fragments the network

---

### **Real-World Examples with Labels**

#### **Example 1: Bot Amplification Network**
```
Metrics:
- Density: 0.08
- Clustering: 0.15
- Hub Ratio: 4.2
- Modularity: 0.22

Structure Type: HIERARCHICAL
Reason: Single hub with 4.2x more connections than second hub
Pattern: @bot_master controlling 50 bot accounts
```

#### **Example 2: Coordinated Troll Network**
```
Metrics:
- Density: 0.78
- Clustering: 0.85
- Hub Ratio: 1.1
- Modularity: 0.18

Structure Type: DISTRIBUTED
Reason: High density and clustering, even distribution
Pattern: 20 accounts all coordinating with each other
```

#### **Example 3: Multi-Team Campaign**
```
Metrics:
- Density: 0.15
- Clustering: 0.68
- Hub Ratio: 1.8
- Modularity: 0.52
- Communities: 4

Structure Type: MIXED
Reason: Strong community structure with 4 sub-communities
Pattern: 150 accounts in 4 teams, each with local coordination
```

#### **Your Network 1 Example:**
```
Metrics:
- Size: 153 accounts
- Density: 0.049 (very low)
- Clustering: 0.707 (very high)
- Modularity: ~0.5 (estimated from density+clustering)

Structure Type: MIXED
Reason: Low density + high clustering = strong sub-communities
Pattern: Multiple tight groups loosely connected
Assessment: Organized campaign with different teams
```

---

## Centrality Measures

### **Overview**
Centrality measures identify **important accounts** in a network based on different criteria.

### **Types of Centrality**

#### **1. Degree Centrality** (What we use for Hub Accounts)

**Definition:**
- Number of direct connections an account has
- Most basic centrality measure

**Formula:**
```
Degree Centrality = (Number of connections) / (Max possible connections)
```

**Identifies:**
- Most connected accounts
- Direct influence spreaders
- Network hubs

**Use case:**
- Finding coordinators
- Identifying key nodes for disruption

---

#### **2. Betweenness Centrality**

**Definition:**
- How often an account lies on the shortest path between other accounts
- Measures "bridging" role

**Formula:**
```
Betweenness = Σ(shortest paths through account) / (total shortest paths)
```

**Identifies:**
- Bridge accounts connecting different communities
- Information bottlenecks
- Accounts that control information flow

**Visual:**
```
Community 1:     Bridge      Community 2:
A --- B --- C ← → X ← → D --- E --- F

X has high betweenness (connects two communities)
```

**Use case:**
- Finding accounts that connect sub-communities
- Identifying key information brokers

---

#### **3. Closeness Centrality**

**Definition:**
- How close an account is to all other accounts
- Average distance to all other nodes

**Formula:**
```
Closeness = 1 / (Average distance to all other accounts)
```

**Identifies:**
- Accounts that can quickly reach others
- Efficient information spreaders

**Use case:**
- Finding accounts best positioned to spread information
- Identifying efficient coordinators

---

#### **4. Eigenvector Centrality**

**Definition:**
- Measures influence based on having connections to other influential accounts
- "Being connected to important people makes you important"

**Analogy:**
- Like Google PageRank
- Not just HOW MANY connections, but WHO you're connected to

**Identifies:**
- Truly influential accounts
- Quality over quantity of connections

**Use case:**
- Finding most influential accounts
- Distinguishing genuine influence from just many connections

---

### **Which Centrality to Use?**

| Goal | Best Centrality Measure |
|------|------------------------|
| Find most connected accounts | **Degree Centrality** |
| Find bridges between groups | **Betweenness Centrality** |
| Find efficient spreaders | **Closeness Centrality** |
| Find most influential | **Eigenvector Centrality** |

**We use Degree Centrality for Hub Accounts because:**
- ✅ Simple and interpretable
- ✅ Directly measures coordination connections
- ✅ Fast to compute
- ✅ Effective for identifying key coordinators

---

## How Networks Are Built

### **The Complete Process**

#### **Step 1: Detect Coordination Evidence (Pairs)**

The `ContentCoordinationDetector` finds pairs of accounts that coordinate:

```
Evidence Examples:
Account A ↔ Account B (posted identical content)
Account B ↔ Account C (used same hashtags)
Account C ↔ Account D (retweeted same source)
Account E ↔ Account F (posted identical content)
Account F ↔ Account G (used same URLs)
```

#### **Step 2: Build NetworkX Graph**

```python
import networkx as nx

G = nx.Graph()

for pair in coordination_pairs:
    account1 = pair['account1']
    account2 = pair['account2']
    confidence = pair['confidence']  # 0.0 to 1.0 confidence score
    evidence_type = pair['type']
    
    # Handle duplicate edges: use MAXIMUM confidence rule
    if G.has_edge(account1, account2):
        # Keep strongest evidence as edge weight
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

**Why Maximum Confidence (Not Sum)?**

Evidence types are often **correlated** (not independent):
- Accounts posting identical content often also share hashtags
- Retweet coordinators may also have temporal synchronization
- Summing would artificially inflate confidence scores

**Maximum reflects the strongest evidence** without double-counting correlated signals.

**Example:**
```python
# Pair detected by 3 signals:
Pair: @user1 ↔ @user2

Evidence 1: Retweet coordination (confidence = 0.70)
Evidence 2: Hashtag coordination (confidence = 0.85)
Evidence 3: URL coordination (confidence = 0.80)

# Edge weight = max(0.70, 0.85, 0.80) = 0.85
# NOT sum (2.35) which would exceed 1.0 and misrepresent evidence strength
```

**Resulting Graph:**
```
Nodes: A, B, C, D, E, F, G
Edges: 
  A-B (weight=1.0, types=['identical_content'])
  B-C (weight=0.85, types=['hashtag_coordination'])
  C-D (weight=0.70, types=['retweet_amplification'])
  E-F (weight=1.0, types=['identical_content'])
  F-G (weight=0.80, types=['url_coordination'])
```

#### **Step 3: Find Connected Components**

NetworkX identifies groups where every account can reach every other account:

```python
networks = list(nx.connected_components(G))
```

**Result:**
```
Network 1: {A, B, C, D}  (all connected through chains)
Network 2: {E, F, G}     (all connected through chains)
```

#### **Step 4: Calculate Network Metrics**

For each network:

```python
# Get all coordination pairs in this network
network_pairs = []
for edge in subgraph.edges(data=True):
    network_pairs.extend(edge[2]['coordination_pairs'])

# Calculate average confidence across all pairs
avg_confidence = np.mean([pair['confidence'] for pair in network_pairs])

# Determine risk level based on confidence
if avg_confidence > 0.8:
    risk_level = 'HIGH'      # >80% confidence - strong evidence
elif avg_confidence > 0.6:
    risk_level = 'MEDIUM'    # 60-80% confidence - moderate evidence
else:
    risk_level = 'LOW'       # <60% confidence - weak evidence

# Calculate NetworkX metrics
density = nx.density(subgraph)
clustering = nx.average_clustering(subgraph)
degree_centrality = nx.degree_centrality(subgraph)

# Detect communities within the network
communities = nx.community.greedy_modularity_communities(subgraph)

network = {
    'network_id': network_id,
    'accounts': list(component),
    'size': len(component),
    'avg_confidence': avg_confidence,
    'risk_level': risk_level,
    'density': density,
    'clustering': clustering,
    'degree_centrality': degree_centrality,
    'communities': communities
}
```

**Understanding Risk Level:**

| Avg Confidence | Risk Level | Interpretation | Example |
|----------------|------------|----------------|---------|
| **> 0.8** | HIGH | Strong evidence of coordination | Network with many identical content pairs or high RT amplification |
| **0.6 - 0.8** | MEDIUM | Moderate evidence | Network with mixed signals (some RT, some hashtags) |
| **< 0.6** | LOW | Weak evidence | Network with mostly low-confidence hashtag overlaps (may be organic) |

**Important:** Network size ≠ risk level!
- Small network (10 accounts) with avg_confidence = 0.95 → **HIGH risk** (tight bot cluster)
- Large network (150 accounts) with avg_confidence = 0.65 → **MEDIUM risk** (broad campaign with mixed signals)

### **Why Accounts Are in Same Network**

**Connected Component Definition:**
- A **maximal set of nodes** where every pair has at least one path connecting them
- No path exists to nodes outside the set

**Transitive Property:**
```
If A coordinates with B
AND B coordinates with C
THEN A and C are in the same network
(Even without direct A-C coordination!)
```

**Example:**
```
Network 1 (153 accounts):
@account1 → @account2 → @account3 → ... → @account153

Even if account1 never coordinated directly with account153,
they're in the same network because there's a CHAIN connecting them.
```

### **Visual Example**

```
Detected Coordination Pairs:
alice ↔ bob
bob ↔ carol  
carol ↔ dave
alice ↔ carol

Build Graph:
    alice --- bob
      |        |
      +---carol---+
           |
          dave

Find Connected Component:
Network 1 = {alice, bob, carol, dave}

All 4 are connected (can trace path between any two)
```

---

## Putting It All Together

### **Complete Analysis Framework**

#### **Step 1: Build the Network**
```
Detect coordination evidence → Build graph → Find networks
```

#### **Step 2: Measure Structure**
```
Calculate:
- Density (how connected)
- Clustering (how cliquish)  
- Structure type (hierarchical/distributed/mixed)
```

#### **Step 3: Find Key Accounts**
```
Identify:
- Hub Accounts (most connected coordinators)
- Most Central (top coordinator)
- Coordination Hubs (amplification targets)
```

#### **Step 4: Detect Sub-Communities**
```
Find:
- Sub-groups within network
- Different teams or campaigns
- Organizational structure
```

#### **Step 5: Interpret Results**
```
Combine all metrics to assess:
- Coordination type (organic vs automated)
- Network structure (centralized vs distributed)
- Risk level (high/medium/low)
- Intervention targets
```

### **Interpretation Guide**

#### **Bot Network Pattern:**
```
✓ High density (0.7+)
✓ High clustering (0.7+)
✓ Distributed structure
✓ Few hub accounts with many connections
✓ Coordination hubs ≠ Hub accounts (separate roles)

Assessment: VERY SUSPICIOUS - Likely bot network
```

#### **Organic Activist Network:**
```
✓ Low density (0.1-0.3)
✓ Medium clustering (0.4-0.6)
✓ Mixed structure with sub-communities
✓ Multiple hub accounts (different leaders)
✓ Coordination hubs overlap with hub accounts

Assessment: LESS SUSPICIOUS - Likely organic coordination
```

#### **Organized Campaign:**
```
✓ Medium density (0.4-0.6)
✓ High clustering (0.6-0.8)
✓ Hierarchical or mixed structure
✓ Clear hub accounts (leaders)
✓ Coordination hubs = specific narratives being pushed

Assessment: MODERATELY SUSPICIOUS - Organized coordination
```

### **Your Network 1 Example**

```
📊 Metrics:
- Size: 153 accounts
- Density: 0.049 (very low)
- Clustering: 0.707 (very high)
- Structure: MIXED

🔍 Interpretation:
- Low density + High clustering = STRONG sub-communities
- Not a monolithic bot network (would have high density)
- Multiple tight groups loosely connected
- Likely organized campaign with different teams

👥 Key Accounts:
Hub Accounts: @Ginger182, @fray64, @MagaconfeA2028
Most Central: @Ginger182
(These are the coordinators)

Coordination Hubs: (If detected)
(These are what's being amplified)

💡 Assessment:
MEDIUM RISK - Organized coordination with sub-teams,
but not typical bot network pattern.
Investigation recommended on hub accounts.
```

---

## 📊 Production System Statistics (Truth Social Dataset)

### **Dataset Overview**
- **Platform**: Truth Social
- **Duration**: 20 days
- **Total posts**: 47,403
- **Unique accounts**: 16,468

### **Phase 4 Production Results**

#### **Coordination Detection:**
- **Total coordination pairs detected**: 1,110
- **Unique accounts involved in coordination**: 211 (1.3% of total accounts)
- **Networks formed**: 20
- **Multi-signal pairs**: 89 (8.0% detected by ≥2 evidence types)

#### **Signal Contributions (Phased Evaluation):**
| Phase | Signals Included | Pairs Detected | Contribution | Cumulative |
|-------|-----------------|----------------|--------------|------------|
| **Phase 1** | Content similarity only | 4 | 0.36% | 4 (0.36%) |
| **Phase 2** | + Hashtag + URL | +11 | 1.0% | 15 (1.4%) |
| **Phase 3** | + Retweet amplification | +1,066 | **96.1%** | 1,081 (97.4%) |
| **Phase 4** | + Temporal sync | +29 | 2.7% | 1,110 (100%) ✅ |
| **Phase 5** | + Behavioral patterns | REJECTED | N/A | N/A |

**Key Finding:** Retweet amplification dominates (96.1% of all coordination detected)

#### **Network Statistics:**

| Network Size | Count | Percentage |
|--------------|-------|------------|
| 2-10 accounts | 13 | 65% |
| 11-50 accounts | 6 | 30% |
| 51+ accounts | 1 | 5% |

**Largest network**: 153 accounts (Network 1)

#### **Risk Level Distribution:**

| Risk Level | Networks | Accounts | Avg Confidence Range |
|------------|----------|----------|---------------------|
| **HIGH** | 5 (25%) | 45 | 0.81 - 0.98 |
| **MEDIUM** | 10 (50%) | 158 | 0.62 - 0.79 |
| **LOW** | 5 (25%) | 8 | 0.52 - 0.59 |

#### **Confidence Score Distribution:**

```
Average confidence across all pairs: 0.72
Median confidence: 0.70

Distribution:
1.0 (perfect): 4 pairs (0.4%) - Identical content
0.8-0.99: 387 pairs (34.9%) - Strong evidence
0.6-0.79: 612 pairs (55.1%) - Medium evidence
< 0.6: 107 pairs (9.6%) - Weak evidence
```

#### **Evidence Type Breakdown:**

| Evidence Type | Pairs | Percentage | Typical Confidence |
|---------------|-------|------------|-------------------|
| Retweet amplification | 1,066 | 96.1% | 0.60-0.95 |
| Temporal synchronization | 29 | 2.6% | 0.80-0.98 |
| Hashtag coordination | 11 | 1.0% | 0.75-0.99 |
| URL coordination | 3 | 0.3% | 0.80 |
| Identical content | 4 | 0.4% | 1.0 |
| High similarity content | 0 | 0% | N/A |

**Note:** Total > 100% because 89 pairs (8.0%) have multiple evidence types

#### **Top Coordination Hubs (Amplification Targets):**

| Account | Retweeters | Bursts | Role |
|---------|-----------|---------|------|
| @maxjett12 | 15+ | 8 | News aggregator/influencer |
| (others anonymized) | 5-12 | 3-6 | Political content sources |

### **Performance Metrics:**

- **Processing time**: ~45 seconds (entire 20-day dataset)
- **Burst detection**: 48 bursts identified (Kleinberg algorithm)
- **Average burst size**: 19.4 active participants (after filtering)
- **Graph building**: 211 nodes, 1,110 edges in NetworkX

### **Detection Quality (Without Ground Truth):**

Since we lack verified coordinated account labels:
- **Validation method**: Multi-signal corroboration + manual inspection
- **Multi-signal pairs**: 89 (8.0%) - detected by 2+ evidence types
- **High-confidence detections**: 391 pairs (35.2%) with confidence ≥ 0.8
- **Manual inspection**: Top 50 high-confidence pairs show plausible coordination patterns

**Conservative by design:** Ultra-conservative thresholds (especially temporal sync ≥80% confidence) minimize false positives at the cost of possible false negatives.

---

## 📚 Quick Reference

### **Metric Cheat Sheet**

| Metric | Range | High Value Means | Low Value Means |
|--------|-------|------------------|-----------------|
| **Density** | 0-1 | Tight coordination | Loose coordination |
| **Clustering** | 0-1 | Cliquish groups | Star pattern |
| **Degree Centrality** | 0-1 | Many connections | Few connections |
| **Modularity** | -1 to 1 | Strong communities | Mixed network |
| **Hub Ratio** | 1.0+ | Centralized | Distributed |

### **Structure Type Determination**

| Clustering | Density | Hub Ratio | Modularity | Structure Type |
|------------|---------|-----------|------------|----------------|
| Low (<0.4) | Low (<0.4) | High (>2.0) | Low (<0.3) | **HIERARCHICAL** |
| High (>0.6) | High (>0.5) | Low (<1.5) | Low (<0.3) | **DISTRIBUTED** |
| High (>0.6) | Low (<0.3) | Any | High (>0.4) | **MIXED** (sub-communities) |
| Medium | Medium | Medium | Medium | **MIXED** (default) |

### **Suspicion Levels**

| Pattern | Suspicion | Likely Explanation |
|---------|-----------|-------------------|
| High density + High clustering | VERY HIGH | Bot network |
| Low density + Low clustering | LOW | Organic influencer |
| Low density + High clustering | MEDIUM | Sub-communities/campaign |
| High density + Low clustering | MEDIUM | Centralized amplification |

### **NetworkX Functions**

```python
# Network metrics
nx.density(G)
nx.average_clustering(G)
nx.connected_components(G)

# Centrality
nx.degree_centrality(G)
nx.betweenness_centrality(G)
nx.closeness_centrality(G)
nx.eigenvector_centrality(G)

# Community detection
nx.community.greedy_modularity_communities(G)
nx.community.label_propagation_communities(G)
```

---

## 🎯 Key Takeaways

1. **Density** = How connected everyone is
2. **Clustering** = How cliquish the groups are
3. **Sub-communities** = Distinct groups within the network
4. **Hub Accounts** = WHO coordinates (most connected)
5. **Coordination Hubs** = WHAT gets amplified (most retweeted)
6. **Most Central** = #1 Hub Account (top coordinator)
7. **Networks are built** from coordination evidence chains
8. **All metrics together** provide complete coordination picture

---

## 📖 Additional Resources

- **NetworkX Documentation:** https://networkx.org/
- **Network Analysis Tutorial:** https://ericmjl.github.io/Network-Analysis-Made-Simple/
- **Community Detection Review:** https://arxiv.org/abs/2408.01257
- **Social Network Analysis:** Standard methods for analyzing coordination patterns

---

*This guide synthesizes all network analysis concepts used in coordination detection. For specific implementation details, see the codebase in `src/components/content_coordination_detector.py`.*

