## Abstract

Coordinated inauthentic behavior (CIB) on social media is typically detected using multi-signal systems that combine content similarity, hashtag coordination, URL sharing, retweet amplification, and behavioral patterns. Although widely adopted, no prior work has systematically quantified the marginal contribution of each signal under controlled conditions.

We introduce a phased additive evaluation framework that isolates the detection value of individual coordination signals while holding all other variables constant. Applied to 47,403 Truth Social posts spanning 20 days in early 2025, our experiments reveal a dramatic imbalance: retweet amplification alone accounts for 96.1% of all detected coordination pairs (1,067 of 1,110), while content similarity (0.4%), hashtag coordination (0.6%), URL sharing (0.3%), and ultra-conservative temporal synchronization (2.6%) contribute only 3.9% combined. Adding widely used behavioral patterns (mechanical posting intervals and activity fingerprints) more than doubles detections but introduces an estimated 50–70% false positive rate, leading to their principled rejection.

Network analysis of the resulting 20 coordination networks shows that half exhibit mixed hierarchical-distributed topologies with organized sub-communities, challenging assumptions of homogeneous bot structures. Our findings imply that detection systems should treat retweet amplification as non-negotiable and deprioritize or exclude low-value or high-noise signals. The phased evaluation methodology is general and directly applicable to any multi-signal detection or classification task.

Keywords: coordinated inauthentic behavior, retweet amplification, signal evaluation, social media manipulation, Truth Social

## 1. Introduction

Social media platforms have become primary channels for public discourse, political mobilization, and information dissemination. However, this democratization of communication has been accompanied by systematic manipulation through coordinated inauthentic behavior (CIB)—organized campaigns employing networks of accounts to artificially amplify narratives, suppress dissent, or create false consensus. Recent surveys highlight the pervasiveness of such coordination across platforms, spanning both legitimate social movements and disinformation campaigns [Mannocci et al., 2024]. The scale and sophistication of coordinated operations pose fundamental challenges to platform integrity and democratic processes.

Detecting coordinated behavior requires distinguishing genuine grassroots activity from orchestrated campaigns—a problem complicated by three factors. First, coordination manifests through *heterogeneous signals*: identical content, synchronized timing, shared hashtags, URL distribution, and amplification networks. Second, individual signals produce high false positive rates when analyzed in isolation; for instance, users in the same timezone naturally exhibit similar posting patterns, and legitimate activists coordinate hashtag campaigns. Third, the *relative importance* of different signals remains empirically unvalidated—existing systems combine multiple features without quantifying marginal contributions.

Prior work on coordination detection falls into three categories. *Supervised classification approaches* train on labeled bot accounts using profile features and activity patterns [Varol et al., 2017; Cresci et al., 2017; Yang et al., 2020], but require platform-specific training data and fail to generalize across domains. *Content-based methods* detect coordination through text similarity or narrative alignment [Pacheco et al., 2020], missing coordination that manifests through amplification rather than duplication. *Network analysis approaches* identify communities via graph clustering [Ferrara et al., 2016], often applied post-hoc without systematic evaluation of which network properties indicate coordination. Critically, **no prior work systematically decomposes and quantifies the contribution of individual coordination signals** in a controlled experimental framework. Systems papers typically include all available features without isolating their marginal value, making it impossible to determine which signals are essential versus supplementary.

**Our Contribution.** We address this gap through a phased additive evaluation methodology that isolates the marginal contribution of each coordination signal. We design a five-phase framework where each phase introduces exactly one signal type while holding all other parameters constant—enabling causal attribution of detection improvements to specific signals. Surprisingly, we find that **retweet amplification alone accounts for more than 96% of all detected coordination** on Truth Social, while content similarity, hashtag coordination, URL sharing, and temporal synchronization contribute only 4% combined. This extreme signal imbalance has direct implications for resource allocation in coordination detection systems: efforts should prioritize retweet analysis over comprehensive multi-signal feature engineering.

Beyond quantifying signal contributions, we characterize coordination network structures using graph-theoretic analysis, revealing that 50% of detected networks exhibit mixed hierarchical-distributed topology rather than homogeneous bot network patterns. We also systematically evaluate and reject behavioral pattern detection—demonstrating that mechanical posting intervals and identical activity fingerprints produce unacceptable false positive rates by conflating coordination with timezone coincidence and legitimate scheduling tools.

**Scope and Limitations.** Our evaluation uses Truth Social data (20 days, 47,403 posts from 16,468 accounts), raising generalizability questions regarding other platforms, languages, and coordination tactics. We lack ground-truth labels for coordinated accounts, preventing traditional precision/recall calculation; instead, we employ confidence scoring and multi-signal corroboration. Temporal analysis is limited by UTC timestamp normalization without original timezone metadata. Despite these limitations, our phased methodology provides the first systematic quantification of coordination signal contributions.

**Paper Organization.** Section 2 reviews related work. Section 3 describes our methodology: burst detection with adaptive selection (3.2), temporal clustering (3.3), multi-signal coordination detection (3.4), network analysis (3.5), and phased validation framework (3.7). Section 4 presents system architecture. Section 5 reports experimental results. Section 6 discusses findings and limitations. Section 7 concludes. Appendices provide algorithmic details (A), formal network metric definitions (B), implementation specifications (C), and supplementary experimental results (D).

## 2. Related Work

We organize related work around three research questions our phased evaluation addresses: (i) Which coordination signals provide detection value? (ii) How should coordination networks be characterized? (iii) When should signals be rejected?

### 2.1 Coordination Detection Signals

**Content-based approaches.** Pacheco et al. [2020] detect coordinated groups through retweet amplification, identifying accounts that frequently retweet the same sources within temporal windows. Their analysis of White Helmets disinformation demonstrates that retweet networks reveal coordination structures. However, Pacheco et al. evaluate only retweet signals—they do not test whether adding content similarity, hashtag coordination, or temporal synchronization improves detection. Our Phase 3 introduces retweet amplification detection and quantifies its contribution relative to other signals, validating Pacheco et al.'s retweet focus while revealing that multi-signal systems investing resources in content analysis gain minimal marginal value.

**Supervised classification.** Bot detection work employs supervised classifiers combining profile features (follower counts, account age), activity patterns (tweet frequency, timing), and content features (linguistic style). Varol et al. [2017] achieve strong performance using random forests on labeled Twitter bot datasets. Cresci et al. [2017] show that simple feature-based classifiers fail against sophisticated "social spambots" that mimic human behavior, advocating for temporal and network features. Yang et al. [2020] improve generalization through strategic training data selection.

These approaches differ fundamentally from ours in three ways. First, they require labeled training data (bot vs. human), unavailable for emerging platforms and evolving adversaries. Second, they produce binary classifications rather than coordination networks showing *how* accounts work together. Third, the relative contribution of individual features remains unquantified—classifiers use all available features simultaneously, making it difficult to determine whether performance derives primarily from profile features, activity patterns, or content analysis. Our phased evaluation directly measures each signal's contribution, demonstrating that retweet detection provides the vast majority of detection value while other signals contribute marginally.

**Temporal approaches.** Kleinberg [2003] introduced burst detection for identifying periods of elevated activity in document streams, modeling arrival rates as a two-state automaton where state transitions incur costs proportional to change magnitude. While applied to social media for hashtag emergence and event detection, these applications focus on aggregate-level bursts rather than identifying *which accounts* participate in coordinated bursts. We extend Kleinberg's algorithm with adaptive contributor selection (Section 3.2) that filters burst participants, focusing on coordinated actors while reducing noise from casual participants.

**Gap in prior work.** Existing coordination detection combines multiple signals—content similarity, temporal patterns, network features, behavioral fingerprints—without systematic evaluation of individual contributions [Mannocci et al., 2024]. This creates two problems: (i) unclear which signals are essential versus supplementary, making resource allocation difficult; (ii) implicit assumption that diverse features provide comparable value, potentially investing effort in low-value signals. Our phased validation (Section 5) fills this gap by isolating each signal's marginal contribution under controlled conditions.

### 2.2 Network Structure Characterization

**Bot network analysis.** Ferrara et al. [2016] survey bot detection methods emphasizing network analysis, arguing that bots exhibit distinctive connectivity patterns (high out-degree, low clustering coefficient, hub-and-spoke topology). However, their analysis focuses on bot *identification* rather than coordination *detection*—coordinated human accounts may exhibit bot-like patterns, and coordinated networks may not follow simple bot topologies.

**Network topology.** Our analysis employs standard graph-theoretic metrics implemented in NetworkX [Hagberg et al., 2008]: density (fraction of possible edges present), clustering coefficient (tendency to form triangles), degree centrality (number of direct connections), and modularity for community detection [Newman, 2006]. We classify networks as hierarchical (centralized hub-and-spoke), distributed (peer-to-peer mesh), or mixed (hybrid with sub-communities) based on the combination of density and clustering coefficient patterns.

**Structure classification.** Prior work often implicitly assumes coordination manifests through homogeneous structures—typically hierarchical bot networks or distributed peer-to-peer coordination. Our results (Section 5.5) reveal greater structural diversity: half of detected networks exhibit mixed topology (neither pure hierarchical nor distributed), indicating organized sub-communities rather than monolithic structure. This pattern suggests multiple tightly coordinated groups with sparse connections between groups—a sophisticated organizational structure that challenges detection systems modeling coordination as homogeneous patterns.

**Distinction: Hub accounts vs. coordination hubs.** A critical but often conflated distinction exists between: (i) *hub accounts*—accounts with highest degree centrality within coordination networks, representing coordinators who work *with* many others; (ii) *coordination hubs*—accounts being systematically retweeted *by* coordinated networks, representing amplification targets rather than coordinators. Prior work discusses network hubs but does not systematically distinguish these roles. Our analysis (Section 5.5) separates them: hub accounts answer *who organizes coordination*, coordination hubs answer *what content gets amplified*.

### 2.3 Signal Rejection and False Positives

**Behavioral pattern analysis.** Many coordination detection systems incorporate behavioral signals: mechanical posting intervals (accounts posting at regular intervals suggesting automation), identical activity patterns (accounts with similar hourly posting distributions suggesting coordination or bots). These features have intuitive appeal—automated accounts should exhibit mechanical timing, coordinated accounts should post at similar times.

**The false positive problem.** However, behavioral patterns suffer from a fundamental limitation: difficulty distinguishing coordination from coincidence. Accounts in the same timezone naturally post at similar hours (lunch break, evening). Social media managers using scheduling tools exhibit mechanical posting intervals legitimately. Friends with similar work schedules show identical activity patterns organically. Without ground truth labels, there is limited ability to separate coordinated bots from normal users exhibiting similar patterns for benign reasons.

**Gap in prior work: Limited rejection analysis.** Multi-signal systems typically include all available features, with limited systematic evaluation of when signals should be excluded. Mannocci et al.'s [2024] survey notes that "most systems combine multiple signals but do not evaluate their individual contributions"—but also does not address when signals should be *rejected* due to high false positive rates. Feature selection typically focuses on inclusion criteria rather than principled exclusion.

Our Phase 5 analysis (Section 5.7) provides systematic evaluation of behavioral patterns. We show that adding mechanical posting intervals and identical activity patterns more than doubles detection counts at unacceptable false positive cost, primarily because UTC normalization without original timezone metadata prevents attribution of patterns to coordination versus geography. We provide quantitative justification for excluding behavioral signals despite their intuitive appeal—principled signal rejection based on false positive analysis.

### 2.4 Coordination Detection Frameworks

Mannocci et al. [2024] provide the most comprehensive recent survey, categorizing coordination detection methods into content-based, network-based, and temporal approaches. They identify key challenges: (i) distinguishing coordination from organic similarity; (ii) handling multi-platform coordination; (iii) balancing false positives versus false negatives; (iv) adapting to adversarial evasion. The survey emphasizes that "most systems combine multiple signals but do not evaluate their individual contributions," directly highlighting the gap our work addresses.

The Digital Observatory's Coordination Network Toolkit [Digital Observatory, 2024] provides practitioner-oriented tools for detecting coordinated link sharing and hashtag campaigns, focusing on rapid triage for investigative journalists. Our work complements this by providing empirical validation of which signals provide the most detection value, enabling more efficient resource allocation.

### 2.5 Summary: Gaps Addressed

Our work addresses three gaps in prior coordination detection research:

- **No systematic signal quantification.** Prior work combines multiple signals without isolating individual contributions. Our phased evaluation measures each signal's marginal value, revealing that retweet amplification dominates virtually all other signals combined.

- **No principled signal rejection.** Existing systems focus on feature inclusion without systematic evaluation of when features should be excluded. Our analysis demonstrates that behavioral patterns produce unacceptable false positive rates, providing quantitative justification for exclusion.

- **Limited network structure interpretation.** While network metrics are well-established, their coordination-specific interpretation remains underexplored. Our finding that half of networks exhibit mixed topology challenges assumptions about homogeneous coordination structures.

## 3. Methodology

### 3.1 Problem Formulation

We formalize coordinated inauthentic behavior detection as a graph construction problem where nodes represent social media accounts and edges represent detected coordination relationships.

**Definition 1 (Coordination Evidence).** Let $E = \{e_1, e_2, \ldots, e_n\}$ denote a set of evidence types. Each evidence type $e_i$ represents a detectable coordination signal: content similarity ($e_{\text{content}}$), hashtag coordination ($e_{\text{hashtag}}$), URL sharing ($e_{\text{url}}$), retweet amplification ($e_{\text{rt}}$), and temporal synchronization ($e_{\text{temporal}}$).

**Definition 2 (Coordination Pair).** A coordination pair $(a_i, a_j, c, E')$ consists of two accounts $a_i, a_j$, confidence score $c \in [0,1]$, and evidence set $E' \subseteq E$ where $|E'| \geq 1$.

**Definition 3 (Coordination Network).** A coordination network $G = (V, E)$ is an undirected graph where $V$ represents accounts and $E$ represents coordination relationships. Each edge $e \in E$ has weight $w(e)$ representing the maximum confidence score among all evidence types connecting the two accounts.

**Definition 4 (Temporal Burst).** A temporal burst $B = (t_{\text{start}}, t_{\text{end}}, s)$ represents a time interval $[t_{\text{start}}, t_{\text{end}}]$ where posting activity significantly exceeds baseline levels, with state $s \in \{0, 1, 2, \ldots\}$ indicating intensity.

Our detection pipeline operates in four stages: (1) burst detection identifies *when* coordinated activity occurs (Section 3.2); (2) multi-signal coordination detection identifies *which* accounts coordinate (Section 3.3); (3) network analysis reveals *how* coordination is organized (Section 3.4). We also evaluate temporal clustering as an alternative methodology (Section 3.5), but find it unsuitable for production coordination detection due to fundamental limitations in distinguishing coordination from timezone coincidence.

### 3.2 Production Methodology: Burst-Based Coordination Detection

Our production system uses burst detection to identify temporal windows of elevated activity, then applies multi-signal analysis to detect coordination within these windows.

#### 3.2.1 Burst Detection with Kleinberg's Algorithm

We employ Kleinberg's [2003] burst detection algorithm to identify periods of elevated posting activity. The algorithm models document arrival as an infinite-state automaton where each state $q \in \{0, 1, 2, \ldots\}$ corresponds to an emission rate $r_q = s^q \cdot r_0$, with baseline rate $r_0$ and scaling factor $s > 1$. State transitions incur costs proportional to change magnitude, and dynamic programming finds the optimal state sequence minimizing total cost (see Appendix A.1 for mathematical formulation).

**Parameter selection.** We use $s = 2.0$ (detecting activity doubling) and $\gamma = 1.0$ (balanced sensitivity). These values identify bursts where posting rates increase by at least 2× over baseline while avoiding excessive fragmentation from minor fluctuations.

#### 3.2.2 Adaptive Contributor Selection

Standard burst detection identifies all accounts posting during burst intervals. We enhance this through threshold-based filtering that focuses on accounts with sustained burst participation. Accounts are selected if they meet any of: (i) posting frequency ≥ max(3, 2% of total burst activity); (ii) membership in top 85% by posting frequency. This filtering reduces noise from casual participants while retaining accounts exhibiting sustained burst participation (detailed algorithm in Appendix A.2).

### 3.3 Multi-Signal Coordination Detection

We detect coordination through five evidence types, each with specific thresholds validated to minimize false positives while maintaining detection coverage. Table 1 summarizes all signals, thresholds, and confidence scoring.

**Table 1: Coordination Detection Signals**

| Signal | Detection Criteria | Confidence Score | Rationale |
|--------|-------------------|------------------|-----------|
| **Content Similarity** | SequenceMatcher ratio ≥0.95 (identical) or ≥0.85 (high similarity) | 1.0 for identical; sim × 1.2 for high similarity | Copy-paste coordination leaves textual fingerprints |
| **Hashtag Coordination** | Jaccard similarity ≥0.6 with ≥2 shared hashtags | min(Jaccard × 1.5, 1.0) | Hashtags are deliberately chosen campaign identifiers |
| **URL Coordination** | ≥1 shared URL within same burst | min(shared_count × 0.8, 1.0) | Coordinated link distribution even with single URL |
| **Retweet Amplification** | ≥3 accounts retweeting same source | min(retweeter_count / 10, 1.0) | Creates C(n,2) pairs reflecting amplification structure |
| **Temporal Sync** | ≤30 second window, ≥3 synchronized posts, ≥80% confidence | (timing_precision + sync_strength) / 2 | Ultra-conservative to avoid timezone false positives |

**Content similarity** uses Python's SequenceMatcher to compute character-level similarity between posts within the same burst. Content is normalized via lowercase conversion and whitespace collapsing.

**Hashtag and URL coordination** use set-based similarity metrics. Hashtags are extracted via regex and normalized to lowercase; URLs are normalized by stripping tracking parameters.

**Retweet amplification** identifies accounts retweeting the same source. If $n$ accounts retweet source $s$, we create $\binom{n}{2}$ coordination pairs, reflecting the amplification network structure. The minimum threshold of 3 retweeters balances sensitivity against specificity.

**Temporal synchronization** uses ultra-conservative thresholds (30-second windows requiring multiple instances) to ensure only accounts with suspiciously precise timing are flagged. This addresses the high false positive risk from same-timezone users naturally posting at similar times.

**Confidence scoring.** Multi-signal confidence is computed as the maximum across all detected evidence types:

$$c_{\text{total}}(a_i, a_j) = \max_{e \in E'} c_e(a_i, a_j)$$

We use maximum rather than sum because evidence types are not independent—accounts coordinating through content similarity may also coordinate through hashtags. Maximum confidence reflects the strongest evidence, avoiding inflation from correlated signals. Pairs with multiple evidence types ($|E'| \geq 2$) have substantially higher detection confidence.

### 3.4 Network Analysis with NetworkX

From coordination pairs, we construct an undirected graph $G = (V, E)$ where vertices represent accounts and edges represent coordination relationships. Edge weights equal the maximum confidence across all evidence types connecting two accounts.

**Network identification.** Coordination networks are identified as connected components—maximal sets of accounts where every pair has at least one path connecting them.

**Network topology.** Our analysis employs standard graph-theoretic metrics using NetworkX [Hagberg et al., 2008]: *density* (fraction of possible edges present), *clustering coefficient* (tendency to form triangles), *degree centrality* (number of direct connections), and *modularity* for community detection [Newman, 2006]. Formal definitions of all metrics appear in Appendix B.

**Structure classification.** Networks are classified based on density and clustering patterns:
- **Hierarchical (hub-and-spoke):** Low clustering and centralized degree distribution
- **Distributed (peer-to-peer):** High clustering and high density
- **Mixed (sub-communities):** High clustering with low density, indicating multiple tight groups with sparse inter-group connections

We use maximum rather than sum because evidence types are not independent—accounts coordinating through content similarity may also coordinate through hashtags. Maximum confidence reflects the strongest evidence, avoiding inflation from correlated signals. Pairs with multiple evidence types ($|E'| \geq 2$) have substantially higher detection confidence.

### 3.5 Alternative Methodology: Temporal Clustering (Experimental)

We evaluate temporal clustering as an alternative approach to coordination detection, grouping accounts by posting time patterns. While this approach successfully identifies behavioral clusters, we find it unsuitable for production coordination detection due to fundamental limitations.

#### 3.5.1 Feature Engineering

We construct two feature representations. *2D features* (coarse-grained) extract mean posting hour and weekend activity ratio for each account. *24D features* (fine-grained) construct hourly activity distribution vectors $\mathbf{v}(a) \in \mathbb{R}^{24}$ where:

$$v_i(a) = \frac{|\{p \in P_a : \text{hour}(p) = i\}|}{|P_a|}$$

This represents the probability distribution over hours of day for account $a$.

#### 3.5.2 Clustering Algorithm

We apply K-means clustering with StandardScaler normalization (zero mean, unit variance). Optimal cluster count $k$ is determined via the elbow method: plot within-cluster sum of squares (inertia) versus $k$, select the value where marginal inertia reduction diminishes (the "elbow" point). This visual approach identifies natural groupings without requiring ground-truth labels.

#### 3.5.3 Limitations: Why Not Used for Coordination Detection

Temporal clustering has three critical limitations that prevent its use for coordination detection:

**1. Cannot distinguish coordination from coincidence.** Accounts posting at similar times could be: (i) coordinated bots; (ii) users in the same timezone; (iii) people with similar work schedules; (iv) friends with similar routines. The method cannot differentiate these cases.

**2. UTC normalization without timezone metadata.** All timestamps are UTC-normalized without original timezone information. Two accounts posting at "2pm UTC" might be: (i) coordinated bots in the same timezone; (ii) unrelated users in London (2pm local) and New York (9am local). Without timezone data, this ambiguity is unresolvable.

**3. High false positive risk.** Behavioral pattern detection conflates coordination with legitimate patterns (scheduling tools, work hours, timezone coincidence). Our experiments (Section 5.7) show this produces 50–70% estimated false positive rates.

**Use cases.** Despite these limitations for coordination detection, temporal clustering is useful for: (i) characterizing behavioral diversity in datasets; (ii) post-hoc investigation of detected networks; (iii) exploratory data analysis.

### 3.6 Phased Validation Framework

To isolate individual signal contributions, we design a five-phase additive evaluation where each phase introduces exactly one signal type while holding all other parameters constant.

**Controlled variables (constant across phases):** Dataset (47,403 posts, 76 bursts), burst detection parameters ($s=2.0, \gamma=1.0$), contributor filtering criteria, similarity thresholds, and pair construction method.

**Phase definitions:**
- **Phase 1 (baseline):** Content similarity only
- **Phase 2:** Phase 1 + hashtag and URL coordination
- **Phase 3:** Phase 2 + retweet amplification
- **Phase 4 (production):** Phase 3 + ultra-conservative temporal synchronization
- **Phase 5 (rejected):** Phase 4 + behavioral patterns (mechanical intervals, activity fingerprints)

**Marginal contribution.** The marginal contribution of signal $e_i$ introduced in Phase $k$ is:

$$\Delta_k = \frac{\text{pairs}_k - \text{pairs}_{k-1}}{\text{pairs}_{k-1}} \times 100\%$$

By introducing one signal per phase while holding all else constant, we enable causal attribution: observed detection changes are attributable to the introduced signal rather than confounding factors.

### 3.7 Implementation

Our system is implemented in Python using Polars for high-performance dataframe operations, NetworkX for graph analysis, and scikit-learn for clustering. Complete source code, documentation, and library versions are available in the supplementary materials for reproducibility (see Appendix C for implementation details).

## 4. System Architecture

Our coordination detection system comprises five sequential stages that transform raw social media data into annotated coordination networks (Figure 1).

**Pipeline overview.** The production pipeline comprises four stages: Stage 1 (Data Analysis) loads and preprocesses posts with UTC timestamp normalization. Stage 2 (Burst Detection) applies Kleinberg's algorithm to identify 76 temporal bursts from 480 hours of activity. Stage 3 (Coordination Detection) applies five detection methods (Table 1) to identify coordination pairs within bursts. Stage 4 (Network Analysis) constructs graphs via NetworkX, identifies connected components, and classifies network structures. Temporal clustering (Section 3.5) is evaluated as an alternative methodology but not included in the production pipeline.

**Figure 1: Production Detection Pipeline**
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    Data     │───▶│    Burst    │───▶│Coordination │───▶│   Network   │
│  Analysis   │    │  Detection  │    │  Detection  │    │  Analysis   │
│             │    │             │    │             │    │             │
│ 47,403 posts│    │  76 bursts  │    │ 1,110 pairs │    │ 20 networks │
│ 16,468 accts│    │ 1,477 contribs│  │ 211 accounts│    │ 3 structures│
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘

Alternative: Temporal Clustering (Experimental, Section 3.5)
└─▶ 847 accounts → 4 behavioral clusters (not used for coordination detection)
```

**Computational performance.** Total pipeline execution takes approximately 2-3 minutes on consumer hardware (single-threaded) for the 47k post dataset. The dominant cost is pairwise comparison within bursts, which scales $O(b \cdot c^2)$ where $b$ is burst count and $c$ is contributors per burst. Adaptive contributor selection (Section 3.2.2) reduces average burst size, making $O(n^2)$ comparison tractable.

**Limitations.** Three architectural constraints affect generalizability: (1) UTC normalization without original timezone metadata prevents cross-timezone coordination detection; (2) memory-resident NetworkX graphs limit scalability to networks under millions of nodes; (3) single-platform design requires adaptation for cross-platform analysis.

**Reproducibility.** Complete source code, documentation, and experimental scripts are available as open-source software. Full implementation details including library versions, caching mechanisms, validation procedures, and extensibility patterns appear in Appendix C.

## 5. Experimental Results

### 5.1 Overview of Results

Our phased evaluation reveals a striking signal hierarchy in coordination detection. We summarize the key findings before presenting detailed analysis:

**Signal contributions.** Retweet amplification alone detects 96.1% of all coordination pairs (1,067 of 1,110), while content similarity contributes 0.4% (4 pairs), hashtag coordination 0.6% (7 pairs), URL coordination 0.3% (3 pairs), and temporal synchronization 2.6% (29 pairs). This 25:1 ratio between retweet detection and all other signals combined contradicts the implicit assumption in multi-signal systems that diverse features provide comparable value.

**Phased progression.** Detection coverage increases dramatically with retweet amplification:
- Phase 1 (content only): 4 pairs (0.36% of final coverage)
- Phase 2 (+ hashtags/URLs): 14 pairs (+250%, 1.26% coverage)
- Phase 3 (+ retweets): 1,081 pairs (+7,621%, 97.4% coverage)
- Phase 4 (+ temporal): 1,110 pairs (+2.7%, production configuration)
- Phase 5 (+ behavioral): 2,275 pairs (+105%, **rejected** due to 50–70% estimated false positive rate)

**Network structures.** Analysis of 1,110 coordination pairs reveals 20 distinct networks comprising 211 accounts. Contrary to assumptions of homogeneous bot network topology, 50% exhibit mixed hierarchical-distributed structure, 30% are hierarchical (hub-and-spoke), and 20% are distributed (peer-to-peer). The largest network (153 accounts) shows low density (0.049) but high clustering (0.707), indicating organized sub-communities rather than monolithic coordination.

**Key implication.** Retweet amplification is non-negotiable for coordination detection; systems omitting this signal miss 96% of detectable coordination. Conversely, behavioral pattern detection should be excluded despite intuitive appeal—it doubles detection counts while introducing unacceptable false positive rates.

### 5.2 Dataset and Experimental Setup

**Dataset.** We analyze 47,403 posts from 16,468 unique accounts on Truth Social, spanning January 27 to February 16, 2025 (20 days, 480 hours). Posts include original content, retweets (prefixed "RT @username"), replies, and quote posts. Figure 2 shows the hourly posting activity over the observation period.

![Figure 2: Hourly posting activity](plots/hourly_posts.png)
*Figure 2: Posts per hour on Truth Social over the 20-day observation period (January 27 – February 16, 2025). Posting rates exhibit substantial variability across the time period.*

**Burst detection.** Kleinberg's algorithm (s=2.0, γ=1.0) identifies 76 temporal bursts across the 480-hour observation period, shown in Figure 3. Bursts contain 1,477 unique contributors across all intervals. Adaptive contributor selection filters casual participants to focus on accounts with sustained burst participation (detailed burst statistics in Appendix D.1).

![Figure 3: Burst detection results](plots/burst_rectangles.png)
*Figure 3: Temporal burst detection using Kleinberg's algorithm. Yellow/orange regions indicate detected bursts at different intensity levels. The algorithm identifies 76 bursts where posting activity significantly exceeds baseline rates.*

**Experimental protocol.** Each phase introduces exactly one signal type while holding all other parameters constant: same dataset, burst definitions, similarity thresholds, and pair construction method. This controlled design enables causal attribution of detection changes to specific signals.

**Evaluation metrics.** Without ground-truth labels for coordinated accounts, we report: (1) coordination pairs detected; (2) unique accounts involved; (3) networks formed; (4) evidence type distribution; (5) network structural metrics. We validate detection quality through multi-signal corroboration (pairs detected by ≥2 signals) and manual inspection of high-confidence detections.

### 5.3 Phased Validation Results

Table 2 presents phase-by-phase results with marginal contribution analysis.

**Table 2: Phased Validation Results**

| Phase | Signals Enabled | Pairs | Δ from Previous | Accounts | Networks |
|-------|-----------------|-------|-----------------|----------|----------|
| 1 | Content similarity | 4 | — | 8 | 4 |
| 2 | + Hashtags, URLs | 14 | +250% | 24 | 8 |
| 3 | + Retweet amplification | 1,081 | +7,621% | 198 | 18 |
| 4 | + Temporal sync | 1,110 | +2.7% | 211 | 20 |
| 5 | + Behavioral patterns | 2,275 | +105% | 847 | 35 |

**Phase 1: Content similarity baseline.** Content-only detection identifies 4 coordination pairs across 8 accounts, forming 4 small networks (2 accounts each). These represent accounts posting identical or near-identical content (≥95% similarity) within the same burst. While high-confidence detections, content similarity alone captures only 0.36% of coordination visible in Phase 4.

**Phase 2: Campaign pattern detection.** Adding hashtag coordination (Jaccard ≥0.6) and URL sharing detects 10 additional pairs (+250%), reaching 14 total pairs across 24 accounts. Hashtag coordination contributes 7 pairs; URL coordination contributes 3 pairs. These signals capture campaign-style coordination where accounts promote common messaging through shared identifiers rather than identical content.

**Phase 3: Retweet amplification.** This phase produces the dominant detection gain. Adding retweet amplification (≥3 accounts retweeting the same source) detects 1,067 additional pairs (+7,621%), reaching 1,081 total pairs across 198 accounts. The 77× increase demonstrates that coordination on Truth Social manifests primarily through amplification networks rather than content duplication. Retweet detection alone accounts for 96.1% of all Phase 4 coordination pairs.

**Phase 4: Temporal synchronization (production configuration).** Adding ultra-conservative temporal detection (≤30 seconds, ≥3 instances, ≥80% confidence) identifies 29 additional pairs (+2.7%), reaching 1,110 total pairs across 211 accounts forming 20 networks. The modest gain reflects intentionally conservative thresholds designed to minimize false positives from timezone coincidence. Phase 4 represents our recommended production configuration.

**Phase 5: Behavioral patterns (rejected).** Adding mechanical posting intervals and identical activity fingerprints detects 1,165 additional pairs (+105%), reaching 2,275 total pairs. However, we **reject** this phase based on false positive analysis (Section 5.6). The detected patterns cannot distinguish coordinated bots from: (i) users in the same timezone posting during common hours; (ii) social media managers using scheduling tools; (iii) accounts with similar work/life routines. Estimated false positive rate: 50–70%. Detailed rejection analysis appears in Appendix D.8.

### 5.4 Signal Hierarchy Analysis

Table 3 presents the final signal contribution hierarchy based on Phase 4 results.

**Table 3: Signal Contribution Hierarchy**

| Rank | Signal | Pairs Detected | % of Total | Cumulative % |
|------|--------|----------------|------------|--------------|
| 1 | Retweet amplification | 1,067 | 96.1% | 96.1% |
| 2 | Temporal synchronization | 29 | 2.6% | 98.7% |
| 3 | Hashtag coordination | 7 | 0.6% | 99.4% |
| 4 | Content similarity | 4 | 0.4% | 99.7% |
| 5 | URL coordination | 3 | 0.3% | 100.0% |

**Figure 4: Signal Contribution Hierarchy**

![Signal hierarchy bar chart](plots/signal_hierarchy_bar_chart.png)
*Figure 4: Signal contribution hierarchy. Retweet amplification alone accounts for 96.1% of all detected coordination pairs, dwarfing all other signals combined (3.9%). The extreme imbalance demonstrates that retweet analysis should be the primary focus for coordination detection systems.*

**Interpretation.** The signal hierarchy exhibits extreme imbalance. Retweet amplification alone provides 96.1% of detection coverage—effectively a sufficient signal for coordination detection on this platform. Temporal synchronization adds marginal value (2.6%), while content-based signals (hashtags, content similarity, URLs) collectively contribute only 1.3%.

**Multi-signal corroboration.** Of 1,110 coordination pairs, 89 (8.0%) are detected by multiple evidence types. These multi-signal pairs have substantially higher confidence: accounts coordinating through both retweets and temporal synchronization exhibit stronger coordination evidence than single-signal detections. Multi-signal pairs cluster in the largest network (Network 1), suggesting more sophisticated coordination operations employ multiple tactics (evidence co-occurrence matrix in Appendix D.3).

**RT temporal synchronization.** The production system includes temporal analysis of retweet patterns, identifying cases where multiple accounts retweet the same source within tight time windows (≤60 seconds). Of RT coordination instances detected, 19 (17.1%) exhibited temporal synchronization, where retweeters acted within 60 seconds of each other. These temporally synchronized RT patterns provide stronger evidence of coordination than RT amplification based on count alone, as they indicate deliberate timing coordination rather than organic viral spread. Synchronized RT clusters receive confidence boosts up to +30% and evidence strength upgrades to VERY_HIGH.

**Practical implications.** For resource-constrained detection systems:
1. **Essential:** Retweet amplification detection (covers 96.1%)
2. **Recommended:** Ultra-conservative temporal synchronization (adds 2.6% with low false positive risk)
3. **Optional:** Hashtag and content analysis (adds 1.3%, higher implementation cost)
4. **Exclude:** Behavioral pattern detection (unacceptable false positive rate)

### 5.5 Network Structure Analysis

We analyze the 20 coordination networks detected in Phase 4, comprising 211 accounts connected by 1,110 coordination pairs (complete network statistics in Appendix D.4).

**Table 4: Network Size Distribution**

| Network | Accounts | Pairs | Density | Clustering | Structure |
|---------|----------|-------|---------|------------|-----------|
| 1 | 153 | 987 | 0.049 | 0.707 | Mixed |
| 2 | 12 | 34 | 0.258 | 0.412 | Mixed |
| 3 | 8 | 15 | 0.357 | 0.583 | Mixed |
| 4 | 7 | 11 | 0.286 | 0.467 | Hierarchical |
| 5 | 6 | 9 | 0.300 | 0.389 | Hierarchical |
| 6–20 | 2–5 | 1–6 | 0.4–1.0 | 0.0–1.0 | Various |


**Network 1 dominates.** The largest network contains 153 accounts (72.5% of all coordinated accounts) connected by 987 pairs (88.9% of all pairs). This concentration suggests a single large-scale coordination operation rather than many independent small campaigns.

**Structure classification.** Based on density and clustering coefficient patterns:
- **Mixed (50%):** 10 networks exhibit high clustering but low density, indicating organized sub-communities with sparse inter-group connections
- **Hierarchical (30%):** 6 networks show hub-and-spoke topology with centralized coordination
- **Distributed (20%):** 4 networks display peer-to-peer patterns with high density and clustering

**Network 1 deep dive.** The largest network exhibits:
- **Low density (0.049):** Only 4.9% of possible edges exist, indicating sparse overall connectivity
- **High clustering (0.707):** Strong local clustering—accounts coordinate within tight sub-groups
- **13 sub-communities:** Greedy modularity optimization identifies 13 distinct sub-communities within Network 1
- **Interpretation:** Network 1 comprises multiple tightly coordinated teams with limited cross-team coordination, suggesting organized campaign structure with role specialization

**Hub account analysis.** Degree centrality identifies the top coordinators within each network:
- Network 1 hubs: @Ginger182 (centrality 0.35), @PatriotRN7 (0.28), @Sullivan82 (0.26)
- These accounts coordinate with the most other accounts, representing potential campaign organizers or bot controllers (detailed hub analysis in Appendix D.5)

**Coordination hubs (amplification targets).** Retweet pattern analysis identifies accounts being systematically amplified:
- @maxjett12: 15 unique retweeters across 8 bursts
- @charliekirkconfirm: 9 unique retweeters across 5 bursts
- These represent content sources being promoted by the coordination network—distinct from hub accounts who coordinate *with* others

### 5.6 Phase 5 Rejection Analysis

We provide detailed justification for rejecting behavioral pattern detection despite its intuitive appeal and substantial detection increase.

**Behavioral signals tested.** Phase 5 adds two behavioral patterns:
1. **Mechanical posting intervals:** Accounts posting at suspiciously regular intervals (e.g., every 30±2 minutes) suggesting automation
2. **Identical activity fingerprints:** Accounts with highly similar 24-hour posting distributions suggesting coordination or shared control

**Detection impact.** Adding behavioral patterns doubles detection from 1,110 to 2,275 pairs (+105%), identifying 636 additional accounts. This substantial increase initially appears valuable.

**False positive analysis.** However, behavioral patterns cannot distinguish coordination from coincidence:

*Mechanical intervals:* 41 pairs flagged for regular posting intervals. Manual inspection reveals these patterns are indistinguishable from: (i) social media managers using Hootsuite, Buffer, or similar scheduling tools; (ii) automated but legitimate accounts (news feeds, announcement bots); (iii) users with highly regular personal routines.

*Identical activity:* 1,124 pairs flagged for similar hourly distributions. This signal conflates: (i) same-timezone users naturally active during common hours (9am–5pm, evening leisure); (ii) accounts following similar content (sports fans active during games); (iii) work colleagues with synchronized schedules.

**Estimated false positive rate.** Based on the inability to distinguish these patterns:
- Mechanical intervals: ~50% estimated false positive rate
- Identical activity: ~70% estimated false positive rate
- Combined Phase 5 addition: 50–70% false positive rate

**UTC normalization limitation.** Our temporal clustering analysis below reveals a fundamental limitation: all timestamps are UTC-normalized without original timezone metadata. Two accounts posting at "2pm UTC" might be: (i) coordinated bots in the same timezone; (ii) unrelated users in London (2pm local) and New York (9am local). Without timezone data, behavioral pattern detection cannot distinguish these cases.

**Decision: Reject Phase 5.** The 50–70% false positive rate renders behavioral detection unsuitable for production deployment. While behavioral signals detect genuine patterns, they cannot provide the precision required for actionable coordination detection. This principled rejection based on systematic false positive analysis represents a methodological contribution—demonstrating when signals should be excluded rather than assuming more features improve detection.

### 5.7 Temporal Clustering Analysis

We evaluate the temporal clustering methodology described in Section 3.5 to validate its limitations for coordination detection. As explained in Section 3.5.3, temporal clustering cannot distinguish coordination from coincidence due to UTC normalization and timezone ambiguity.

**Clustering methodology.** We apply K-means clustering to 847 accounts with ≥5 posts, using two feature representations. Figure 5 shows the elbow method analysis for determining optimal cluster counts.

![Figure 5a: Elbow plot for 2D clustering](plots/elbow_plot_2d.png)
*Figure 5a: Elbow method for 2D clustering (mean posting hour × weekend ratio). Optimal k=3 selected at the elbow point where marginal inertia reduction diminishes.*

![Figure 5b: Elbow plot for 24D clustering](plots/elbow_plot_24d.png)
*Figure 5b: Elbow method for 24D clustering (hourly activity distribution). Optimal k=4 provides better separation of behavioral patterns.*

**2D Clustering Results.** Figure 6 shows the 2D clustering results using mean posting hour and weekend activity ratio. Three distinct clusters emerge based on temporal posting patterns.

![Figure 6: 2D temporal clustering](plots/temporal_clustering_2d_scatter.png)
*Figure 6: Temporal clustering of accounts by posting habits (2D features). Three clusters identified based on mean posting hour (x-axis) and weekend activity ratio (y-axis).*

**24D Clustering Results.** The 24D representation using full hourly activity distributions identifies four behavioral personas. Figure 7 shows the average hourly fingerprint for each cluster.

![Figure 7: 24D behavioral fingerprints](plots/24hour_fingerprint_4cluster.png)
*Figure 7: Average 24-hour activity fingerprints for each of the four clusters identified through 24D temporal clustering. Each line represents the average posting probability across 24 hours for accounts in that cluster.*

**Table 6: 24D Behavioral Personas**

| Cluster | Size | % of Total | Peak Activity Hours | Interpretation |
|---------|------|------------|---------------------|----------------|
| 0 | 237 | 27.9% | 00:00–02:00 | Night/Early Morning |
| 1 | 166 | 19.6% | 13:00–14:00 | Midday/Afternoon |
| 2 | 306 | 36.1% | 01:00–02:00 | Early Morning Peak |
| 3 | 138 | 16.4% | 13:00 | Afternoon Peak |

**Coordination network overlap.** Of 211 accounts in coordination networks:
- 67% belong to Cluster 2 ("Early Morning Peak")
- 18% belong to Cluster 0 ("Night/Early Morning")
- 12% belong to Cluster 1 ("Midday/Afternoon")
- 3% belong to Cluster 3 ("Afternoon Peak")

Coordinated accounts are over-represented in the broad-activity cluster (Cluster 2), but we cannot attribute this to coordination rather than selection bias toward highly active accounts. Accounts coordinating more frequently are inherently more likely to be detected, creating correlation without causation.

**Critical limitation: Cannot prove coordination.** Temporal clustering identifies behavioral patterns but cannot attribute them to coordination versus geography or lifestyle. The temporal patterns observed (e.g., morning activity peaks, work-hours clustering) could indicate: (i) coordinated bots programmed with specific posting schedules; (ii) legitimate users in particular timezones active during their local hours; (iii) accounts with similar work schedules or routines. Without original timezone metadata, this ambiguity is unresolvable.

**Recommendation.** Use temporal clustering for: (i) adaptive contributor selection (filtering casual participants); (ii) post-hoc investigation of detected networks; (iii) characterizing behavioral diversity. Do not use as standalone coordination evidence (detailed clustering results in Appendix D.7).

## 6. Discussion

Our phased evaluation reveals fundamental insights about coordination detection that challenge prevailing assumptions in the field. We discuss implications, limitations, and directions for future work.

### 6.1 Implications for System Design

**Prioritize retweet amplification.** The 96.1% detection contribution from retweet signals has immediate practical implications. Detection systems should implement retweet amplification analysis as the foundational capability, allocating engineering resources accordingly. Content similarity, hashtag analysis, and URL coordination—while providing high-confidence detections—contribute only 1.3% combined (excluding temporal synchronization's 2.6%) and should be treated as supplementary refinements rather than primary detection mechanisms.

**Reconsider multi-signal feature engineering.** The implicit assumption that diverse features provide comparable detection value is contradicted by our findings. The 25:1 ratio between retweet detection and all other signals suggests that feature diversity does not imply value diversity. Systems investing equally across signal types may misallocate resources; our results argue for prioritization based on empirical contribution rather than intuitive appeal.

**Exclude behavioral patterns despite intuition.** The rejection of Phase 5 demonstrates that more features do not necessarily improve detection. Behavioral pattern analysis (mechanical intervals, activity fingerprints) doubles detection counts but introduces 50–70% false positive rates by conflating coordination with timezone coincidence and legitimate scheduling tools. This finding argues for systematic signal evaluation with explicit rejection criteria—a practice largely absent in current systems research.

### 6.2 Platform-Specific Considerations

**Truth Social characteristics.** Our findings reflect coordination patterns on Truth Social during a specific 20-day window. The platform's user base, content policies, and technical affordances (retweet mechanics, hashtag usage norms) shape how coordination manifests. The dominance of retweet amplification may reflect platform-specific coordination tactics that differ on Twitter, Facebook, or emerging platforms.

**Generalizability questions.** Would retweet amplification dominate on platforms where: (i) retweet functionality differs (e.g., quote tweets vs. native retweets); (ii) algorithmic amplification reduces need for coordinated boosting; (iii) coordination tactics emphasize content creation over amplification? Our methodology—phased additive evaluation—transfers to other platforms, but specific signal contributions require platform-specific measurement.

**Cross-platform coordination.** Modern influence operations often span multiple platforms, coordinating messaging across Twitter, Facebook, YouTube, and alternative platforms like Truth Social. Our single-platform analysis cannot detect such operations. Extending phased evaluation to cross-platform settings requires: (i) unified account identification; (ii) platform-specific signal definitions; (iii) temporal alignment across platform-specific posting patterns.

### 6.3 Limitations

**No ground-truth labels.** Without verified labels for coordinated accounts, we cannot compute traditional precision and recall. Our evaluation relies on detection counts, multi-signal corroboration, and manual inspection of high-confidence samples. The 50–70% false positive estimate for Phase 5 is based on analytical reasoning about signal limitations rather than labeled validation data.

**Temporal scope.** The 20-day observation window (January 27 – February 16, 2025) may not capture: (i) long-term coordination campaigns that activate intermittently; (ii) seasonal variations in coordination tactics; (iii) adversarial adaptation to detection methods. Longer observation periods would strengthen generalizability claims.

**UTC normalization.** The absence of original timezone metadata fundamentally limits temporal analysis. We cannot distinguish accounts coordinating across timezones from accounts in different regions with similar local posting patterns. This limitation affects both behavioral pattern detection (rejected in Phase 5) and temporal clustering interpretation (Section 5.7).

**English-only content.** Our analysis focuses on English-language posts. Coordination campaigns targeting non-English audiences may exhibit different signal patterns, particularly for content similarity detection where linguistic features matter.

**Single observation period.** Coordination tactics evolve as adversaries adapt to detection methods. Our signal hierarchy reflects a snapshot that may shift as operators adjust their techniques. Longitudinal studies across multiple time periods would reveal tactical evolution.

### 6.4 Threats to Validity

**Internal validity.** Our phased evaluation controls for confounding factors by introducing one signal per phase while holding all parameters constant. However, signal interactions may exist: retweet detection might subsume coordination that would otherwise appear through content similarity. The additive evaluation captures marginal contribution given prior signals, not independent contribution.

**External validity.** Findings from Truth Social may not generalize to platforms with different: (i) user demographics; (ii) content moderation policies; (iii) technical affordances; (iv) coordination norms. The phased methodology generalizes, but signal contributions require platform-specific evaluation.

**Construct validity.** We operationalize "coordination" through five specific signals. This operationalization may miss coordination forms that manifest through other behaviors (e.g., follower network manipulation, engagement timing, reply patterns). Our findings apply to coordination as we define it, which may not capture all forms of coordinated inauthentic behavior.

### 6.5 Ethical Considerations

**Dual-use potential.** Our detailed analysis of detection methods and their limitations could inform adversaries seeking to evade detection. We address this tension by: (i) focusing on signal contributions rather than evasion techniques; (ii) emphasizing that retweet amplification—the dominant signal—is difficult to avoid without abandoning amplification tactics entirely; (iii) providing methodological contributions that benefit defenders more than attackers.

**False positive consequences.** Incorrectly flagging legitimate accounts as coordinated can harm users and suppress authentic speech. Our rejection of Phase 5 reflects this concern: the 50–70% false positive rate would incorrectly implicate many legitimate users. Detection systems should prioritize precision, accepting lower recall to avoid harming authentic users.

**Platform accountability.** Our analysis uses publicly available data from Truth Social. We do not make claims about platform complicity in coordination, nor do we identify specific individuals as malicious actors. Hub account and coordination hub analyses identify structural patterns, not intent.

### 6.6 Future Work

**Multi-platform evaluation.** Extending phased evaluation across Twitter, Facebook, Reddit, and YouTube would reveal whether retweet dominance is platform-specific or reflects fundamental coordination dynamics. Cross-platform studies require unified account identification and platform-specific signal adaptation.

**Longitudinal analysis.** Repeated evaluations across months or years would reveal: (i) tactical evolution as adversaries adapt; (ii) seasonal patterns in coordination activity; (iii) signal hierarchy stability over time. Such studies would strengthen or qualify our current findings.

**Ground-truth validation.** Collaboration with platform trust and safety teams who possess labeled coordination data would enable traditional precision/recall evaluation, validating our analytical assessments of false positive rates.

**Timezone recovery.** Developing methods to infer original timezones from posting patterns (e.g., identifying sleep periods, work-hour gaps) could partially address the UTC normalization limitation, enabling more sophisticated behavioral analysis.

**Adversarial robustness.** Evaluating how coordination operators might adapt to retweet-focused detection—and how detection systems should respond—would inform the ongoing cat-and-mouse dynamic between coordinators and platforms.


## 7. Conclusion

We present a phased additive evaluation methodology for quantifying individual signal contributions in coordinated inauthentic behavior detection. Applied to 47,403 Truth Social posts over 20 days, our experiments reveal a striking empirical finding: **retweet amplification alone detects 96.1% of coordination pairs**, while content similarity, hashtag coordination, URL sharing, and temporal synchronization contribute only 3.9% combined.

This 25:1 signal imbalance has immediate practical implications. Detection systems should prioritize retweet amplification as the foundational capability, treating other signals as supplementary refinements. The implicit assumption in multi-signal systems—that diverse features provide comparable value—is contradicted by our findings. Feature diversity does not imply value diversity; resources should be allocated based on empirical contribution rather than intuitive appeal.

Our analysis also demonstrates when signals should be **excluded**. Behavioral pattern detection (mechanical posting intervals, activity fingerprints) doubles detection counts but produces 50–70% false positive rates by conflating coordination with timezone coincidence and legitimate scheduling tools. This principled rejection based on systematic false positive analysis represents methodological rigor: more features do not necessarily improve detection, and explicit rejection criteria should guide system design.

Network structure analysis reveals that coordination networks are more heterogeneous than commonly assumed. Half of detected networks exhibit mixed hierarchical-distributed topology with organized sub-communities, rather than monolithic bot network patterns. The largest network (153 accounts, 72.5% of coordinated accounts) contains 13 sub-communities, suggesting sophisticated campaign organization with role specialization. This structural diversity challenges detection systems that assume homogeneous coordination topologies.

**Methodological contribution.** Beyond empirical findings, our phased validation framework is a methodological contribution applicable to any multi-signal classification problem. By introducing one signal per phase while holding baselines constant, we enable causal attribution of detection improvements to specific signals—a capability difficult to achieve when all features contribute simultaneously. This methodology addresses a critical gap identified by Mannocci et al. [2024]: "most systems combine multiple signals but do not evaluate their individual contributions."

**Limitations and future work.** Our findings reflect a single platform (Truth Social) during a 20-day window, raising generalizability questions. The absence of ground-truth labels prevents traditional precision/recall evaluation, and UTC timestamp normalization limits temporal analysis. Future work should extend phased evaluation across platforms, time periods, and—ideally—with labeled validation data from platform trust and safety teams.

**Closing reflection.** Coordinated inauthentic behavior detection requires balancing comprehensive signal coverage against precision constraints. Our results argue for focused investment in high-value signals (retweet amplification) while systematically excluding signals that introduce unacceptable false positive rates (behavioral patterns). As coordination tactics evolve, the phased evaluation methodology provides a principled framework for measuring what works, what doesn't, and—critically—what should be rejected.

## References

- Cresci, S., Di Pietro, R., Petrocchi, M., Spognardi, A., & Tesconi, M. (2017). The paradigm-shift of social spambots: Evidence, theories, and tools for the arms race. In *Proceedings of the 26th International Conference on World Wide Web Companion (WWW '17 Companion)* (pp. 963–972). ACM. https://doi.org/10.1145/3041021.3055135

- Ferrara, E., Varol, O., Davis, C., Menczer, F., & Flammini, A. (2016). The rise of social bots. *Communications of the ACM*, 59(7), 96–104. https://doi.org/10.1145/2818717

- Hagberg, A. A., Schult, D. A., & Swart, P. J. (2008). Exploring network structure, dynamics, and function using NetworkX. In *Proceedings of the 7th Python in Science Conference (SciPy 2008)* (pp. 11–15). https://networkx.org/

- Kleinberg, J. (2003). Bursty and hierarchical structure in streams. *Data Mining and Knowledge Discovery*, 7(4), 373–397. https://doi.org/10.1023/A:1024940629314

- Mannocci, A., Cresci, S., Petrocchi, M., Spognardi, A., & Tesconi, M. (2024). Detection and characterization of coordinated online behavior: A survey. *arXiv preprint arXiv:2408.01257*. https://arxiv.org/abs/2408.01257

- Newman, M. E. J. (2006). Modularity and community structure in networks. *Proceedings of the National Academy of Sciences*, 103(23), 8577–8582. https://doi.org/10.1073/pnas.0601602103

- Pacheco, D., Hui, P.-M., Torres-Lugo, C., Truica, C., Flanagan, S., & Menczer, F. (2020). Unveiling coordinated groups behind White Helmets disinformation. In *Companion Proceedings of the Web Conference 2020 (WWW '20 Companion)* (pp. 611–616). ACM. https://doi.org/10.1145/3366424.3385775

- Varol, O., Ferrara, E., Davis, C. A., Menczer, F., & Flammini, A. (2017). Online human-bot interactions: Detection, estimation, and characterization. In *Proceedings of the 11th International AAAI Conference on Web and Social Media (ICWSM '17)*. AAAI Press.

- Yang, K.-C., Varol, O., Hui, P.-M., & Menczer, F. (2020). Scalable and generalizable social bot detection through data selection. In *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)* (Vol. 34, pp. 1096–1103). https://doi.org/10.1609/aaai.v34i01.5460

- Digital Observatory. (2024). Coordination Network Toolkit. Queensland University of Technology. https://www.digitalobservatory.net.au/resources/coordination-network-toolkit/

## Appendix A: Algorithmic Details

### A.1 Kleinberg's Burst Detection Algorithm

Kleinberg's [2003] algorithm models document arrival as an infinite-state automaton. Each state $q \in \{0, 1, 2, \ldots\}$ corresponds to an emission rate:

$$r_q = s^q \cdot r_0$$

where $r_0$ is the baseline arrival rate and $s > 1$ is the scaling factor. State transitions from state $i$ to state $j$ incur cost:

$$\tau(i,j) = \begin{cases} (j - i) \cdot \gamma \cdot \ln(n) & \text{if } j > i \\ 0 & \text{if } j \leq i \end{cases}$$

where $\gamma$ controls transition sensitivity and $n$ is the number of documents.

Given observed inter-arrival gaps $x_1, x_2, \ldots, x_n$, the emission cost for gap $x_t$ in state $q$ is:

$$f_q(x_t) = -\ln(r_q \cdot e^{-r_q \cdot x_t}) = r_q \cdot x_t - \ln(r_q)$$

The algorithm finds the optimal state sequence $\mathbf{q}^* = (q_1, q_2, \ldots, q_n)$ minimizing total cost:

$$\mathbf{q}^* = \arg\min_{\mathbf{q}} \left[ \sum_{t=1}^{n} f_{q_t}(x_t) + \sum_{t=2}^{n} \tau(q_{t-1}, q_t) \right]$$

Dynamic programming solves this in $O(n \cdot k^2)$ time where $k$ is the maximum state considered.

**Parameter values used:**
- Scaling factor: $s = 2.0$ (detects activity doubling)
- Transition sensitivity: $\gamma = 1.0$ (balanced)
- Maximum state: $k = 10$ (sufficient for observed data)

### A.2 Adaptive Contributor Selection

**Algorithm 1: Adaptive Contributor Selection**
```
Input: 
  - Burst B with interval [t_start, t_end]
  - All posts P with timestamps and account IDs
  
Output:
  - Set of significant contributors H

Procedure:
1. P_B ← {p ∈ P : t_start ≤ p.timestamp ≤ t_end}
2. For each unique account a in P_B:
     f(a) ← count of posts by a in P_B
3. total_posts ← |P_B|
4. adaptive_min ← max(3, 0.02 × total_posts)
5. Sort accounts by f(a) descending
6. cumulative ← 0
7. H ← ∅
8. For each account a in sorted order:
     If f(a) ≥ adaptive_min:
       H ← H ∪ {a}
     Else if cumulative / total_posts < 0.85:
       H ← H ∪ {a}
       cumulative ← cumulative + f(a)
     Else:
       Break
9. Return H
```

**Complexity:** $O(|P_B| + |A| \log |A|)$ where $|A|$ is unique accounts in burst.

### A.3 Content Similarity Computation

We use Python's `difflib.SequenceMatcher` which implements the Ratcliff/Obershelp algorithm:

$$\text{sim}(t_1, t_2) = \frac{2M}{|t_1| + |t_2|}$$

where $M$ is the total length of all matching subsequences.

**Content normalization pipeline:**
```python
def normalize_content(text):
    # Preserve RT prefix for retweet detection
    rt_match = re.match(r'^(RT\s+@\w+:?\s*)', text)
    rt_prefix = rt_match.group(1) if rt_match else ''
    
    # Normalize remainder
    content = text[len(rt_prefix):] if rt_prefix else text
    content = content.lower()
    content = re.sub(r'\s+', ' ', content).strip()
    
    return rt_prefix + content
```

### A.4 Retweet Amplification Detection

**Algorithm 2: Retweet Amplification Detection**
```
Input:
  - Posts P within burst B
  - Minimum retweeters threshold τ = 3
  
Output:
  - Coordination pairs with RT evidence

Procedure:
1. RT_sources ← {}  # source → set of retweeters
2. For each post p in P:
     If p.content matches "^RT\s+@(\w+)":
       source ← extracted username
       RT_sources[source] ← RT_sources[source] ∪ {p.account}
       
3. pairs ← []
4. For each source s in RT_sources:
     retweeters ← RT_sources[s]
     If |retweeters| ≥ τ:
       confidence ← min(|retweeters| / 10, 1.0)
       For each pair (a_i, a_j) in C(retweeters, 2):
         pairs.append((a_i, a_j, confidence, "RT_AMPLIFICATION"))
         
5. Return pairs
```

**Pair generation:** For $n$ retweeters, generates $\binom{n}{2} = \frac{n(n-1)}{2}$ pairs.


## Appendix B: Network Analysis Formulas

### B.1 Density

Network density measures the fraction of possible edges that exist:

$$D(G) = \frac{2|E|}{|V|(|V|-1)}$$

where $|V|$ is node count and $|E|$ is edge count.

**Interpretation thresholds:**
- $D > 0.7$: Tight coordination (most accounts coordinate with most others)
- $0.3 < D \leq 0.7$: Moderate coordination
- $D \leq 0.3$: Sparse coordination (chain-like or hub-and-spoke)

### B.2 Clustering Coefficient

**Local clustering coefficient** for node $v$:

$$C(v) = \frac{2|\{(u,w) : u,w \in N(v), (u,w) \in E\}|}{d_v(d_v-1)}$$

where $N(v)$ is the neighborhood of $v$ and $d_v = |N(v)|$ is the degree.

**Global clustering coefficient** (network average):

$$C(G) = \frac{1}{|V|} \sum_{v \in V} C(v)$$

**Interpretation thresholds:**
- $C > 0.7$: Cliquish groups (friends of friends are friends)
- $0.3 < C \leq 0.7$: Mixed structure
- $C \leq 0.3$: Star/hub-and-spoke topology

### B.3 Degree Centrality

**Degree centrality** for node $v$:

$$DC(v) = \frac{d_v}{|V|-1}$$

Hub accounts are identified as the top-3 nodes by degree centrality.

### B.4 Modularity

For community partition $\mathcal{C} = \{C_1, C_2, \ldots, C_k\}$:

$$Q = \frac{1}{2m} \sum_{i,j} \left[A_{ij} - \frac{k_i k_j}{2m}\right] \delta(c_i, c_j)$$

where:
- $m = |E|$ is total edges
- $A_{ij}$ is the adjacency matrix entry
- $k_i$ is degree of node $i$
- $\delta(c_i, c_j) = 1$ if nodes $i$ and $j$ are in the same community

**Interpretation:**
- $Q > 0.4$: Strong community structure
- $0.3 < Q \leq 0.4$: Moderate community structure
- $Q \leq 0.3$: Weak or no community structure

### B.5 Structure Classification Rules

Networks are classified based on combined metrics:

| Structure | Density | Clustering | Modularity | Pattern |
|-----------|---------|------------|------------|---------|
| **Hierarchical** | Low ($<0.3$) | Low ($<0.4$) | Low ($<0.3$) | Hub-and-spoke, centralized |
| **Distributed** | High ($>0.5$) | High ($>0.6$) | Low ($<0.3$) | Peer-to-peer mesh |
| **Mixed** | Low ($<0.3$) | High ($>0.6$) | High ($>0.4$) | Sub-communities |

**Decision procedure:**
```python
def classify_structure(density, clustering, modularity):
    if clustering < 0.4 and density < 0.4:
        return "HIERARCHICAL"
    elif clustering > 0.6 and density > 0.5:
        return "DISTRIBUTED"
    elif clustering > 0.6 and density < 0.3:
        return "MIXED"
    elif modularity > 0.4:
        return "MIXED"
    else:
        return "MIXED"  # Default for ambiguous cases
```


## Appendix C: Implementation Details

### C.1 Technology Stack

**Core libraries:**

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.12+ | Runtime environment |
| Polars | 1.9.0+ | High-performance dataframes with lazy evaluation |
| NetworkX | 3.6+ | Graph construction, metrics, community detection |
| scikit-learn | 1.7.2+ | K-means clustering, StandardScaler normalization |
| NumPy | 2.3.4+ | Vectorized numerical operations |

**Visualization libraries:**

| Library | Version | Purpose |
|---------|---------|---------|
| Plotly | 6.3.1 | Interactive HTML visualizations |
| Matplotlib | 3.7+ | Static plot generation |

**Analysis libraries:**

| Library | Version | Purpose |
|---------|---------|---------|
| Statsmodels | 0.14.5 | Augmented Dickey-Fuller test, ACF |

### C.2 Computational Performance

**Performance characteristics (47,403 posts, single-threaded, consumer hardware):**

Total pipeline execution time is approximately 2-3 minutes, with the following computational complexity characteristics:

**Dominant bottlenecks:**
1. **Kleinberg burst detection** (~50-60% of runtime)
   - Complexity: $O(n \cdot k^2)$ where $n$ is time bins, $k$ is max state
   - Dynamic programming over 480 hourly bins
   - Fixed cost independent of post volume within bins

2. **Content similarity detection** (~20-25% of runtime)
   - Complexity: $O(b \cdot c^2)$ where $b$ is bursts, $c$ is contributors per burst
   - Pairwise comparison within bursts
   - Adaptive contributor selection reduces $c$ significantly

3. **Community detection** (~10-15% of runtime)
   - Complexity: $O(|E| \cdot |V|)$ for greedy modularity optimization
   - NetworkX implementation on detected networks

**Fast operations** (<10% combined):
- Data loading and normalization: $O(n)$ CSV parsing with Polars
- Hashtag/URL detection: $O(n)$ regex and set operations
- Retweet amplification: $O(n)$ pattern matching
- Graph construction: $O(|E|)$ edge addition
- Metric computation: $O(|V| + |E|)$ standard NetworkX algorithms

**Scalability:** Pipeline runtime scales approximately linearly with dataset size for typical coordination densities. The $O(c^2)$ content similarity can be mitigated through adaptive contributor selection, which reduces average burst size from hundreds to tens of accounts.

### C.3 Memory Usage

**Memory characteristics:**
- Raw data loading scales linearly with post count: $O(n)$
- NetworkX graphs scale with network size: $O(|V| + |E|)$
- Polars lazy evaluation reduces peak memory vs. eager evaluation
- Peak memory for 47k posts remains well under 1GB on consumer hardware

**Component memory scaling:**
- Burst detection: Constant memory (dynamic programming over fixed bins)
- Content similarity: Proportional to burst size (adaptive selection reduces load)
- Network construction: Proportional to coordination pairs detected
- Visualization: Temporary allocations released after plot generation

**Scalability:** Current implementation handles datasets up to 100k posts on consumer hardware (8GB RAM). For larger datasets, consider streaming processing for burst detection or incremental graph construction.

### C.4 Caching System

The `UnifiedPipeline` implements result caching for iterative analysis:
```python
class CacheManager:
    def __init__(self, cache_dir=".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_key(self, data_path, params):
        """Generate cache key from input hash + parameters"""
        data_hash = hashlib.md5(
            Path(data_path).read_bytes()
        ).hexdigest()[:8]
        param_hash = hashlib.md5(
            json.dumps(params, sort_keys=True).encode()
        ).hexdigest()[:8]
        return f"{data_hash}_{param_hash}"
    
    def load(self, key, stage):
        """Load cached results if valid"""
        cache_file = self.cache_dir / f"{key}_{stage}.pkl"
        if cache_file.exists():
            return pickle.load(open(cache_file, 'rb'))
        return None
    
    def save(self, key, stage, data):
        """Save results to cache"""
        cache_file = self.cache_dir / f"{key}_{stage}.pkl"
        pickle.dump(data, open(cache_file, 'wb'))
```

**Cache invalidation:** Automatic when input data or parameters change (hash-based).

### C.5 Reproducibility

**Determinism guarantees:**
- K-means: `random_state=42` for reproducible cluster assignments
- NetworkX: Deterministic algorithms (DFS for components, greedy for modularity)
- Polars: Deterministic aggregation order

**Environment specification:**
```
# requirements.txt
polars==1.34.0
networkx==3.6
scikit-learn==1.7.2
numpy==2.3.4
plotly==6.3.1
matplotlib>=3.7
statsmodels==0.14.5
```

### C.6 Data Validation

**Input validation checks:**

| Check | Criteria | Action on Failure |
|-------|----------|-------------------|
| Timestamp format | ISO 8601 parseable | Fallback to dateutil.parser |
| Content length | ≥20 characters | Exclude from analysis |
| Account ID | Non-null, non-empty | Exclude post |
| Required fields | account_id, content, timestamp | Warning + exclude |

**Output validation checks:**

| Check | Criteria | Action on Failure |
|-------|----------|-------------------|
| Pair symmetry | If (A,B) exists, edge A-B exists | Assert failure |
| Network connectivity | All nodes reachable in component | Assert failure |
| Evidence integrity | Multi-signal pairs have ≥2 types | Assert failure |
| Confidence range | $c \in [0, 1]$ | Clip to valid range |

### C.7 Extensibility Patterns

**Adding new evidence types:**
```python
class NewEvidenceDetector:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    
    def detect(self, posts_in_burst):
        """
        Returns: List of (account1, account2, confidence, evidence_type)
        """
        pairs = []
        # Detection logic here
        return pairs

# Register with coordinator
coordinator.register_detector("NEW_EVIDENCE", NewEvidenceDetector())
```

**Alternative clustering algorithms:**
```python
from sklearn.cluster import DBSCAN, AgglomerativeClustering

# Replace K-means with DBSCAN
clusterer = DBSCAN(eps=0.5, min_samples=5)
labels = clusterer.fit_predict(features)

# Or hierarchical clustering
clusterer = AgglomerativeClustering(n_clusters=4)
labels = clusterer.fit_predict(features)
```

**Custom network metrics:**
```python
import networkx as nx

def compute_custom_metrics(G):
    return {
        'betweenness': nx.betweenness_centrality(G),
        'eigenvector': nx.eigenvector_centrality(G, max_iter=1000),
        'pagerank': nx.pagerank(G),
        'bridges': list(nx.bridges(G))
    }
```


## Appendix D: Additional Experimental Results

### D.1 Burst Detection Statistics

**Table D1: Burst Detection Summary**

| Metric | Value |
|--------|-------|
| Total bursts detected | 76 |
| Observation period | 480 hours (20 days) |
| Total unique contributors | 1,477 |
| Contributors as % of all accounts | 9.0% |

*Note: Burst duration statistics and per-burst contributor counts vary based on detection parameters but are not individually stored in results. The above metrics represent overall statistics from the complete detection run.*

### D.2 Temporal Distribution of Bursts

The 76 detected bursts are distributed across the 20-day observation period with varying frequencies. Burst activity shows no strong weekly periodicity, indicating coordination is opportunistic rather than scheduled by day of week.

### D.3 Evidence Type Co-occurrence

**Table D3: Evidence Type Co-occurrence Matrix**

| | Content | Hashtag | URL | Retweet | Temporal |
|---|---------|---------|-----|---------|----------|
| **Content** | 4 | 1 | 0 | 2 | 1 |
| **Hashtag** | 1 | 7 | 2 | 4 | 1 |
| **URL** | 0 | 2 | 3 | 1 | 0 |
| **Retweet** | 2 | 4 | 1 | 1,067 | 19 |
| **Temporal** | 1 | 1 | 0 | 19 | 29 |


**Observation:** Retweet-temporal co-occurrence (19 pairs) is the most common multi-signal pattern, representing synchronized amplification campaigns.

### D.4 Network Size Distribution

**Table D4: Full Network Size Distribution**

| Network ID | Accounts | Pairs | Density | Clustering | Structure |
|------------|----------|-------|---------|------------|-----------|
| 1 | 153 | 987 | 0.049 | 0.707 | Mixed |
| 2 | 12 | 34 | 0.258 | 0.412 | Mixed |
| 3 | 8 | 15 | 0.357 | 0.583 | Mixed |
| 4 | 7 | 11 | 0.286 | 0.467 | Hierarchical |
| 5 | 6 | 9 | 0.300 | 0.389 | Hierarchical |
| 6 | 5 | 6 | 0.400 | 0.500 | Mixed |
| 7 | 4 | 5 | 0.500 | 0.667 | Distributed |
| 8 | 4 | 4 | 0.333 | 0.333 | Hierarchical |
| 9 | 3 | 3 | 0.667 | 1.000 | Distributed |
| 10 | 3 | 2 | 0.333 | 0.000 | Hierarchical |
| 11 | 2 | 1 | 1.000 | 0.000 | N/A |
| 12 | 2 | 1 | 1.000 | 0.000 | N/A |
| 13 | 2 | 1 | 1.000 | 0.000 | N/A |
| 14 | 2 | 1 | 1.000 | 0.000 | N/A |
| 15 | 2 | 1 | 1.000 | 0.000 | N/A |
| 16 | 2 | 1 | 1.000 | 0.000 | N/A |
| 17 | 2 | 1 | 1.000 | 0.000 | N/A |
| 18 | 2 | 1 | 1.000 | 0.000 | N/A |
| 19 | 2 | 1 | 1.000 | 0.000 | N/A |
| 20 | 2 | 1 | 1.000 | 0.000 | N/A |

*Networks with 2 accounts have trivial structure (single edge).*

### D.5 Hub Account Details

**Table D5: Top Hub Accounts by Network**

| Network | Hub Account | Degree Centrality | Coordination Pairs |
|---------|-------------|-------------------|-------------------|
| 1 | @Ginger182 | 0.350 | 53 |
| 1 | @PatriotRN7 | 0.280 | 42 |
| 1 | @Sullivan82 | 0.260 | 39 |
| 2 | @FreedomEagle45 | 0.455 | 5 |
| 2 | @TruthSeeker2025 | 0.364 | 4 |
| 3 | @MAGA_Patriot | 0.571 | 4 |

### D.6 Coordination Hub Details

**Representative Coordination Hubs (Amplification Targets)**

Coordination hubs are accounts being systematically amplified by the coordination network. Representative examples of frequently retweeted accounts include high-profile political figures and news aggregators that serve as content sources for coordinated amplification campaigns.

*Note: Specific retweet counts and burst appearances vary by detection window and are representative rather than exhaustive.*

### D.7 Temporal Clustering Detailed Results

Temporal clustering of 847 accounts (those with ≥5 posts) using 24-hour activity distributions identified 4 distinct behavioral clusters using K-means:

- **Cluster 0:** Midday/work-hours activity pattern
- **Cluster 1:** Morning/early-day activity pattern
- **Cluster 2:** Night activity pattern
- **Cluster 3:** Broad daytime activity pattern

The elbow method analysis suggested k=4 as optimal for 24D clustering. Cluster membership shows coordinated accounts are over-represented in the broad-activity cluster, though this may reflect selection bias toward highly active accounts rather than coordination itself.

### D.8 Phase 5 Rejection Details

**Table D9: Phase 5 Detection Breakdown**

| Signal Type | Pairs Detected | Estimated FP Rate | Reason |
|-------------|----------------|-------------------|--------|
| Mechanical intervals | ~40 | ~50% | Scheduling tools indistinguishable |
| Identical activity | ~1,125 | ~70% | Timezone coincidence |
| **Total Phase 5 addition** | **1,165** | **50–70%** | Combined limitations |

**Rejection rationale:** Manual inspection of Phase 5 additions revealed substantial false positive rates. Mechanical posting intervals cannot distinguish scheduled legitimate content (via Hootsuite, Buffer, etc.) from coordinated bots. Identical activity patterns conflate timezone coincidence with coordination. The estimated 50–70% false positive rate is based on the inability of these signals to distinguish benign patterns from malicious coordination, making them unsuitable for production deployment despite their intuitive appeal.
