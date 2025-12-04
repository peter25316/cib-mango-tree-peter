# CIB Mango Tree: Coordinated Inauthentic Behavior Detection

A multi-signal coordination detection system with phased evaluation framework for social media platforms. This research reveals that **retweet amplification alone detects 96.1% of coordination**, challenging assumptions about multi-signal detection complexity.

## 🎯 Key Findings

**Main Discovery:** Retweet amplification detection accounts for 96.1% of all detected coordination pairs, while content similarity, hashtag coordination, URL sharing, and temporal synchronization contribute only 3.9% combined—a 25:1 signal ratio.

**Methodology Contribution:** Phased additive evaluation framework that isolates individual signal contributions, enabling causal attribution of detection improvements to specific signals and principled signal rejection based on false positive analysis.

**Network Insights:** Coordination networks exhibit heterogeneous structure (50% mixed topology, 30% hierarchical, 20% distributed) with sophisticated sub-community organization, challenging assumptions about uniform bot network patterns.

## 📊 Quick Results Summary

**Dataset:** 47,403 Truth Social posts over 20 days (January 27 – February 16, 2025) from 16,468 unique accounts

**Detection Results (Phase 4 - Production Configuration):**
- **Total coordination pairs detected:** 1,110
- **Coordinated accounts identified:** 211 (1.3% of total accounts)
- **Networks detected:** 20 (largest: 153 accounts, 72.5% of coordinated accounts)
- **Temporal bursts:** 76 periods of elevated activity

**Signal Contribution Hierarchy:**
1. 🔁 **Retweet amplification:** 96.1% (1,067 pairs) — ESSENTIAL
2. ⏱️ **Temporal synchronization:** 2.6% (29 pairs) — Recommended
3. #️⃣ **Hashtag coordination:** 0.6% (7 pairs) — Optional
4. 📝 **Content similarity:** 0.4% (4 pairs) — Optional
5. 🔗 **URL coordination:** 0.3% (3 pairs) — Optional

## 🎥 Video Showcase

Watch our interactive dashboard and detection pipeline in action:

<video src="demo/fig/poster-video.mp4" controls width="100%"></video>

> - **Direct Link:** [Click here to download/view the video](demo/fig/poster-video.mp4)



> - **Local:** Run `demo/interactive_burst_app.py` for live interactive dashboard

**What the video demonstrates:**
- 📊 Interactive Streamlit dashboard for burst detection visualization
- 🕸️ Network coordination analysis with real-time metrics
- 📈 Temporal burst patterns and activity fingerprints
- 🔍 Multi-signal coordination detection in action

## 📁 Repository Structure

```
cib-mango-tree-peter/
├── research_paper/               # 📄 Complete academic research paper
│   ├── research_paper.md         # Full research paper (Markdown)
│   ├── research_paper.pdf        # PDF version
│   └── convert_to_pdf.py         # Markdown to PDF converter
├── reports/                      # 📊 Project reports
│   ├── Final_Report.md           # Final project report (Markdown)
│   ├── Final_Report.pdf          # PDF version
│   ├── convert_to_pdf.py         # Markdown to PDF converter
│   └── Progress_Report/          # Weekly progress updates
│       └── Progress_Reports.md
├── src/                          # 💻 Source code
│   ├── components/               # Core detection modules
│   │   ├── burst_detector_enhanced.py
│   │   ├── content_coordination_detector.py
│   │   ├── data_analyzer.py
│   │   ├── kleinberg_utils.py
│   │   ├── temporal_clusterer.py
│   │   └── visualizer.py
│   ├── docs/                     # 📚 Detailed documentation
│   │   ├── COORDINATION_DETECTION_EXPLAINED.md
│   │   ├── KLEINBERG_ALGORITHM_EXPLAINED.md
│   │   ├── NETWORK_ANALYSIS_CONCEPTS.md
│   │   ├── NETWORK_EDGE_ANALYSIS.md
│   │   └── TEMPORAL_CLUSTERING_EXPLAINED.md
│   ├── shellscripts/             # 🔧 Pipeline scripts
│   │   └── unified_pipeline.py   # Main detection pipeline
│   └── tests/                    # 🧪 Testing and experiments
│       ├── experiments/          # Phased evaluation framework
│       │   ├── phase1_content_only.py
│       │   ├── phase2_add_patterns.py
│       │   ├── phase3_add_retweets.py
│       │   ├── phase4_add_temporal.py
│       │   ├── phase5_behavioral.py (REJECTED)
│       │   ├── PHASE_EXPERIMENTS.md
│       │   ├── run_all_phases.py
│       │   └── results/          # JSON result files
│       └── test_suite.py         # Unit tests
├── demo/                         # 🎨 Interactive dashboard
│   ├── interactive_burst_app.py  # Streamlit visualization app
│   ├── fig/                      # Static figures for demo
│   └── plots/                    # Generated visualizations
├── data/                         # 📊 Sample datasets
│   └── sampledata_truthsocial.csv
├── cookbooks/                    # 📓 Jupyter notebooks
│   └── capstone-analysis.ipynb
├── presentation/                 # 🎤 Presentation materials
├── proposal/                     # 📋 Project proposal documents
├── requirements.txt              # 📦 Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/peter25316/cib-mango-tree-peter.git
cd cib-mango-tree-peter

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Complete Detection Pipeline

```bash
# Run full analysis pipeline
python src/shellscripts/unified_pipeline.py

# Results will be saved to cache/ and visualizations to plots/
```

### Run Interactive Dashboard

```bash
# Launch Streamlit app
cd demo
streamlit run interactive_burst_app.py

# Open browser to http://localhost:8501
```

### Run Phased Experiments

```bash
# Run individual phases
cd src/tests/experiments
python phase1_content_only.py
python phase2_add_patterns.py
python phase3_add_retweets.py
python phase4_add_temporal.py

# Or run all phases sequentially
python run_all_phases.py
```

## 📖 Documentation

### Research Paper
- **[Complete Research Paper](research_paper/research_paper.md)** - Full academic paper with methodology, experimental results, discussion, and appendices

### Methodology Documentation
- **[Coordination Detection](src/docs/COORDINATION_DETECTION_EXPLAINED.md)** - Multi-signal detection methodology
- **[Kleinberg Algorithm](src/docs/KLEINBERG_ALGORITHM_EXPLAINED.md)** - Burst detection algorithm explanation
- **[Network Analysis](src/docs/NETWORK_ANALYSIS_CONCEPTS.md)** - Graph metrics and structure classification
- **[Temporal Clustering](src/docs/TEMPORAL_CLUSTERING_EXPLAINED.md)** - Alternative methodology (experimental)

### Experimental Results
- **[Phase Experiments](src/tests/experiments/PHASE_EXPERIMENTS.md)** - Detailed phased evaluation (Phase 1-5) with results and analysis
- **[Progress Reports](reports/Progress_Report/Progress_Reports.md)** - Weekly development log

## 🔬 Methodology Overview

### 1. Burst Detection (Kleinberg's Algorithm)
Identifies temporal windows of elevated posting activity using infinite-state automaton modeling with parameters s=2.0, γ=1.0.

### 2. Multi-Signal Coordination Detection
Detects coordination through five evidence types:
- **Content Similarity:** SequenceMatcher ≥0.95 (identical) or ≥0.85 (high similarity)
- **Hashtag Coordination:** Jaccard similarity ≥0.6 with ≥2 shared hashtags
- **URL Coordination:** ≥1 shared URL within burst window
- **Retweet Amplification:** ≥3 accounts retweeting same source
- **Temporal Synchronization:** ≤30 second windows with ≥3 synchronized posts

### 3. Network Analysis (NetworkX)
Constructs coordination graphs and computes:
- Density, clustering coefficient, degree centrality, modularity
- Structure classification (hierarchical, distributed, mixed)
- Hub identification and sub-community detection

### 4. Phased Validation Framework
Progressive signal evaluation enabling causal attribution:
- **Phase 1:** Content similarity baseline
- **Phase 2:** + Hashtag and URL coordination
- **Phase 3:** + Retweet amplification (96.1% coverage)
- **Phase 4:** + Temporal synchronization (production)
- **Phase 5:** + Behavioral patterns ❌ **REJECTED** (50-70% false positive rate)

## 💡 Key Contributions

### 1. Empirical Signal Quantification
First systematic measurement of individual signal contributions in coordination detection, revealing extreme signal imbalance (96.1% vs 3.9%).

### 2. Principled Signal Rejection
Demonstrates when signals should be **excluded** based on false positive analysis. Behavioral pattern detection rejected despite doubling detection counts due to unacceptable false positive rates.

### 3. Network Structure Heterogeneity
Challenges assumptions about uniform coordination topology. Half of detected networks exhibit mixed structure with organized sub-communities rather than monolithic bot patterns.

### 4. Reusable Methodology
Phased additive evaluation framework applicable to any multi-signal classification problem, enabling controlled measurement of feature contributions.

## 📊 Network Analysis Highlights

**Network 1 (Largest):**
- 153 accounts (72.5% of all coordinated accounts)
- 987 coordination pairs (88.9% of all pairs)
- Low density (0.049) + High clustering (0.707) = **Mixed structure**
- 13 distinct sub-communities identified
- Interpretation: Organized campaign with role specialization

**Structure Distribution:**
- **Mixed (50%):** 10 networks with high clustering but low density
- **Hierarchical (30%):** 6 hub-and-spoke networks
- **Distributed (20%):** 4 peer-to-peer networks

**Hub Accounts:**
- @Ginger182 (centrality 0.35) — Network 1 organizer
- @PatriotRN7 (centrality 0.28) — High coordinator
- @Sullivan82 (centrality 0.26) — Active coordinator

## ⚙️ Requirements

### System Requirements
- **Python:** 3.12 or higher
- **RAM:** 8GB recommended (handles datasets up to 100k posts)
- **OS:** Windows, macOS, or Linux

### Core Dependencies
```
polars >= 1.9.0          # High-performance dataframes
networkx >= 3.6          # Graph analysis
scikit-learn >= 1.7.2    # K-means clustering
numpy >= 2.3.4           # Numerical operations
plotly >= 6.3.1          # Interactive visualizations
matplotlib >= 3.7        # Static plots
streamlit                # Dashboard (for demo)
```

See [requirements.txt](requirements.txt) for complete dependency list.

## 🎓 Citation

If you use this work in your research, please cite:

```bibtex
@misc{cib-mango-tree-2025,
  author = {Peter Huynh},
  title = {Coordinated Inauthentic Behavior Detection: A Phased Signal Evaluation on Truth Social},
  year = {2025},
  institution = {George Washington University},
  note = {Capstone Research Project}
}
```

## 📈 Project Status

| Component | Status |
|-----------|--------|
| ✅ Burst Detection | Complete |
| ✅ Multi-Signal Coordination Detection | Complete |
| ✅ Network Analysis with NetworkX | Complete |
| ✅ Phased Evaluation (Phase 1-4) | Complete |
| ❌ Behavioral Pattern Detection (Phase 5) | Rejected (high false positive rate) |
| ✅ Interactive Dashboard | Complete |
| 📝 Research Paper | Finalization in progress |

## ⚠️ Important Notes

### Limitations
1. **Dataset:** Single platform (Truth Social), 20-day window — generalizability questions for other platforms/languages
2. **No ground truth:** Lacks labeled data for traditional precision/recall evaluation
3. **UTC normalization:** Timestamp conversion without original timezone metadata limits temporal analysis
4. **False positive risk:** Behavioral patterns conflate coordination with timezone coincidence

### Recommendations
- **Essential:** Implement retweet amplification detection (96.1% coverage)
- **Recommended:** Add ultra-conservative temporal synchronization (2.6% additional coverage)
- **Optional:** Hashtag/content/URL analysis (1.3% combined, higher implementation cost)
- **Exclude:** Behavioral pattern detection (50-70% false positive rate)

### Usage Guidance
This is a **research tool** for detecting coordination patterns. Results should be:
- Manually reviewed before taking action
- Interpreted as probabilistic evidence, not definitive proof
- Used in combination with domain expertise

## 📧 Contact

**Project Author:** Peter Huynh  
**Email:** longh@gwu.edu
**Institution:** George Washington University
**Course:** Capstone Project — Coordinated Inauthentic Behavior Detection  
**Date:** December 2025

## 📄 License

[Choose one:]
- **Academic Research** — All Rights Reserved

---

## 🙏 Acknowledgments

- **Kleinberg's Algorithm:** Jon Kleinberg (2003) for burst detection methodology
- **NetworkX:** Hagberg et al. (2008) for graph analysis tools
- **Newman's Modularity:** M.E.J. Newman (2006) for community detection
- **Coordination Research:** Mannocci et al. (2024), Pacheco et al. (2020) for foundational work

---

**Note:** This research project demonstrates phased evaluation methodology for coordination detection. The 96.1% retweet dominance finding has direct implications for resource allocation in detection systems—prioritize amplification networks over complex multi-signal feature engineering.

**Last Updated:** December 2, 2025

