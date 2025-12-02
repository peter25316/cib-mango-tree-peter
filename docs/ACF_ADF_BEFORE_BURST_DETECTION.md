# ACF and ADF: Pre-Burst Detection Data Quality Checks

**Purpose:** Understand your time series data characteristics before running burst detection algorithms.

---

## What Are ACF and ADF?

### ACF - Autocorrelation Function

**What it is:** Measures how correlated a time series is with itself at different time lags.

**Simple explanation:** "Do posts at hour X predict posts at hour X+k?"

**Formula:**
```
ACF(lag k) = Correlation between X_t and X_{t-k}
```

**What it shows:**
- **High ACF at lag 24:** Posts repeat every 24 hours (daily pattern)
- **High ACF at lag 168:** Posts repeat every 7 days (weekly pattern)
- **Declining ACF:** Pattern gets weaker over time
- **Oscillating ACF:** Regular cycles present

**Example from your data:**
- ACF peaks every ~24 hours → Clear daily posting pattern
- Pattern persists for 7+ days → Consistent behavior

---

### ADF - Augmented Dickey-Fuller Test

**What it is:** Statistical test to determine if your time series is "stationary" or "non-stationary"

**Simple explanation:** "Does the data's behavior stay consistent over time, or does it drift?"

**What it tests:**
- **Null Hypothesis (H0):** Data is non-stationary (has a unit root)
- **Alternative (H1):** Data is stationary

**How to interpret:**
- **p-value < 0.05:** Data is stationary ✅
- **p-value ≥ 0.05:** Data is non-stationary ⚠️

**Your result:** p ≈ 0 (extremely stationary) → Perfect for analysis! ✅

---

## Why They Matter for Burst Detection

### 1. ACF Tells You About Patterns

**What you learn:**
- ✅ **Periodicity:** Is there a 24-hour cycle? Weekly cycle?
- ✅ **Predictability:** Are patterns consistent or random?
- ✅ **Correlation structure:** How far back do patterns persist?

**Why this helps burst detection:**
```
If ACF shows strong 24-hour patterns:
→ Bursts that break this pattern are more significant
→ You know what "normal" looks like
→ Deviations from the pattern indicate coordination
```

**Example:**
- Normal: Posts at 9am every day (high ACF at lag 24)
- Burst: Sudden spike at 3am (breaks the pattern)
- Interpretation: Likely coordinated, not organic

---

### 2. ADF Tells You About Data Stability

**What you learn:**
- ✅ **Stationarity:** Does baseline activity stay constant?
- ✅ **Trend:** Is activity growing/declining over time?
- ✅ **Reliability:** Can you trust the baseline for comparison?

**Why this helps burst detection:**

**Stationary data (p < 0.05):**
```
✅ Stable baseline → Easy to identify bursts
✅ Consistent patterns → Reliable detection
✅ Mean reversion → Bursts are clear deviations
```

**Non-stationary data (p ≥ 0.05):**
```
⚠️ Changing baseline → Harder to define "normal"
⚠️ Trends → What looks like a burst might be gradual growth
⚠️ Drift → Need to adjust detection strategy
```

---

## How They Work Together

### The Workflow:

```
1. Load hourly post data
   ↓
2. Run ADF test
   → Stationary? ✅ Proceed
   → Non-stationary? ⚠️ Consider differencing or detrending
   ↓
3. Calculate ACF
   → Shows 24-hour pattern? ✅ Good - consistent behavior
   → No pattern? ⚠️ Data might be too noisy
   ↓
4. Run Kleinberg burst detection
   → Uses stable baseline (from ADF)
   → Detects deviations from patterns (informed by ACF)
   → Identifies coordination
```

---

## Real Example: Your Truth Social Data

### ADF Result: p ≈ 0 (Extremely Stationary)

**What this means:**
- ✅ Posting rate is stable over 20 days
- ✅ No long-term trends
- ✅ Activity returns to baseline after spikes
- ✅ Perfect for burst detection!

**Implications:**
- Baseline rate is reliable
- Bursts stand out clearly
- Results are interpretable

---

### ACF Result: Peaks Every ~24 Hours

**What this means:**
- ✅ Strong daily pattern (people post at similar times each day)
- ✅ Pattern persists for entire dataset
- ✅ Weekly cycles visible (weekend vs weekday)

**Implications:**
- Can distinguish organic patterns from coordination
- Time-of-day context matters
- Bursts that break daily patterns are suspicious

---

## Does Burst Detection REQUIRE These Tests?

### Short Answer: **NO** ❌

Kleinberg's burst detection algorithm **works without ACF/ADF tests**.

### Why They're Still Useful:

**ACF and ADF are diagnostic tools**, not requirements:

| Aspect | Without ACF/ADF | With ACF/ADF |
|--------|-----------------|---------------|
| **Burst detection** | ✅ Works fine | ✅ Works fine |
| **Understanding results** | ❌ Limited context | ✅ Full context |
| **Confidence in findings** | ⚠️ Uncertain | ✅ High |
| **Interpretation** | ❌ Harder | ✅ Easier |
| **Validation** | ❌ No baseline check | ✅ Validated assumptions |

---

## When to Worry vs When to Celebrate

### 🎉 Great Results (Your Case):

**ADF:** p ≈ 0 (highly stationary)
- ✅ Data is well-behaved
- ✅ Burst detection will work excellently
- ✅ Results are trustworthy

**ACF:** Clear 24-hour peaks
- ✅ Consistent patterns
- ✅ Predictable behavior
- ✅ Easy to spot anomalies

**Recommendation:** Proceed confidently! Your data is perfect for burst detection.

---

### ⚠️ Concerning Results (Hypothetical):

**ADF:** p = 0.30 (non-stationary)
- ⚠️ Data might have trends
- ⚠️ Baseline drifting over time
- ⚠️ Consider differencing: analyze changes in post rate, not absolute counts

**ACF:** No clear pattern, rapid decay
- ⚠️ Random/noisy data
- ⚠️ No consistent behavior
- ⚠️ Bursts might be harder to interpret

**Recommendation:** Proceed with caution, potentially preprocess data.

---

## Practical Takeaways

### What ACF Shows You:

1. **Temporal patterns exist** → Coordination that breaks patterns is suspicious
2. **Pattern strength** → Strong patterns = easier burst detection
3. **Cycle length** → 24-hour cycle common in social media

### What ADF Shows You:

1. **Data is stable** → Reliable baseline for comparison
2. **No problematic trends** → Can trust burst identification
3. **Statistical validity** → Analysis assumptions are met

### Why Both Matter:

- **ACF:** Describes the "shape" of your patterns
- **ADF:** Validates the "stability" of your data
- **Together:** Give you confidence that burst detection results are meaningful

---

## In the CIB Mango Tree Project

### Where Used:

**Streamlit Interactive App:** 
- Shows ACF plot and ADF test results
- Educational - helps users understand their data
- Optional diagnostics before running burst detection

**Main Pipeline:**
- Does NOT run ACF/ADF tests
- Directly runs Kleinberg burst detection
- Assumes data is reasonable

### Why This Design?

**Interactive app = Learning tool**
- Users explore data characteristics
- See why burst detection works
- Build intuition

**Main pipeline = Production tool**
- Efficiency matters
- Assumes validated data
- No unnecessary overhead

---

## Quick Reference

### ACF (Autocorrelation Function)

**Purpose:** Find temporal patterns  
**Interpretation:** Peaks show recurring cycles  
**Your data:** 24-hour pattern (daily posting rhythm)  
**Needed?** No, but very informative  

### ADF (Augmented Dickey-Fuller)

**Purpose:** Test for stationarity  
**Interpretation:** p < 0.05 = stationary (good!)  
**Your data:** p ≈ 0 (extremely stationary)  
**Needed?** No, but validates assumptions  

### Bottom Line:

✅ **Your data passes both checks perfectly**  
✅ **Burst detection will work excellently**  
✅ **Results will be reliable and interpretable**  

These tests don't make burst detection work, but they tell you **why** it works and **how to interpret** the results!

---

**Created:** December 2, 2025  
**For:** CIB Mango Tree Project  
**Purpose:** Educational explanation of pre-detection diagnostics

