# Online Retail -- Profit Leakage Analytics

A data analytics project investigating how product returns silently erode profitability in an online retail business, even as gross revenue appears healthy. Built as part of a B.Tech Data Analytics Reverse Learning Project.

---

## Problem Statement

An online retailer shows **GBP 8.91 million** in gross revenue over a 12-month period, yet an estimated **GBP 611,000** leaks out through product returns -- consuming **17.1%** of gross profit. This project identifies where profit leakage occurs across products, customers, geographies, and price segments, and recommends targeted strategies to reduce return losses.

---

## Dataset

| Detail | Value |
|---|---|
| **Source** | UCI Machine Learning Repository / Kaggle |
| **File** | `Online Retail.xlsx` |
| **Records** | 541,909 (raw) / 406,789 (cleaned) |
| **Period** | December 2010 -- December 2011 |
| **Countries** | 38 |
| **Customers** | 4,372 |
| **Products** | 4,070 |

**Columns:** InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

Returns are identified by: cancelled invoice codes (prefix `C`) or negative quantities.

---

## Dashboard

Interactive Streamlit dashboard with five analytical sections:

### 1. Executive Profit Leakage Summary
- 8 KPI metric cards (gross revenue, return losses, net revenue, return rate, gross profit, net profit, profit erosion, transaction return rate)
- Monthly revenue vs return losses trend (dual-axis)
- Revenue waterfall chart (Gross Revenue → COGS → Gross Profit → Return Losses → Net Profit)
- Country-level return rate comparison (top 15)

### 2. Product & Category Return Intelligence
- Top 15 products by absolute return loss
- Category return rates (top 20 by volume)
- Price band vs return risk analysis (dual-axis)
- Pareto (80/20) return concentration analysis

### 3. Customer Behaviour & Policy Risk
- Top 15 customers by return loss
- Customer risk segmentation (High / Medium / Low / No Returns)
- Monthly cohort retention heatmap
- Customer segmentation scatter (Spend vs Return Loss)
- Temporal return patterns (day-of-week and hour-of-day)

### 4. Profit Lift Simulator
- Interactive sliders: return reduction %, gross margin assumption, cost multiplier
- Current vs projected scenario comparison
- Profit sensitivity curve across all reduction levels

### 5. Recommendations & Ethics
- Prioritised action tracker (10 data-driven recommendations)
- Ethical safeguards (customer fairness, data privacy, algorithmic bias, transparency, proportionality)
- KPI definitions and assumptions reference

---

## Key Findings

| # | Insight |
|---|---|
| 1 | GBP 611K in return losses erodes 17.1% of estimated gross profit |
| 2 | Only **67 products** (3.4% of returned items) account for **80%** of all return losses |
| 3 | High-price items (GBP 50-1000) show disproportionately elevated return rates |
| 4 | A small high-risk customer segment drives outsized return losses |
| 5 | Customer retention drops below 15% by month six |
| 6 | Several international markets show 2-3x higher return rates than the UK baseline |
| 7 | Post-holiday months show disproportionate return spikes |
| 8 | A 25% reduction in return losses would recover approx. GBP 176K in net profit |
| 9 | Return processing overhead (at 1.15x multiplier) adds GBP 92K in hidden costs |
| 10 | Low-value items drive return volume; high-value items drive return monetary loss |

---

## Profit Assumptions

The dataset contains no cost data. The following assumptions are used and clearly parameterised in the dashboard simulator:

| Parameter | Value | Basis |
|---|---|---|
| COGS | 60% of revenue | Industry-standard e-retail benchmark |
| Gross Profit Margin | 40% | Derived from COGS assumption |
| Return Processing Cost Multiplier | 1.15x | Handling, shipping, restocking overhead |

---

## Data Cleaning

| Step | Action | Records Affected |
|---|---|---|
| 1 | Drop null `CustomerID` rows | 135,080 |
| 2 | Drop null `Description` rows | 1,454 |
| 3 | Drop zero/negative `UnitPrice` rows | ~40 |
| 4 | Strip whitespace from strings | All |
| 5 | Cast `CustomerID` to integer | All |
| 6 | Engineer features: `is_return`, `Revenue`, `YearMonth`, `PriceBand`, `Category` | All |

**Final dataset:** 406,789 rows (397,884 sales + 8,905 returns)

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3 |
| Dashboard | Streamlit |
| Visualisation | Plotly |
| Data Processing | Pandas, NumPy |
| Data Source | Excel (openpyxl) |

---

## Project Structure

```
Online_Retail/
├── app.py                 # Streamlit dashboard (main application)
├── Online Retail.xlsx     # Dataset
├── Academic_Report.txt    # Full academic report
└── README.md              # This file
```

---

## How to Run

```bash
# Install dependencies
pip install streamlit pandas numpy plotly openpyxl

# Launch the dashboard
streamlit run app.py
```

The dashboard opens at `http://localhost:8501` in your browser.

---

## Business Recommendations

| Priority | Issue | Action | Expected Impact |
|---|---|---|---|
| Critical | 67 products drive 80% of return losses | Quality audit + enhanced descriptions for these SKUs | Recover up to 60% of concentrated losses |
| Critical | High-risk customers generate outsized returns | Tiered return policy with transparent appeals | Reduce segment losses by 30-40% |
| High | High-price band has elevated return risk | Size guides, AR previews, video demos | Lower returns by 15-20% |
| High | Certain countries show 2-3x return rates | Localise descriptions and shipping expectations | Improve rates by 10-15% |
| High | Post-holiday return spikes | Extended pre-dispatch quality checks | Mitigate spikes by 10-20% |
| Medium | Subjective-attribute categories return often | AR try-on tools and review integration | Reduce by 12-18% |
| Medium | No actual cost data | Capture true COGS and return processing costs | Enable exact ROI calculations |
| Medium | 25% of transactions lack CustomerID | Mandate ID capture at all touchpoints | Expand analysable base by 25% |
| Low | Day/hour patterns in return rates | Align proactive outreach with high-return windows | 5-8% reduction |
| Low | New customers have higher return propensity | Post-first-purchase onboarding emails | 10-15% reduction |

---

## Ethical Considerations

1. **Customer Fairness** -- Policies must not penalise legitimate quality complaints; transparent appeals required
2. **Data Privacy** -- Customer profiling must comply with GDPR; identifiers anonymised in outputs
3. **Algorithmic Bias** -- Risk segmentation audited to prevent geographic or demographic disadvantage
4. **Transparency** -- Policy changes communicated proactively, never applied retroactively
5. **Proportionality** -- Interventions graduated to match identified risk level

---

## Limitations

- Profit figures are estimated (no actual cost data in dataset)
- 24.9% of records excluded due to missing CustomerID
- Product categorisation uses a crude first-word proxy (no official taxonomy)
- No return-reason data available for root-cause analysis
- Single-year observation window limits trend validation
- Return processing costs are assumed, not measured

---

## Future Work

- Integrate actual COGS and fulfilment cost data
- Add return-reason classification taxonomy
- Build predictive return model (ML-based)
- Implement A/B testing framework for interventions
- Deploy to cloud with live database connection
- Apply NLP to product descriptions for automated categorisation
- Incorporate Customer Lifetime Value (CLV) into risk segmentation
- Extend to multi-year longitudinal analysis

---

## Author

**Astitva** -- B.Tech Data Analytics

---

*Data Source: UCI Machine Learning Repository / Kaggle*
