# IPO Analytics

## Introduction

IPO Analytics is an end-to-end data analytics project on Indian IPOs listed on NSE/BSE from 2019 to 2024. The project starts from curated IPO issue data, enriches it with post-listing return fields, engineers analytical features, and evaluates listing-day performance using exploratory analysis, regression, sampling, clustering, distribution fitting, and time-series summaries.

The analysis focuses on practical financial-market questions:

- How strongly do Grey Market Premium (GMP), subscription demand, issue size, and sector explain listing-day returns?
- Which IPOs, sectors, and time periods produced the strongest or weakest listing outcomes?
- Can IPOs be grouped into distinct risk-return profiles?
- What does the empirical distribution of listing gains imply about upside and downside risk?

## How to Use

### 1. Clone the repository

```bash
git clone https://github.com/aayushtiwari845/IPO-Analytics.git
cd IPO-Analytics
```

### 2. Install dependencies

```bash
pip install pandas numpy scipy scikit-learn statsmodels matplotlib seaborn plotly dash yfinance
```

### 3. Run the interactive dashboard

```bash
python ipo_dashboard.py
```

Open the dashboard at:

```text
http://localhost:8050
```

### 4. Explore the notebook

Open `indian_ipo_analytics (1).ipynb` in Jupyter Notebook, JupyterLab, or VS Code to review the full analysis workflow, including ingestion, preprocessing, plots, model outputs, and statistical results.

### 5. Review generated outputs

The repository includes generated visual outputs:

- `eda_overview.png`
- `exp1_boxplots.png`
- `exp2_regression.png`
- `exp2_simple_regression.png`
- `exp3_sampling.png`
- `exp4_cluster_selection.png`
- `exp4_clusters.png`
- `exp5_distributions.png`

## Results

The notebook analysis covers 63 IPO records across 27 sectors from 2019-04-26 to 2024-11-27. The feature-engineered DataFrame contains 34 columns with no missing values after preprocessing.

Key dataset results:

- Total IPOs analyzed: 63
- Profitable listings: 47 / 63, or 74.6%
- Average listing gain: +33.45%
- Median listing gain: +19.35%
- Standard deviation of listing gain: 42.51%
- Maximum listing gain: Tata Technologies at +140.0%
- Worst listing: Kalyan Jewellers at -14.9%
- Total capital raised: Rs. 219,475 Cr
- Post-listing return coverage after enrichment/fallback processing: 100% for 3M, 6M, and 1Y fields

Regression results:

- Train R2: 0.9922
- Test R2: 0.9834
- Test RMSE: 5.59%
- Test MAE: 4.15%
- 5-fold cross-validation R2: 0.9808 +/- 0.0117
- GMP alone explains 99.0% of listing-day gain variance in the notebook run

Sampling results:

- Population mean listing gain: 33.45%
- Simple Random Sample mean at n=15: 38.28%
- Stratified sample mean at n=15: 40.57%
- Systematic sample mean at n=15: 28.22%

Clustering results:

- Optimal K-Means cluster count: k=2
- Highest silhouette score: 0.4740
- Cluster 0: 33 IPOs, +63.1% average listing gain, 87.2x average subscription, Rs. 1,761 Cr average issue size
- Cluster 1: 30 IPOs, +0.8% average listing gain, 7.3x average subscription, Rs. 5,379 Cr average issue size

Distribution fitting results:

- Student-t distribution fit: KS statistic 0.1512, p-value 0.1011
- Logistic distribution fit: KS statistic 0.1629, p-value 0.0629
- Normal distribution fit was rejected at the 5% level with p-value 0.0054
- Probability of positive listing gain: 0.746
- Probability of gain above 20%: 0.476
- Probability of loss worse than 10%: 0.048

## Inferences

GMP is the strongest observed pre-listing indicator for listing-day performance in this dataset. The regression analysis shows that GMP alone explains nearly all measured variance in listing-day gains for the notebook run, while the full feature model keeps strong generalization with a 0.9834 test R2.

Subscription intensity is also highly informative. IPOs with stronger demand, especially high institutional and non-institutional participation, generally appear in the high-return cluster. The stronger cluster had 87.2x average subscription and +63.1% average listing gain, compared with 7.3x subscription and +0.8% average gain in the weaker cluster.

IPO size appears to moderate listing performance. Larger issues tended to list closer to issue price, while smaller and mid-sized IPOs with strong GMP and subscription demand showed higher upside volatility.

Listing gains are not normally distributed. The distribution has fat tails and positive skew, making Student-t and Logistic fits more suitable than a Normal distribution for modeling listing gain behavior.

## Conclusions

The project demonstrates a complete financial data analytics workflow using Python, Pandas, NumPy, SciPy, Scikit-learn, Statsmodels, Plotly, Dash, and yfinance. It combines data ingestion, feature engineering, exploratory analysis, statistical modeling, machine learning, and dashboarding into a single IPO analysis system.

The analysis concludes that GMP and subscription demand are the most important signals for short-term IPO listing performance in the collected 2019-2024 Indian IPO dataset. High-demand IPOs formed a distinct high-return cluster, while larger and weakly subscribed issues were more likely to produce muted or negative listing outcomes.

Overall, the project shows how structured financial data can be transformed into decision-ready insights through reproducible analytics, model validation, and interactive visualization.
