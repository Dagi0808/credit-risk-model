# Credit Risk Probability Model for Alternative Data

An end-to-end credit risk scoring system built for Bati Bank's 
buy-now-pay-later service, using eCommerce transaction data.

---

## Credit Scoring Business Understanding

### 1. How does Basel II influence the need for interpretability?

The Basel II Capital Accord requires banks to hold capital reserves 
proportional to their credit risk. To satisfy regulators, a model must 
be transparent and auditable — not just accurate. Regulators can ask 
"why was this customer rejected?" and the bank must answer clearly.

This is why interpretable models like Logistic Regression with Weight 
of Evidence (WoE) encoding are standard in credit scoring. Every 
coefficient maps directly to a scoreable customer attribute. Black-box 
models like Gradient Boosting require additional validation effort and 
post-hoc tools (e.g., SHAP values) to meet the same regulatory standard.

In practice, Basel II turns model interpretability from a nice-to-have 
into a compliance requirement.

### 2. Why is a proxy variable necessary, and what risks does it introduce?

The Xente eCommerce dataset contains no historical loan repayment data — 
because the buy-now-pay-later product is new. Without a default label, 
we cannot train a supervised model directly.

A proxy variable is a substitute signal that correlates with the 
unobservable outcome we care about (default). We construct this using 
RFM analysis: customers who are disengaged (low frequency, low spending, 
long since last transaction) are labeled as high-risk proxies.

**Business risks this introduces:**
- **Label noise**: A disengaged customer may have simply churned, not 
  defaulted. The proxy conflates two different behaviors.
- **Concept drift**: eCommerce engagement may not generalize to loan 
  repayment behavior.
- **Regulatory exposure**: If the proxy correlates with demographics 
  (e.g., geography), it may introduce discriminatory lending outcomes.
- **Feedback loops**: Denying credit to proxy-high-risk customers 
  prevents them from ever building a repayment history.

These risks must be disclosed in model documentation and monitored 
continuously after deployment.

### 3. Trade-offs: Logistic Regression vs Gradient Boosting

| Dimension | Logistic Regression + WoE | Gradient Boosting |
|-----------|--------------------------|-------------------|
| Interpretability | High — coefficients are explainable | Low — needs SHAP |
| Regulatory acceptance | High — standard in scorecards | Requires extra validation |
| Predictive performance | Moderate | High |
| Overfitting risk | Low | Higher without tuning |
| Scorecard conversion | Straightforward | Complex |

**In practice:** We train both. Logistic Regression is the auditable 
baseline. Gradient Boosting is the performance benchmark. The final 
choice balances AUC score against interpretability requirements.

---

## Project Structure

\```
credit-risk-model/
├── .github/workflows/ci.yml
├── data/                    ← gitignored
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── data_processing.py
│   ├── train.py
│   ├── predict.py
│   └── api/
│       ├── main.py
│       └── pydantic_models.py
├── tests/
│   └── test_data_processing.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
\```

## Setup

\```bash
pip install -r requirements.txt
python src/data_processing.py
python src/train.py
uvicorn src.api.main:app --reload
\```

## Team
- Kerod
- Mahbubah  
- Feven