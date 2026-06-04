Credit Risk Modeling Project
End-to-End Credit Risk Prediction System using Proxy Target & MLflow

Project Overview
This project builds a complete credit risk scoring system from scratch, following industry best practices (Basel II principles). Since the dataset does not contain actual default labels, a proxy target (is_high_risk) was engineered using RFM analysis and K-Means clustering.
The solution includes:

Feature engineering pipeline
Proxy target creation
Model training with experiment tracking
Model deployment via FastAPI + Docker
Automated CI/CD pipeline


Project Structure
Bashcredit-risk-model/
├── src/
│   ├── data_processing.py          # Task 3 + Task 4
│   ├── train.py                    # Task 5
│   └── api/
│       ├── main.py
│       └── pydantic_models.py
├── tests/
│   └── test_data_processing.py
├── notebooks/
│   └── eda.ipynb
├── data/
│   ├── raw/
│   └── processed/
├── .github/workflows/ci.yml        # CI/CD Pipeline
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md

Key Features

Proxy Target Engineering: RFM + K-Means clustering to identify high-risk customers
Feature Engineering: Aggregated transaction features, datetime components, scaling
Modeling: Logistic Regression, Decision Tree, Random Forest with MLflow tracking
Best Model: Random Forest (F1 Score: ~0.9987, ROC-AUC: ~1.0000)
Deployment: FastAPI REST API with Docker containerization
CI/CD: GitHub Actions with linting (flake8) and testing (pytest)


How to Run the Project
1. Install Dependencies
Bashpip install -r requirements.txt
2. Process Data (Task 3 + 4)
Bashpython src/data_processing.py
3. Train Models (Task 5)
Bashmlflow ui   # Open in browser: http://127.0.0.1:5000
python src/train.py
4. Run the API (Task 6)
Option A: Using Docker (Recommended)
Bashdocker-compose up --build
Option B: Locally
Bashuvicorn src.api.main:app --reload --port 8000

API Endpoints

GET / → Health check
POST /predict → Predict credit risk probability

Example Request Body:
JSON{
  "total_transaction_amount": 12500.0,
  "avg_transaction_amount": 520.83,
  "transaction_count": 24,
  "std_transaction_amount": 340.5,
  "max_transaction_amount": 1500.0,
  "min_transaction_amount": 50.0,
  "avg_transaction_hour": 14.2,
  "unique_months_active": 6,
  "Recency": 12,
  "Frequency": 24,
  "Monetary": 12500.0
}

Business Understanding Highlights

Basel II Compliance: Emphasis on model interpretability and documentation
Proxy Target: Used due to absence of default labels — introduces risk of mislabeling
Model Trade-off: Chose Random Forest for performance while maintaining reasonable explainability


Future Improvements

Add SHAP/LIME explanations for model interpretability
Implement WoE + IV feature selection
Add A/B testing for model versions
Deploy on cloud (AWS/Heroku)


Team / Author

Student:Dagmawit Dagne
Project: Credit Risk Modeling (10 Academy)
Completion Date: June 2026