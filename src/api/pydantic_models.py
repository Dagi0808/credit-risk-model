from pydantic import BaseModel

class PredictionRequest(BaseModel):
    total_transaction_amount: float
    avg_transaction_amount: float
    transaction_count: int
    std_transaction_amount: float
    max_transaction_amount: float
    min_transaction_amount: float
    avg_transaction_hour: float
    unique_months_active: int
    Recency: int
    Frequency: int
    Monetary: float


class PredictionResponse(BaseModel):
    risk_probability: float
    is_high_risk: bool
    message: str