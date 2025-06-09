from enum import Enum
from pydantic import BaseModel, Field, constr, conlist, PositiveInt, NonNegativeFloat, UUID4, ConfigDict
from typing import List, Dict, Optional, Union


class Currency(str, Enum):
    """Supported currencies for transactions."""
    INR = "INR"
    USD = "USD"
    EUR = "EUR"


class TransactionType(str, Enum):
    """Types of credit card transactions."""
    ONLINE_SHOPPING = "online_shopping"
    GROCERY = "grocery"
    FUEL = "fuel"
    DINING = "dining"
    TRAVEL = "travel"
    UTILITIES = "utilities"
    ENTERTAINMENT = "entertainment"
    INSURANCE = "insurance"
    EDUCATION = "education"
    HEALTHCARE = "healthcare"
    OTHER = "other"


class GoalType(str, Enum):
    """Types of redemption goals."""
    FREE_FLIGHT = "free_flight"
    FREE_HOTEL = "free_hotel"
    MAX_VALUE = "max_value"


class Card(BaseModel):
    card_id: UUID4 = Field(description='Internal UUID for the saved card')
    issuer: constr(strip_whitespace=True, to_upper=True) = Field(example='HDFC')
    name: str = Field(example='Infinia')
    reward_currency: str = Field(example='Reward Points')
    reward_balance: NonNegativeFloat = 0


class TransactionRequest(BaseModel):
    """Request model for transaction optimization."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "amount": 25000,  # ₹250.00 in paise
                "currency": "INR",
                "transaction_type": "online_shopping",
                "merchant_name": "Amazon India",
                "cards_to_consider": ["hdfc_regalia", "sbi_cashback", "icici_amazon_pay"]
            }
        }
    )
    
    amount: int = Field(..., description="Transaction amount in paise (100 paise = ₹1)", gt=0)
    currency: Currency = Field(default=Currency.INR, description="Transaction currency")
    transaction_type: TransactionType = Field(..., description="Type of transaction")
    merchant_name: Optional[str] = Field(None, description="Name of the merchant (optional)")
    cards_to_consider: List[str] = Field(..., description="List of credit card IDs to consider", min_length=1)


class RedemptionRequest(BaseModel):
    """Request model for redemption optimization."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "goal_type": "free_flight",
                "goal_detail": "Delhi to London business class",
                "point_balances": {
                    "amex_platinum": 125000,
                    "chase_sapphire": 85000,
                    "hdfc_diners": 45000
                }
            }
        }
    )
    
    goal_type: GoalType = Field(..., description="Type of redemption goal")
    goal_detail: Optional[str] = Field(None, description="Specific details about the goal")
    point_balances: Dict[str, int] = Field(
        ..., 
        description="Dictionary mapping card IDs to point balances",
        min_length=1
    )


class RouteStep(BaseModel):
    """Individual step in an optimization route."""
    
    step: int = Field(..., description="Step number in the route")
    platform: str = Field(..., description="Platform or service to use")
    action: str = Field(..., description="Action to take in this step") 
    card_used: Optional[str] = Field(None, description="Credit card used for this step")
    expected_reward_value: float = Field(0.0, description="Expected reward value in ₹")
    notes: Optional[str] = Field(None, description="Additional notes or considerations")


class OptimizationResponse(BaseModel):
    """Response model for both transaction and redemption optimization."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "best_route": [
                    {
                        "step": 1,
                        "platform": "Amazon Pay ICICI Card", 
                        "action": "Use for 5% cashback on Amazon purchases",
                        "card_used": "icici_amazon_pay",
                        "expected_reward_value": 12.50,
                        "notes": "Maximum cashback for Amazon transactions"
                    }
                ],
                "total_reward_value": 12.50,
                "estimated_roi_percent": 5.0,
                "justification": "Detailed explanation of why this is the optimal strategy..."
            }
        }
    )
    
    best_route: List[RouteStep] = Field(..., description="Step-by-step optimization route")
    total_reward_value: float = Field(..., description="Total expected reward value in ₹")
    estimated_roi_percent: Optional[float] = Field(None, description="Estimated ROI percentage")
    justification: str = Field(..., description="Detailed explanation of the recommendation")


class ScrapeRequest(BaseModel):
    """Request model for web scraping."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "url": "https://example.com/credit-cards",
                "selectors": [
                    "h1.title",
                    ".reward-rate",
                    "#annual-fee"
                ]
            }
        }
    )
    
    url: str = Field(..., description="URL to scrape", min_length=1)
    selectors: List[str] = Field(..., description="CSS selectors to extract", min_length=1)


class ScrapeResponse(BaseModel):
    """Response model for web scraping."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "url": "https://example.com/credit-cards",
                "data": {
                    "h1.title": "Premium Rewards Card",
                    ".reward-rate": "2% on all purchases",
                    "#annual-fee": "₹2,999 + GST"
                },
                "selectors_found": 3,
                "cached": False
            }
        }
    )
    
    url: str = Field(..., description="URL that was scraped")
    data: Dict[str, str] = Field(..., description="Mapping of selectors to extracted text")
    selectors_found: int = Field(..., description="Number of selectors that returned data")
    cached: bool = Field(..., description="Whether the result was served from cache") 