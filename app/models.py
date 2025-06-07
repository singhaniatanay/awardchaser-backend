from enum import Enum
from pydantic import BaseModel, Field, constr, conlist, PositiveInt, NonNegativeFloat, UUID4
from typing import List, Dict, Optional


class Currency(str, Enum):
    INR = 'INR'
    USD = 'USD'


class TransactionType(str, Enum):
    UTILITY = 'UTILITY_BILL'
    SHOPPING = 'SHOPPING'
    FUEL = 'FUEL'
    TRAVEL = 'TRAVEL'
    OTHER = 'OTHER'


class GoalType(str, Enum):
    MAX_VALUE = 'MAX_VALUE'  # best ₹ value
    FREE_FLIGHT = 'FREE_FLIGHT'
    FREE_HOTEL = 'FREE_HOTEL'


class Card(BaseModel):
    card_id: UUID4 = Field(description='Internal UUID for the saved card')
    issuer: constr(strip_whitespace=True, to_upper=True) = Field(example='HDFC')
    name: str = Field(example='Infinia')
    reward_currency: str = Field(example='Reward Points')
    reward_balance: NonNegativeFloat = 0


class TransactionRequest(BaseModel):
    user_id: UUID4
    amount: PositiveInt = Field(description='Transaction amount in smallest currency unit, e.g. paise')
    currency: Currency = Currency.INR
    transaction_type: TransactionType
    merchant_name: Optional[str] = None
    cards_to_consider: conlist(UUID4, min_length=1)

    model_config = {
        'json_schema_extra': {
            'example': {
                'user_id': '550e8400-e29b-41d4-a716-446655440000',
                'amount': 500000,  # ₹5000 in paise
                'currency': 'INR',
                'transaction_type': 'SHOPPING',
                'merchant_name': 'Amazon India',
                'cards_to_consider': [
                    '123e4567-e89b-12d3-a456-426614174000',
                    '987fcdeb-51a2-43d1-9f12-123456789abc'
                ]
            }
        }
    }


class RedemptionRequest(BaseModel):
    user_id: UUID4
    goal_type: GoalType
    goal_detail: Optional[str] = Field(example='DEL-JFK one-way business')
    point_balances: Dict[UUID4, NonNegativeFloat] = Field(description='card_id → balance')

    model_config = {
        'json_schema_extra': {
            'example': {
                'user_id': '550e8400-e29b-41d4-a716-446655440000',
                'goal_type': 'FREE_FLIGHT',
                'goal_detail': 'DEL-JFK one-way business',
                'point_balances': {
                    '123e4567-e89b-12d3-a456-426614174000': 75000.0,
                    '987fcdeb-51a2-43d1-9f12-123456789abc': 125000.0
                }
            }
        }
    }


class RouteStep(BaseModel):
    step: PositiveInt
    platform: str = Field(example='Gyftr')
    action: str = Field(example='Buy Amazon Pay voucher')
    card_used: UUID4
    expected_reward_value: NonNegativeFloat = 0  # ₹ or mile equivalent
    notes: Optional[str]


class OptimizationResponse(BaseModel):
    best_route: List[RouteStep]
    total_reward_value: NonNegativeFloat = Field(description='₹ or mile equivalent')
    estimated_roi_percent: Optional[float]
    justification: str

    model_config = {
        'json_schema_extra': {
            'example': {
                'best_route': [
                    {
                        'step': 1,
                        'platform': 'Gyftr',
                        'action': 'Buy Amazon Pay voucher worth ₹5000',
                        'card_used': '123e4567-e89b-12d3-a456-426614174000',
                        'expected_reward_value': 250.0,
                        'notes': 'HDFC Infinia gives 5% reward rate on Gyftr'
                    },
                    {
                        'step': 2,
                        'platform': 'Amazon',
                        'action': 'Use voucher for shopping',
                        'card_used': '123e4567-e89b-12d3-a456-426614174000',
                        'expected_reward_value': 0.0,
                        'notes': 'No additional rewards for voucher usage'
                    }
                ],
                'total_reward_value': 250.0,
                'estimated_roi_percent': 5.0,
                'justification': 'Using HDFC Infinia via Gyftr maximizes reward rate at 5% for this shopping transaction'
            }
        }
    } 