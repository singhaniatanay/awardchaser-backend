"""
Transaction Optimization API Router

Provides endpoints for optimizing credit card transactions using AI-powered recommendations.
"""

import logging
from typing import List
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

from app.models import TransactionRequest, RedemptionRequest, OptimizationResponse, RouteStep
from app.core.retrieval import get_vector_client
from app.core.llm import ask_llm


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["optimization"])


def create_search_query(request: TransactionRequest) -> str:
    """
    Create an intelligent search query based on transaction request.
    
    Args:
        request: Transaction request with details
        
    Returns:
        str: Optimized search query for vector database
    """
    # Convert amount from paise to rupees for better readability
    amount_rupees = request.amount / 100
    
    # Build search components
    search_parts = []
    
    # Add transaction type
    search_parts.append(f"{request.transaction_type.value.replace('_', ' ').lower()}")
    
    # Add amount context
    if amount_rupees >= 100000:  # 1 lakh+
        search_parts.append("high value transaction")
    elif amount_rupees >= 50000:  # 50k+
        search_parts.append("large transaction")
    elif amount_rupees >= 10000:  # 10k+
        search_parts.append("medium transaction")
    else:
        search_parts.append("small transaction")
    
    # Add merchant context if provided
    if request.merchant_name:
        search_parts.append(f"{request.merchant_name}")
    
    # Add reward optimization keywords
    search_parts.extend([
        "credit card rewards",
        "cashback",
        "points",
        "miles",
        "reward rate",
        "bonus category"
    ])
    
    # Create the search query
    search_query = " ".join(search_parts)
    
    logger.debug(f"Created search query: '{search_query}' for transaction: {request.transaction_type.value}, ₹{amount_rupees}")
    return search_query


def create_redemption_search_query(request: RedemptionRequest) -> str:
    """
    Create an intelligent search query based on redemption request.
    
    Args:
        request: Redemption request with goal and balances
        
    Returns:
        str: Optimized search query for vector database
    """
    # Build search components
    search_parts = []
    
    # Add goal type
    goal_type_text = request.goal_type.value.replace('_', ' ').lower()
    search_parts.append(goal_type_text)
    
    # Add goal detail context if provided
    if request.goal_detail:
        search_parts.append(request.goal_detail.lower())
        
        # Extract travel context from goal detail
        if any(keyword in request.goal_detail.lower() for keyword in ['flight', 'airline', 'air']):
            search_parts.extend(["airline", "miles", "transfer partners", "aviation"])
        if any(keyword in request.goal_detail.lower() for keyword in ['hotel', 'stay', 'accommodation']):
            search_parts.extend(["hotel", "points", "loyalty program"])
        if any(keyword in request.goal_detail.lower() for keyword in ['business', 'first class']):
            search_parts.append("premium travel")
    
    # Add redemption-specific keywords based on goal type
    if request.goal_type.value == "FREE_FLIGHT":
        search_parts.extend([
            "airline miles",
            "transfer partners", 
            "award flights",
            "redemption rates",
            "airline alliances",
            "sweet spots"
        ])
    elif request.goal_type.value == "FREE_HOTEL":
        search_parts.extend([
            "hotel points",
            "hotel loyalty",
            "transfer ratios",
            "award nights",
            "hotel chains"
        ])
    else:  # MAX_VALUE
        search_parts.extend([
            "points value",
            "transfer ratios",
            "redemption options",
            "best value",
            "cash equivalent"
        ])
    
    # Add general redemption keywords
    search_parts.extend([
        "point transfers",
        "redemption strategy",
        "transfer partners",
        "point pooling",
        "conversion rates"
    ])
    
    # Calculate total points for context
    total_points = sum(request.point_balances.values())
    if total_points >= 500000:
        search_parts.append("high value redemption")
    elif total_points >= 100000:
        search_parts.append("premium redemption")
    else:
        search_parts.append("standard redemption")
    
    # Create the search query
    search_query = " ".join(search_parts)
    
    logger.debug(f"Created redemption search query: '{search_query}' for goal: {request.goal_type.value}")
    return search_query


def create_optimization_prompt(request: TransactionRequest) -> str:
    """
    Create the user prompt for LLM optimization.
    
    Args:
        request: Transaction request with details
        
    Returns:
        str: Formatted prompt for LLM
    """
    # Convert amount from paise to rupees
    amount_rupees = request.amount / 100
    
    # Format transaction details
    transaction_details = f"""
Transaction Type: {request.transaction_type.value.replace('_', ' ').title()}
Amount: ₹{amount_rupees:,.2f} ({request.currency.value})
Merchant: {request.merchant_name or 'Not specified'}
Number of Cards Available: {len(request.cards_to_consider)}
"""
    
    prompt = f"""
Please analyze this transaction and recommend the optimal credit card strategy:

{transaction_details.strip()}

Based on the available credit card information and current market conditions, please provide:

1. **Recommended Strategy**: Which specific approach will maximize rewards for this transaction
2. **Expected Reward Value**: Estimated cashback/points/miles value in ₹
3. **Step-by-Step Route**: Detailed steps to execute this strategy
4. **Alternative Options**: Any backup strategies if the primary option isn't available
5. **Justification**: Clear reasoning for why this is the optimal choice

Focus on practical, actionable advice that maximizes the reward-to-effort ratio for this specific transaction.
"""
    
    return prompt


def create_redemption_prompt(request: RedemptionRequest) -> str:
    """
    Create the user prompt for LLM redemption optimization.
    
    Args:
        request: Redemption request with goal and balances
        
    Returns:
        str: Formatted prompt for LLM
    """
    # Format point balances
    balances_text = []
    total_points = 0
    for card_id, balance in request.point_balances.items():
        balances_text.append(f"Card {str(card_id)[-8:]}: {balance:,.0f} points")
        total_points += balance
    
    balances_summary = "\n".join(balances_text)
    
    # Format redemption details
    redemption_details = f"""
Goal Type: {request.goal_type.value.replace('_', ' ').title()}
Goal Detail: {request.goal_detail or 'Not specified'}
Total Points Available: {total_points:,.0f}
Number of Card Programs: {len(request.point_balances)}

Point Balances:
{balances_summary}
"""
    
    prompt = f"""
Please analyze these points balances and suggest the best transfer route to achieve the specified goal:

{redemption_details.strip()}

Based on current transfer partners, redemption rates, and market conditions, please provide:

1. **Recommended Transfer Route**: Which points to transfer and where for maximum value
2. **Expected Redemption Value**: Estimated value of the redemption in ₹
3. **Step-by-Step Process**: Detailed steps to execute this transfer strategy
4. **Alternative Routes**: Backup options if primary transfers aren't available
5. **Timing Considerations**: Any transfer bonuses or promotional rates to consider
6. **Justification**: Clear reasoning for why this route provides the best value

Focus on maximizing the value per point while ensuring the goal is achievable with current balances.
"""
    
    return prompt


def parse_llm_response_to_optimization(
    llm_response: str,
    request_cards: List[str]
) -> OptimizationResponse:
    """
    Parse LLM response into structured OptimizationResponse.
    
    Args:
        llm_response: Raw response from LLM
        request_cards: List of card IDs from the request
        
    Returns:
        OptimizationResponse: Structured response object
    """
    try:
        # Extract key information from LLM response
        # This is a simplified parser - in production, you might want more sophisticated parsing
        
        # Default values
        total_reward_value = 0.0
        estimated_roi_percent = None
        
        # Try to extract reward value
        lines = llm_response.split('\n')
        for line in lines:
            line_lower = line.lower()
            if 'reward' in line_lower and '₹' in line:
                # Try to extract numeric value
                import re
                numbers = re.findall(r'₹([\d,]+(?:\.\d{2})?)', line)
                if numbers:
                    try:
                        total_reward_value = float(numbers[0].replace(',', ''))
                    except ValueError:
                        pass
            
            # Try to extract ROI percentage
            if ('roi' in line_lower or 'return' in line_lower) and '%' in line:
                percentages = re.findall(r'([\d.]+)%', line)
                if percentages:
                    try:
                        estimated_roi_percent = float(percentages[0])
                    except ValueError:
                        pass
        
        # Create a simplified route step
        # In a real implementation, you'd parse the LLM response more carefully
        route_steps = [
            RouteStep(
                step=1,
                platform="Recommended Platform",
                action="Use optimal strategy as described",
                card_used=request_cards[0] if request_cards else "N/A",
                expected_reward_value=total_reward_value,
                notes="See detailed analysis in justification"
            )
        ]
        
        return OptimizationResponse(
            best_route=route_steps,
            total_reward_value=total_reward_value,
            estimated_roi_percent=estimated_roi_percent,
            justification=llm_response
        )
        
    except Exception as e:
        logger.error(f"Error parsing LLM response: {e}")
        # Return a fallback response
        return OptimizationResponse(
            best_route=[
                RouteStep(
                    step=1,
                    platform="Analysis Available",
                    action="Review detailed recommendation",
                    card_used=request_cards[0] if request_cards else "N/A",
                    expected_reward_value=0.0,
                    notes="See justification for full analysis"
                )
            ],
            total_reward_value=0.0,
            estimated_roi_percent=None,
            justification=llm_response
        )


@router.post("/optimize_transaction", response_model=OptimizationResponse)
async def optimize_transaction(request: TransactionRequest) -> OptimizationResponse:
    """
    Optimize a credit card transaction to maximize rewards.
    
    This endpoint analyzes the transaction details, searches for relevant 
    credit card information, and uses AI to recommend the optimal strategy.
    
    Args:
        request: Transaction details including amount, type, and available cards
        
    Returns:
        OptimizationResponse: Detailed optimization strategy with expected rewards
        
    Raises:
        HTTPException: If optimization fails due to system errors
    """
    try:
        logger.info(f"Optimizing transaction: {request.transaction_type.value}, "
                   f"₹{request.amount/100:,.2f}, {len(request.cards_to_consider)} cards")
        
        # Validate request
        if not request.cards_to_consider:
            raise HTTPException(
                status_code=400,
                detail="At least one card must be provided for optimization"
            )
        
        # Create intelligent search query
        search_query = create_search_query(request)
        
        # Retrieve relevant documents
        try:
            vector_client = get_vector_client()
            relevant_docs = vector_client.query(search_query, k=5)
            logger.info(f"Retrieved {len(relevant_docs)} relevant documents")
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            # Continue without documents rather than failing completely
            relevant_docs = []
            logger.warning("Continuing optimization without document context")
        
        # Create system prompt
        system_prompt = """You are AwardChaser, an expert AI advisor specializing in credit card reward optimization. 

Your expertise includes:
- Deep knowledge of credit card reward structures, bonus categories, and annual benefits
- Understanding of merchant category codes (MCCs) and how they affect rewards
- Awareness of seasonal promotions, limited-time offers, and bonus activations
- Knowledge of redemption strategies and transfer partners
- Experience with manufactured spending and reward maximization techniques

Analyze the transaction and available context to propose the best route for maximizing rewards. Consider all aspects including:
- Base reward rates vs bonus category rates
- Annual fee vs benefit calculations  
- Transfer partner value propositions
- Timing considerations for promotions
- Risk factors and backup strategies

Provide specific, actionable recommendations with clear reasoning."""
        
        # Create user prompt
        user_prompt = create_optimization_prompt(request)
        
        # Get LLM recommendation
        try:
            llm_response = ask_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                docs=relevant_docs,
                temperature=0.3  # Lower temperature for more consistent financial advice
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to generate optimization recommendation"
            )
        
        # Parse response into structured format
        optimization_result = parse_llm_response_to_optimization(
            llm_response, 
            [str(card_id) for card_id in request.cards_to_consider]
        )
        
        logger.info(f"Transaction optimization complete: ₹{optimization_result.total_reward_value} potential value")
        return optimization_result
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error in optimize_transaction: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during transaction optimization"
        )


@router.post("/optimize_redemption", response_model=OptimizationResponse)
async def optimize_redemption(request: RedemptionRequest) -> OptimizationResponse:
    """
    Optimize point redemption to maximize value for a specific goal.
    
    This endpoint analyzes point balances and redemption goals, searches for relevant 
    transfer partner information, and uses AI to recommend the optimal redemption strategy.
    
    Args:
        request: Redemption details including point balances and goal
        
    Returns:
        OptimizationResponse: Detailed redemption strategy with expected value
        
    Raises:
        HTTPException: If optimization fails due to system errors
    """
    try:
        total_points = sum(request.point_balances.values())
        logger.info(f"Optimizing redemption: {request.goal_type.value}, "
                   f"{total_points:,.0f} total points, {len(request.point_balances)} card programs")
        
        # Validate request
        if not request.point_balances:
            raise HTTPException(
                status_code=400,
                detail="At least one point balance must be provided for redemption optimization"
            )
        
        if total_points <= 0:
            raise HTTPException(
                status_code=400,
                detail="Total point balance must be greater than zero"
            )
        
        # Create intelligent search query for redemption
        search_query = create_redemption_search_query(request)
        
        # Retrieve relevant documents
        try:
            vector_client = get_vector_client()
            relevant_docs = vector_client.query(search_query, k=5)
            logger.info(f"Retrieved {len(relevant_docs)} relevant redemption documents")
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            # Continue without documents rather than failing completely
            relevant_docs = []
            logger.warning("Continuing redemption optimization without document context")
        
        # Create system prompt for redemption optimization
        system_prompt = """You are AwardChaser, an expert AI advisor specializing in credit card points and miles redemption optimization.

Your expertise includes:
- Comprehensive knowledge of airline and hotel transfer partners across all major credit card programs
- Real-time awareness of transfer bonuses, promotional rates, and sweet spot redemptions
- Deep understanding of award charts, dynamic pricing, and redemption values
- Experience with complex multi-program transfers and point pooling strategies
- Knowledge of seasonal availability, routing rules, and booking strategies

Suggest the best transfer route to achieve the specified redemption goal. Consider all aspects including:
- Current transfer ratios and any promotional bonuses
- Transfer partner award charts and sweet spots
- Seasonal availability and booking windows
- Alternative routing options and flexibility
- Risk factors like devaluations or partner changes

Provide specific, actionable transfer recommendations with clear value calculations."""
        
        # Create user prompt for redemption
        user_prompt = create_redemption_prompt(request)
        
        # Get LLM recommendation
        try:
            llm_response = ask_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                docs=relevant_docs,
                temperature=0.3  # Lower temperature for more consistent financial advice
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to generate redemption recommendation"
            )
        
        # Parse response into structured format
        optimization_result = parse_llm_response_to_optimization(
            llm_response,
            [str(card_id) for card_id in request.point_balances.keys()]
        )
        
        logger.info(f"Redemption optimization complete: ₹{optimization_result.total_reward_value} potential value")
        return optimization_result
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error in optimize_redemption: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during redemption optimization"
        )


@router.get("/optimization/health")
async def optimization_health_check():
    """
    Health check endpoint for the optimization service.
    
    Returns:
        dict: Service status and component health
    """
    try:
        # Test vector client
        vector_client = get_vector_client()
        collection_info = vector_client.get_collection_info()
        
        # Test LLM (simple test)
        test_response = ask_llm(
            system_prompt="You are a test assistant.",
            user_prompt="Respond with 'OK' if you can process this request.",
            docs=None,
            max_tokens=10
        )
        
        return {
            "status": "healthy",
            "vector_db": {
                "status": "connected",
                "collection": collection_info["name"],
                "documents": collection_info["points_count"]
            },
            "llm": {
                "status": "connected" if "OK" in test_response.upper() else "degraded",
                "model": "gpt-4o-mini"
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        ) 