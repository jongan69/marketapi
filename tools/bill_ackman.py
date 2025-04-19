def analyze_business_quality(metrics: list, financial_line_items: list) -> dict:
    """
    Analyze whether the company has a high-quality business with stable or growing cash flows,
    durable competitive advantages (moats), and potential for long-term growth.
    Also tries to infer brand strength if intangible_assets data is present (optional).
    """
    score = 0
    details = []
    
    if not metrics or not financial_line_items:
        return {
            "score": 0,
            "details": "Insufficient data to analyze business quality"
        }
    
    # 1. Multi-period revenue growth analysis
    revenues = [item.get('revenue') for item in financial_line_items if item.get('revenue') is not None]
    if len(revenues) >= 2:
        initial, final = revenues[0], revenues[-1]
        if initial and final and final > initial:
            growth_rate = (final - initial) / abs(initial)
            if growth_rate > 0.5:  # e.g., 50% cumulative growth
                score += 2
                details.append(f"Revenue grew by {(growth_rate*100):.1f}% over the full period (strong growth).")
            else:
                score += 1
                details.append(f"Revenue growth is positive but under 50% cumulatively ({(growth_rate*100):.1f}%).")
        else:
            details.append("Revenue did not grow significantly or data insufficient.")
    else:
        details.append("Not enough revenue data for multi-period trend.")
    
    # 2. Operating margin and free cash flow consistency
    fcf_vals = [item.get('free_cash_flow') for item in financial_line_items if item.get('free_cash_flow') is not None]
    op_margin_vals = [item.get('operating_margin') for item in financial_line_items if item.get('operating_margin') is not None]
    
    if op_margin_vals:
        above_15 = sum(1 for m in op_margin_vals if m > 0.15)
        if above_15 >= (len(op_margin_vals) // 2 + 1):
            score += 2
            details.append("Operating margins have often exceeded 15% (indicates good profitability).")
        else:
            details.append("Operating margin not consistently above 15%.")
    else:
        details.append("No operating margin data across periods.")
    
    if fcf_vals:
        positive_fcf_count = sum(1 for f in fcf_vals if f > 0)
        if positive_fcf_count >= (len(fcf_vals) // 2 + 1):
            score += 1
            details.append("Majority of periods show positive free cash flow.")
        else:
            details.append("Free cash flow not consistently positive.")
    else:
        details.append("No free cash flow data across periods.")
    
    # 3. Return on Equity (ROE) check from the latest metrics
    latest_metrics = metrics[0]
    roe = latest_metrics.get('return_on_equity')
    if roe and roe > 0.15:
        score += 2
        details.append(f"High ROE of {roe:.1%}, indicating a competitive advantage.")
    elif roe:
        details.append(f"ROE of {roe:.1%} is moderate.")
    else:
        details.append("ROE data not available.")
    
    # 4. (Optional) Brand Intangible (if intangible_assets are fetched)
    # intangible_vals = [item.get('intangible_assets') for item in financial_line_items if item.get('intangible_assets')]
    # if intangible_vals and sum(intangible_vals) > 0:
    #     details.append("Significant intangible assets may indicate brand value or proprietary tech.")
    #     score += 1
    
    return {
        "score": score,
        "details": "; ".join(details)
    }


def analyze_financial_discipline(metrics: list, financial_line_items: list) -> dict:
    """
    Analyze the company's financial discipline:
    - Debt management
    - Capital allocation
    - Cost control
    - Working capital efficiency
    """
    score = 0
    details = []
    
    if not metrics or not financial_line_items:
        return {
            "score": 0,
            "details": "Insufficient data to analyze financial discipline"
        }
    
    # 1. Debt Management - Check debt/equity and interest coverage
    latest = financial_line_items[0]
    debt_to_equity = latest.get('debt_to_equity')
    if debt_to_equity is not None:
        if debt_to_equity < 1.0:
            score += 2
            details.append(f"Conservative debt level with D/E ratio of {debt_to_equity:.2f}")
        elif debt_to_equity < 2.0:
            score += 1
            details.append(f"Moderate debt level with D/E ratio of {debt_to_equity:.2f}")
        else:
            details.append(f"High debt level with D/E ratio of {debt_to_equity:.2f}")
    else:
        details.append("Debt to equity data not available")
    
    # 2. Working Capital Management
    current_ratio = latest.get('current_ratio')
    if current_ratio is not None:
        if current_ratio > 2.0:
            score += 2
            details.append(f"Strong working capital position with current ratio of {current_ratio:.2f}")
        elif current_ratio > 1.5:
            score += 1
            details.append(f"Adequate working capital with current ratio of {current_ratio:.2f}")
        else:
            details.append(f"Tight working capital with current ratio of {current_ratio:.2f}")
    else:
        details.append("Current ratio data not available")
    
    # 3. Cost Control - Check operating margin trend
    op_margins = [item.get('operating_margin') for item in financial_line_items if item.get('operating_margin') is not None]
    if len(op_margins) >= 2:
        margin_trend = op_margins[0] - op_margins[-1]
        if margin_trend > 0.02:  # 2% improvement
            score += 2
            details.append(f"Improving cost control with {(margin_trend*100):.1f}% margin expansion")
        elif margin_trend > 0:
            score += 1
            details.append(f"Stable cost control with slight margin improvement")
        else:
            details.append("Operating margins not improving")
    else:
        details.append("Insufficient operating margin data")
    
    # 4. Capital Allocation - Check ROIC trend
    roic_vals = [item.get('return_on_invested_capital') for item in financial_line_items if item.get('return_on_invested_capital') is not None]
    if roic_vals:
        avg_roic = sum(roic_vals) / len(roic_vals)
        if avg_roic > 0.15:  # 15% average ROIC
            score += 2
            details.append(f"Excellent capital allocation with {avg_roic:.1%} avg ROIC")
        elif avg_roic > 0.10:
            score += 1
            details.append(f"Good capital allocation with {avg_roic:.1%} avg ROIC")
        else:
            details.append(f"Subpar capital allocation with {avg_roic:.1%} avg ROIC")
    else:
        details.append("ROIC data not available")
    
    # Normalize score to 0-10
    normalized_score = min(10, (score / 8) * 10)  # max possible raw score is 8
    
    return {
        "score": normalized_score,
        "details": "; ".join(details)
    }


def analyze_activism_potential(metrics: list, financial_line_items: list) -> dict:
    """
    Analyze potential for activist intervention:
    - Undervaluation relative to peers/intrinsic value
    - Operational inefficiencies
    - Corporate governance issues
    - Balance sheet opportunities
    """
    score = 0
    details = []
    
    if not metrics or not financial_line_items:
        return {
            "score": 0,
            "details": "Insufficient data to analyze activism potential"
        }
    
    # 1. Operational Efficiency Check
    latest = financial_line_items[0]
    op_margin = latest.get('operating_margin')
    if op_margin is not None:
        if op_margin < 0.10:  # Below 10% operating margin
            score += 2
            details.append(f"Low operating margin of {op_margin:.1%} suggests operational improvement potential")
        elif op_margin < 0.15:
            score += 1
            details.append(f"Moderate operating margin of {op_margin:.1%} might have room for improvement")
    else:
        details.append("Operating margin data not available")
    
    # 2. Balance Sheet Efficiency
    current_ratio = latest.get('current_ratio')
    if current_ratio is not None:
        if current_ratio > 3.0:  # Potentially too much cash/working capital
            score += 2
            details.append(f"High current ratio of {current_ratio:.1f} suggests potential for balance sheet optimization")
        elif current_ratio > 2.5:
            score += 1
            details.append(f"Elevated current ratio of {current_ratio:.1f} might indicate some balance sheet inefficiency")
    else:
        details.append("Current ratio data not available")
    
    # 3. Capital Allocation Efficiency
    roic = latest.get('return_on_invested_capital')
    if roic is not None:
        if roic < 0.08:  # Below 8% ROIC
            score += 2
            details.append(f"Low ROIC of {roic:.1%} suggests potential for improved capital allocation")
        elif roic < 0.12:
            score += 1
            details.append(f"Moderate ROIC of {roic:.1%} might have room for improvement")
    else:
        details.append("ROIC data not available")
    
    # 4. Dividend and Buyback Policy
    payout_ratio = latest.get('payout_ratio')
    if payout_ratio is not None:
        if payout_ratio < 0.2 and latest.get('cash_and_equivalents', 0) > 0:  # Low payout despite having cash
            score += 2
            details.append("Low capital return despite cash reserves suggests potential for increased shareholder returns")
        elif payout_ratio < 0.3:
            score += 1
            details.append("Moderate capital return policy might have room for enhancement")
    else:
        details.append("Payout ratio data not available")
    
    # Normalize score to 0-10
    normalized_score = min(10, (score / 8) * 10)  # max possible raw score is 8
    
    return {
        "score": normalized_score,
        "details": "; ".join(details)
    }


def analyze_valuation(metrics, financial_line_items):
    """
    Analyzes the valuation metrics of a company to determine if it's undervalued.
    
    Args:
        metrics (dict): Dictionary containing the company's metrics
        financial_line_items (dict): Dictionary containing financial line items
    
    Returns:
        tuple: (score, message) where score is 0-10 and message explains the analysis
    """
    latest = metrics[0] if isinstance(metrics, list) else metrics
    
    # Initialize variables
    score = 0
    reasons = []
    
    # Get the free cash flow
    fcf = latest.get('free_cash_flow', 0)
    
    # Get enterprise value components
    market_cap = latest.get('outstanding_shares', 0) * latest.get('stock_price', 0)
    total_debt = latest.get('total_liabilities', 0)
    cash = latest.get('cash', 0)
    enterprise_value = market_cap + total_debt - cash
    
    # Calculate key ratios
    if enterprise_value > 0:
        ev_fcf_ratio = enterprise_value / fcf if fcf > 0 else float('inf')
        if ev_fcf_ratio < 15:
            score += 2
            reasons.append(f"Attractive EV/FCF ratio: {ev_fcf_ratio:.1f}x")
        elif ev_fcf_ratio > 25:
            reasons.append(f"High EV/FCF ratio: {ev_fcf_ratio:.1f}x")
    
    # Check P/E ratio
    pe_ratio = latest.get('pe_ratio', 0)
    if 0 < pe_ratio < 15:
        score += 2
        reasons.append(f"Attractive P/E ratio: {pe_ratio:.1f}x")
    elif pe_ratio > 25:
        reasons.append(f"High P/E ratio: {pe_ratio:.1f}x")
    
    # Check price to book ratio
    price_to_book = latest.get('price_to_book', 0)
    if 0 < price_to_book < 1.5:
        score += 2
        reasons.append(f"Trading below 1.5x book value: {price_to_book:.1f}x")
    
    # Check dividend yield
    dividend_yield = latest.get('dividend_yield', 0)
    if dividend_yield > 3:
        score += 1
        reasons.append(f"Attractive dividend yield: {dividend_yield:.1f}%")
    
    # Normalize score to 0-10 scale
    normalized_score = (score / 7) * 10
    
    return normalized_score, reasons

