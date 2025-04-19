from tools.utils import get_numeric_value

def analyze_fundamentals(latest_metrics):
    """Analyze company fundamentals based on Buffett's criteria."""
    print(f"DEBUG: Starting analyze_fundamentals with metrics: {latest_metrics}")
    if not latest_metrics:
        print("DEBUG: No metrics available, returning early")
        return 0, "No metrics available"
    
    score = 0
    reasons = []
    
    # Check Return on Equity (ROE)
    roe = get_numeric_value(latest_metrics.get("return_on_equity"))
    print(f"DEBUG: ROE value: {roe}")
    if roe is not None and roe > 0.15:  # 15% threshold
        score += 2
        reasons.append("Strong ROE above 15%")
        print(f"DEBUG: Added 2 points for strong ROE. New score: {score}")
    elif roe is not None:
        reasons.append(f"Weak ROE: {roe:.1%}")
        print(f"DEBUG: Weak ROE noted: {roe:.1%}")
    
    # Check Debt to Equity
    debt_to_equity = get_numeric_value(latest_metrics.get("debt_to_equity"))
    print(f"DEBUG: Debt to Equity value: {debt_to_equity}")
    if debt_to_equity is not None and debt_to_equity <= 0.5:  # Conservative threshold
        score += 2
        reasons.append("Conservative debt levels")
        print(f"DEBUG: Added 2 points for conservative debt. New score: {score}")
    elif debt_to_equity is not None:
        reasons.append(f"High debt to equity: {debt_to_equity:.1f}")
        print(f"DEBUG: High debt to equity noted: {debt_to_equity:.1f}")
    
    # Check Operating Margin
    operating_margin = get_numeric_value(latest_metrics.get("operating_margin"))
    print(f"DEBUG: Operating Margin value: {operating_margin}")
    if operating_margin is not None and operating_margin > 0.15:  # 15% threshold
        score += 2
        reasons.append("Strong operating margins")
        print(f"DEBUG: Added 2 points for strong margins. New score: {score}")
    elif operating_margin is not None:
        reasons.append(f"Weak operating margins: {operating_margin:.1%}")
        print(f"DEBUG: Weak operating margins noted: {operating_margin:.1%}")
    
    # Check Current Ratio
    current_ratio = get_numeric_value(latest_metrics.get("current_ratio"))
    print(f"DEBUG: Current Ratio value: {current_ratio}")
    if current_ratio is not None and current_ratio > 1.5:  # Healthy liquidity
        score += 1
        reasons.append("Strong liquidity position")
        print(f"DEBUG: Added 1 point for strong liquidity. New score: {score}")
    elif current_ratio is not None:
        reasons.append(f"Weak liquidity: {current_ratio:.1f}")
        print(f"DEBUG: Weak liquidity noted: {current_ratio:.1f}")
    
    # Normalize score to 0-1 range
    normalized_score = score / 7
    print(f"DEBUG: Final score: {score}, Normalized score: {normalized_score}")
    print(f"DEBUG: Reasons: {reasons}")
    
    return normalized_score, "; ".join(reasons)


def analyze_consistency(metrics):
    """Analyze earnings consistency and growth."""
    print(f"DEBUG: Starting analyze_consistency with {len(metrics) if metrics else 0} metrics")
    if not metrics or len(metrics) < 4:
        print("DEBUG: Insufficient metrics for consistency analysis, returning 0")
        return 0

    score = 0
    
    # Check earnings growth trend
    earnings_values = []
    for i, item in enumerate(metrics):
        print(f"DEBUG: Processing metric {i+1}/{len(metrics)}")
        net_income = item.get('net_income')
        print(f"DEBUG: Net income from metric: {net_income}")
        numeric_income = get_numeric_value(net_income)
        if numeric_income is not None:
            earnings_values.append(numeric_income)
            print(f"DEBUG: Added earnings value: {numeric_income}")
    
    print(f"DEBUG: Collected {len(earnings_values)} earnings values: {earnings_values}")
    
    if len(earnings_values) >= 4:
        # Simple check: is each period's earnings bigger than the next?
        try:
            print("DEBUG: Checking if earnings are consistently growing")
            earnings_growth = all(earnings_values[i] > earnings_values[i + 1] for i in range(len(earnings_values) - 1))
            print(f"DEBUG: Earnings growth check result: {earnings_growth}")
            if earnings_growth:
                score += 3
                print(f"DEBUG: Added 3 points for consistent growth. New score: {score}")
            
            # Calculate total growth rate from oldest to latest
            if len(earnings_values) >= 2 and earnings_values[-1] != 0:
                growth_rate = (earnings_values[0] - earnings_values[-1]) / abs(earnings_values[-1])
                print(f"DEBUG: Calculated growth rate: {growth_rate:.2%}")
                if growth_rate > 0.10:  # 10% growth
                    score += 2
                    print(f"DEBUG: Added 2 points for high growth rate. New score: {score}")
        except (TypeError, ValueError) as e:
            # If any comparison fails, return 0
            print(f"DEBUG: Error in consistency analysis: {str(e)}")
            return 0
    
    normalized_score = score / 5  # Normalize to 0-1 range
    print(f"DEBUG: Final score: {score}, Normalized score: {normalized_score}")
    return normalized_score


def analyze_moat(metrics: list) -> dict[str, any]:
    """
    Evaluate whether the company likely has a durable competitive advantage (moat).
    For simplicity, we look at stability of ROE/operating margins over multiple periods
    or high margin over the last few years. Higher stability => higher moat score.
    """
    print(f"DEBUG: Starting analyze_moat with {len(metrics) if metrics else 0} metrics")
    if not metrics or len(metrics) < 3:
        print("DEBUG: Insufficient metrics for moat analysis, returning early")
        return {"score": 0, "max_score": 3, "details": "Insufficient data for moat analysis"}

    reasoning = []
    moat_score = 0
    historical_roes = []
    historical_margins = []

    for i, m in enumerate(metrics):
        print(f"DEBUG: Processing metric {i+1}/{len(metrics)} for moat analysis")
        roe = get_numeric_value(m.get("return_on_equity"))
        if roe is not None:
            historical_roes.append(roe)
            print(f"DEBUG: Added ROE value: {roe}")
        
        margin = get_numeric_value(m.get("operating_margin"))
        if margin is not None:
            historical_margins.append(margin)
            print(f"DEBUG: Added margin value: {margin}")

    print(f"DEBUG: Collected {len(historical_roes)} ROE values: {historical_roes}")
    print(f"DEBUG: Collected {len(historical_margins)} margin values: {historical_margins}")

    # Check for stable or improving ROE
    if len(historical_roes) >= 3:
        print("DEBUG: Checking for stable ROE")
        stable_roe = all(r > 0.15 for r in historical_roes)
        print(f"DEBUG: ROE stability check result: {stable_roe}")
        if stable_roe:
            moat_score += 1
            reasoning.append("Stable ROE above 15% across periods (suggests moat)")
            print(f"DEBUG: Added 1 point for stable ROE. New score: {moat_score}")
        else:
            reasoning.append("ROE not consistently above 15%")
            print("DEBUG: ROE not consistently above 15%")

    # Check for stable or improving operating margin
    if len(historical_margins) >= 3:
        print("DEBUG: Checking for stable operating margin")
        stable_margin = all(m > 0.15 for m in historical_margins)
        print(f"DEBUG: Margin stability check result: {stable_margin}")
        if stable_margin:
            moat_score += 1
            reasoning.append("Stable operating margins above 15% (moat indicator)")
            print(f"DEBUG: Added 1 point for stable margins. New score: {moat_score}")
        else:
            reasoning.append("Operating margin not consistently above 15%")
            print("DEBUG: Operating margin not consistently above 15%")

    # Check for consistent or improving margins
    if len(historical_margins) >= 3:
        print("DEBUG: Checking for margin improvement")
        margin_improvement = all(historical_margins[i] >= historical_margins[i + 1] for i in range(len(historical_margins) - 1))
        print(f"DEBUG: Margin improvement check result: {margin_improvement}")
        if margin_improvement:
            moat_score += 1
            reasoning.append("Consistently improving operating margins")
            print(f"DEBUG: Added 1 point for improving margins. New score: {moat_score}")
        else:
            reasoning.append("Operating margins not consistently improving")
            print("DEBUG: Operating margins not consistently improving")

    normalized_score = moat_score / 3  # Normalize to 0-1 range
    print(f"DEBUG: Final moat score: {moat_score}/3, Normalized score: {normalized_score}")
    print(f"DEBUG: Reasoning: {reasoning}")
    
    return {
        "score": normalized_score,
        "max_score": 3,
        "details": "; ".join(reasoning)
    }


def analyze_management_quality(financial_line_items: list) -> dict[str, any]:
    """
    Checks for share dilution or consistent buybacks, and some dividend track record.
    A simplified approach:
      - if there's net share repurchase or stable share count, it suggests management
        might be shareholder-friendly.
      - if there's a big new issuance, it might be a negative sign (dilution).
    """
    print(f"DEBUG: Starting analyze_management_quality with {len(financial_line_items) if financial_line_items else 0} items")
    if not financial_line_items:
        print("DEBUG: No financial line items provided, returning early")
        return {"score": 0, "max_score": 2, "details": "Insufficient data for management analysis"}

    reasoning = []
    mgmt_score = 0

    # Check share count stability
    share_counts = []
    for item in financial_line_items:
        shares = get_numeric_value(item.get("outstanding_shares"))
        if shares is not None:
            share_counts.append(shares)
            print(f"DEBUG: Added share count: {shares}")

    print(f"DEBUG: Collected {len(share_counts)} share counts: {share_counts}")

    if len(share_counts) >= 3:
        # Check if share count is stable or decreasing (buybacks)
        share_stability = all(share_counts[i] >= share_counts[i + 1] for i in range(len(share_counts) - 1))
        print(f"DEBUG: Share stability check result: {share_stability}")
        if share_stability:
            mgmt_score += 1
            reasoning.append("Stable or decreasing share count (suggests buybacks)")
            print(f"DEBUG: Added 1 point for share stability. New score: {mgmt_score}")
        else:
            reasoning.append("Share count increasing (potential dilution)")
            print("DEBUG: Share count increasing noted")

    # Check dividend track record
    dividend_yields = []
    for item in financial_line_items:
        yield_val = get_numeric_value(item.get("dividend_yield"))
        if yield_val is not None:
            dividend_yields.append(yield_val)
            print(f"DEBUG: Added dividend yield: {yield_val}")

    print(f"DEBUG: Collected {len(dividend_yields)} dividend yields: {dividend_yields}")

    if len(dividend_yields) >= 3:
        # Check if dividend yield is stable or increasing
        dividend_stability = all(dividend_yields[i] >= dividend_yields[i + 1] for i in range(len(dividend_yields) - 1))
        print(f"DEBUG: Dividend stability check result: {dividend_stability}")
        if dividend_stability:
            mgmt_score += 1
            reasoning.append("Stable or increasing dividend yield")
            print(f"DEBUG: Added 1 point for dividend stability. New score: {mgmt_score}")
        else:
            reasoning.append("Dividend yield not consistently stable")
            print("DEBUG: Dividend yield not consistently stable")

    normalized_score = mgmt_score / 2  # Normalize to 0-1 range
    print(f"DEBUG: Final management score: {mgmt_score}/2, Normalized score: {normalized_score}")
    print(f"DEBUG: Reasoning: {reasoning}")
    
    return {
        "score": normalized_score,
        "max_score": 2,
        "details": "; ".join(reasoning)
    }


def calculate_owner_earnings(latest_metrics: dict) -> dict[str, any]:
    """Calculate owner earnings (Buffett's preferred measure of true earnings power).
    Owner Earnings = Net Income + Depreciation - Maintenance CapEx"""
    print(f"DEBUG: Starting calculate_owner_earnings with metrics: {latest_metrics}")
    if not latest_metrics:
        print("DEBUG: No metrics available for owner earnings calculation, returning early")
        return {"owner_earnings": None, "details": ["No metrics available for owner earnings calculation"]}

    # Get the raw metrics first for debugging
    raw_net_income = latest_metrics.get("net_income")
    raw_depreciation = latest_metrics.get("depreciation")
    raw_maintenance_capex = latest_metrics.get("maintenance_capex")

    print(f"DEBUG: Raw net_income type: {type(raw_net_income)}, value: {raw_net_income}")
    print(f"DEBUG: Raw depreciation type: {type(raw_depreciation)}, value: {raw_depreciation}")
    print(f"DEBUG: Raw maintenance_capex type: {type(raw_maintenance_capex)}, value: {raw_maintenance_capex}")

    # Get the required metrics using get_numeric_value
    net_income = get_numeric_value(raw_net_income)
    depreciation = get_numeric_value(raw_depreciation)
    maintenance_capex = get_numeric_value(raw_maintenance_capex)

    print(f"DEBUG: Processed Net Income: {net_income}")
    print(f"DEBUG: Processed Depreciation: {depreciation}")
    print(f"DEBUG: Processed Maintenance CapEx: {maintenance_capex}")

    # Ensure all values are floats and not None
    if None in (net_income, depreciation, maintenance_capex):
        print("DEBUG: Missing required metrics for owner earnings calculation")
        return {"owner_earnings": None, "details": ["Missing required metrics for owner earnings calculation"]}

    try:
        owner_earnings = float(net_income) + float(depreciation) - float(maintenance_capex)
        print(f"DEBUG: Calculated owner earnings: {owner_earnings}")

        return {
            "owner_earnings": owner_earnings,
            "details": [
                f"Net Income: {net_income}",
                f"Depreciation: {depreciation}",
                f"Maintenance CapEx: {maintenance_capex}",
                f"Owner Earnings: {owner_earnings}"
            ]
        }
    except (TypeError, ValueError) as e:
        print(f"DEBUG: Error calculating owner earnings: {str(e)}")
        return {"owner_earnings": None, "details": [f"Error calculating owner earnings: {str(e)}"]}


def calculate_intrinsic_value(latest_metrics: dict) -> dict[str, any]:
    """Calculate intrinsic value using DCF with owner earnings."""
    print(f"DEBUG: Starting calculate_intrinsic_value with metrics: {latest_metrics}")
    if not latest_metrics:
        print("DEBUG: No metrics available for intrinsic value calculation, returning early")
        return {"intrinsic_value": None, "details": ["No metrics available for intrinsic value calculation"]}

    # Get owner earnings first
    owner_earnings_result = calculate_owner_earnings(latest_metrics)
    owner_earnings = owner_earnings_result.get("owner_earnings")
    
    if owner_earnings is None:
        print("DEBUG: Could not calculate owner earnings, returning early")
        return {"intrinsic_value": None, "details": ["Could not calculate owner earnings"]}

    # Get shares outstanding
    shares_outstanding = get_numeric_value(latest_metrics.get("outstanding_shares"))
    print(f"DEBUG: Shares outstanding: {shares_outstanding}")

    if not shares_outstanding:
        print("DEBUG: Missing shares outstanding data, returning early")
        return {"intrinsic_value": None, "details": ["Missing shares outstanding data"]}

    # Simple DCF calculation
    discount_rate = 0.10  # 10% discount rate
    growth_rate = 0.05   # 5% growth rate
    terminal_multiple = 12  # Terminal multiple of 12x earnings
    
    print(f"DEBUG: Using discount rate: {discount_rate:.1%}")
    print(f"DEBUG: Using growth rate: {growth_rate:.1%}")
    print(f"DEBUG: Using terminal multiple: {terminal_multiple}x")

    # Project 10 years of earnings
    projected_earnings = []
    current_earnings = owner_earnings
    for i in range(10):
        projected_earnings.append(current_earnings)
        print(f"DEBUG: Year {i+1} projected earnings: {current_earnings}")
        current_earnings *= (1 + growth_rate)

    # Calculate terminal value
    terminal_value = projected_earnings[-1] * terminal_multiple
    print(f"DEBUG: Terminal value: {terminal_value}")

    # Discount all cash flows
    present_value = 0
    for i, earnings in enumerate(projected_earnings):
        discounted_value = earnings / ((1 + discount_rate) ** (i + 1))
        present_value += discounted_value
        print(f"DEBUG: Year {i+1} discounted value: {discounted_value}")

    # Add discounted terminal value
    terminal_value_discounted = terminal_value / ((1 + discount_rate) ** 10)
    present_value += terminal_value_discounted
    print(f"DEBUG: Discounted terminal value: {terminal_value_discounted}")

    # Calculate per share value
    intrinsic_value_per_share = present_value / shares_outstanding
    print(f"DEBUG: Intrinsic value per share: {intrinsic_value_per_share}")

    return {
        "intrinsic_value": intrinsic_value_per_share,
        "details": [
            f"Owner Earnings: {owner_earnings}",
            f"Shares Outstanding: {shares_outstanding}",
            f"Present Value: {present_value}",
            f"Intrinsic Value per Share: {intrinsic_value_per_share}"
        ]
    }
