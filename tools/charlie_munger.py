def analyze_moat_strength(metrics: list, financial_line_items: list) -> dict:
    """
    Analyze the business's competitive advantage using Munger's approach:
    - Consistent high returns on capital (ROIC)
    - Pricing power (stable/improving gross margins)
    - Low capital requirements
    - Network effects and intangible assets (R&D investments, goodwill)
    """
    score = 0
    details = []
    
    if not metrics or not financial_line_items:
        print("MISSING DATA: No metrics or financial_line_items provided for moat strength analysis")
        return {
            "score": 0,
            "details": "Insufficient data to analyze moat strength"
        }
    
    # 1. Return on Invested Capital (ROIC) analysis - Munger's favorite metric
    roic_values = [item.return_on_invested_capital for item in financial_line_items 
                   if hasattr(item, 'return_on_invested_capital') and item.return_on_invested_capital is not None]
    
    if roic_values:
        # Check if ROIC consistently above 15% (Munger's threshold)
        high_roic_count = sum(1 for r in roic_values if r > 0.15)
        if high_roic_count >= len(roic_values) * 0.8:  # 80% of periods show high ROIC
            score += 3
            details.append(f"Excellent ROIC: >15% in {high_roic_count}/{len(roic_values)} periods")
        elif high_roic_count >= len(roic_values) * 0.5:  # 50% of periods
            score += 2
            details.append(f"Good ROIC: >15% in {high_roic_count}/{len(roic_values)} periods")
        elif high_roic_count > 0:
            score += 1
            details.append(f"Mixed ROIC: >15% in only {high_roic_count}/{len(roic_values)} periods")
        else:
            details.append("Poor ROIC: Never exceeds 15% threshold")
    else:
        print("MISSING DATA: No ROIC data available for moat strength analysis")
        details.append("No ROIC data available")
    
    # 2. Pricing power - check gross margin stability and trends
    gross_margins = [item.gross_margin for item in financial_line_items 
                    if hasattr(item, 'gross_margin') and item.gross_margin is not None]
    
    if gross_margins and len(gross_margins) >= 3:
        # Munger likes stable or improving gross margins
        margin_trend = sum(1 for i in range(1, len(gross_margins)) if gross_margins[i] >= gross_margins[i-1])
        if margin_trend >= len(gross_margins) * 0.7:  # Improving in 70% of periods
            score += 2
            details.append("Strong pricing power: Gross margins consistently improving")
        elif sum(gross_margins) / len(gross_margins) > 0.3:  # Average margin > 30%
            score += 1
            details.append(f"Good pricing power: Average gross margin {sum(gross_margins)/len(gross_margins):.1%}")
        else:
            details.append("Limited pricing power: Low or declining gross margins")
    else:
        print("MISSING DATA: Insufficient gross margin data for moat strength analysis")
        details.append("Insufficient gross margin data")
    
    # 3. Capital intensity - Munger prefers low capex businesses
    if len(financial_line_items) >= 3:
        capex_to_revenue = []
        for item in financial_line_items:
            if (hasattr(item, 'capital_expenditure') and item.capital_expenditure is not None and 
                hasattr(item, 'revenue') and item.revenue is not None and item.revenue > 0):
                # Note: capital_expenditure is typically negative in financial statements
                capex_ratio = abs(item.capital_expenditure) / item.revenue
                capex_to_revenue.append(capex_ratio)
        
        if capex_to_revenue:
            avg_capex_ratio = sum(capex_to_revenue) / len(capex_to_revenue)
            if avg_capex_ratio < 0.05:  # Less than 5% of revenue
                score += 2
                details.append(f"Low capital requirements: Avg capex {avg_capex_ratio:.1%} of revenue")
            elif avg_capex_ratio < 0.10:  # Less than 10% of revenue
                score += 1
                details.append(f"Moderate capital requirements: Avg capex {avg_capex_ratio:.1%} of revenue")
            else:
                details.append(f"High capital requirements: Avg capex {avg_capex_ratio:.1%} of revenue")
        else:
            print("MISSING DATA: No capital expenditure data available for moat strength analysis")
            details.append("No capital expenditure data available")
    else:
        print("MISSING DATA: Insufficient data for capital intensity analysis")
        details.append("Insufficient data for capital intensity analysis")
    
    # 4. Intangible assets - Munger values R&D and intellectual property
    r_and_d = [item.research_and_development for item in financial_line_items
              if hasattr(item, 'research_and_development') and item.research_and_development is not None]
    
    goodwill_and_intangible_assets = [item.goodwill_and_intangible_assets for item in financial_line_items
               if hasattr(item, 'goodwill_and_intangible_assets') and item.goodwill_and_intangible_assets is not None]

    if r_and_d and len(r_and_d) > 0:
        if sum(r_and_d) > 0:  # If company is investing in R&D
            score += 1
            details.append("Invests in R&D, building intellectual property")
    else:
        print("MISSING DATA: No R&D data available for moat strength analysis")
    
    if (goodwill_and_intangible_assets and len(goodwill_and_intangible_assets) > 0):
        score += 1
        details.append("Significant goodwill/intangible assets, suggesting brand value or IP")
    else:
        print("MISSING DATA: No goodwill/intangible assets data available for moat strength analysis")
    
    # Scale score to 0-10 range
    final_score = min(10, score * 10 / 9)  # Max possible raw score is 9
    
    return {
        "score": final_score,
        "details": "; ".join(details)
    }


def analyze_management_quality(financial_line_items: list, insider_trades: list) -> dict:
    """
    Evaluate management quality using Munger's criteria:
    - Capital allocation wisdom
    - Insider ownership and transactions
    - Cash management efficiency
    - Candor and transparency
    - Long-term focus
    """
    score = 0
    details = []
    
    if not financial_line_items:
        print("MISSING DATA: No financial_line_items provided for management quality analysis")
        return {
            "score": 0,
            "details": "Insufficient data to analyze management quality"
        }
    
    # 1. Capital allocation - Check FCF to net income ratio
    # Munger values companies that convert earnings to cash
    fcf_values = [item.free_cash_flow for item in financial_line_items 
                 if hasattr(item, 'free_cash_flow') and item.free_cash_flow is not None]
    
    net_income_values = [item.net_income for item in financial_line_items 
                        if hasattr(item, 'net_income') and item.net_income is not None]
    
    if fcf_values and net_income_values and len(fcf_values) == len(net_income_values):
        # Calculate FCF to Net Income ratio for each period
        fcf_to_ni_ratios = []
        for i in range(len(fcf_values)):
            if net_income_values[i] and net_income_values[i] > 0:
                fcf_to_ni_ratios.append(fcf_values[i] / net_income_values[i])
        
        if fcf_to_ni_ratios:
            avg_ratio = sum(fcf_to_ni_ratios) / len(fcf_to_ni_ratios)
            if avg_ratio > 1.1:  # FCF > net income suggests good accounting
                score += 3
                details.append(f"Excellent cash conversion: FCF/NI ratio of {avg_ratio:.2f}")
            elif avg_ratio > 0.9:  # FCF roughly equals net income
                score += 2
                details.append(f"Good cash conversion: FCF/NI ratio of {avg_ratio:.2f}")
            elif avg_ratio > 0.7:  # FCF somewhat lower than net income
                score += 1
                details.append(f"Moderate cash conversion: FCF/NI ratio of {avg_ratio:.2f}")
            else:
                details.append(f"Poor cash conversion: FCF/NI ratio of only {avg_ratio:.2f}")
        else:
            print("MISSING DATA: Could not calculate FCF to Net Income ratios")
            details.append("Could not calculate FCF to Net Income ratios")
    else:
        print("MISSING DATA: Missing FCF or Net Income data for management quality analysis")
        details.append("Missing FCF or Net Income data")
    
    # 2. Debt management - Munger is cautious about debt
    debt_values = [item.total_debt for item in financial_line_items 
                  if hasattr(item, 'total_debt') and item.total_debt is not None]
    
    equity_values = [item.shareholders_equity for item in financial_line_items 
                    if hasattr(item, 'shareholders_equity') and item.shareholders_equity is not None]
    
    if debt_values and equity_values and len(debt_values) == len(equity_values):
        # Calculate D/E ratio for most recent period
        recent_de_ratio = debt_values[0] / equity_values[0] if equity_values[0] > 0 else float('inf')
        
        if recent_de_ratio < 0.3:  # Very low debt
            score += 3
            details.append(f"Conservative debt management: D/E ratio of {recent_de_ratio:.2f}")
        elif recent_de_ratio < 0.7:  # Moderate debt
            score += 2
            details.append(f"Prudent debt management: D/E ratio of {recent_de_ratio:.2f}")
        elif recent_de_ratio < 1.5:  # Higher but still reasonable debt
            score += 1
            details.append(f"Moderate debt level: D/E ratio of {recent_de_ratio:.2f}")
        else:
            details.append(f"High debt level: D/E ratio of {recent_de_ratio:.2f}")
    else:
        print("MISSING DATA: Missing debt or equity data for management quality analysis")
        details.append("Missing debt or equity data")
    
    # 3. Cash management efficiency - Munger values appropriate cash levels
    cash_values = [item.cash_and_equivalents for item in financial_line_items
                  if hasattr(item, 'cash_and_equivalents') and item.cash_and_equivalents is not None]
    revenue_values = [item.revenue for item in financial_line_items
                     if hasattr(item, 'revenue') and item.revenue is not None]
    
    if cash_values and revenue_values and len(cash_values) > 0 and len(revenue_values) > 0:
        # Calculate cash to revenue ratio (Munger likes 10-20% for most businesses)
        cash_to_revenue = cash_values[0] / revenue_values[0] if revenue_values[0] > 0 else 0
        
        if 0.1 <= cash_to_revenue <= 0.25:
            # Goldilocks zone - not too much, not too little
            score += 2
            details.append(f"Prudent cash management: Cash/Revenue ratio of {cash_to_revenue:.2f}")
        elif 0.05 <= cash_to_revenue < 0.1 or 0.25 < cash_to_revenue <= 0.4:
            # Reasonable but not ideal
            score += 1
            details.append(f"Acceptable cash position: Cash/Revenue ratio of {cash_to_revenue:.2f}")
        elif cash_to_revenue > 0.4:
            # Too much cash - potentially inefficient capital allocation
            details.append(f"Excess cash reserves: Cash/Revenue ratio of {cash_to_revenue:.2f}")
        else:
            # Too little cash - potentially risky
            details.append(f"Low cash reserves: Cash/Revenue ratio of {cash_to_revenue:.2f}")
    else:
        print("MISSING DATA: Insufficient cash or revenue data for management quality analysis")
        details.append("Insufficient cash or revenue data")
    
    # 4. Insider activity - Munger values skin in the game
    if insider_trades and len(insider_trades) > 0:
        # Count buys vs. sells
        buys = sum(1 for trade in insider_trades if hasattr(trade, 'transaction_type') and 
                   trade.transaction_type and trade.transaction_type.lower() in ['buy', 'purchase'])
        sells = sum(1 for trade in insider_trades if hasattr(trade, 'transaction_type') and 
                    trade.transaction_type and trade.transaction_type.lower() in ['sell', 'sale'])
        
        # Calculate the buy ratio
        total_trades = buys + sells
        if total_trades > 0:
            buy_ratio = buys / total_trades
            if buy_ratio > 0.7:  # Strong insider buying
                score += 2
                details.append(f"Strong insider buying: {buys}/{total_trades} transactions are purchases")
            elif buy_ratio > 0.4:  # Balanced insider activity
                score += 1
                details.append(f"Balanced insider trading: {buys}/{total_trades} transactions are purchases")
            elif buy_ratio < 0.1 and sells > 5:  # Heavy selling
                score -= 1  # Penalty for excessive selling
                details.append(f"Concerning insider selling: {sells}/{total_trades} transactions are sales")
            else:
                details.append(f"Mixed insider activity: {buys}/{total_trades} transactions are purchases")
        else:
            print("MISSING DATA: No recorded insider transactions")
            details.append("No recorded insider transactions")
    else:
        print("MISSING DATA: No insider trading data available for management quality analysis")
        details.append("No insider trading data available")
    
    # 5. Consistency in share count - Munger prefers stable/decreasing shares
    share_counts = [item.outstanding_shares for item in financial_line_items
                   if hasattr(item, 'outstanding_shares') and item.outstanding_shares is not None]
    
    if share_counts and len(share_counts) >= 3:
        if share_counts[0] < share_counts[-1] * 0.95:  # 5%+ reduction in shares
            score += 2
            details.append("Shareholder-friendly: Reducing share count over time")
        elif share_counts[0] < share_counts[-1] * 1.05:  # Stable share count
            score += 1
            details.append("Stable share count: Limited dilution")
        elif share_counts[0] > share_counts[-1] * 1.2:  # >20% dilution
            score -= 1  # Penalty for excessive dilution
            details.append("Concerning dilution: Share count increased significantly")
        else:
            details.append("Moderate share count increase over time")
    else:
        print("MISSING DATA: Insufficient share count data for management quality analysis")
        details.append("Insufficient share count data")
    
    # Scale score to 0-10 range
    # Maximum possible raw score would be 12 (3+3+2+2+2)
    final_score = max(0, min(10, score * 10 / 12))
    
    return {
        "score": final_score,
        "details": "; ".join(details)
    }


def analyze_predictability(financial_line_items: list) -> dict:
    """
    Assess the predictability of the business - Munger strongly prefers businesses
    whose future operations and cashflows are relatively easy to predict.
    """
    score = 0
    details = []
    
    if not financial_line_items or len(financial_line_items) < 5:
        print("MISSING DATA: Insufficient financial_line_items for predictability analysis (need 5+ years)")
        return {
            "score": 0,
            "details": "Insufficient data to analyze business predictability (need 5+ years)"
        }
    
    # 1. Revenue stability and growth
    revenues = [item.revenue for item in financial_line_items 
               if hasattr(item, 'revenue') and item.revenue is not None]
    
    if revenues and len(revenues) >= 5:
        # Calculate year-over-year growth rates
        growth_rates = [(revenues[i] / revenues[i+1] - 1) for i in range(len(revenues)-1)]
        
        avg_growth = sum(growth_rates) / len(growth_rates)
        growth_volatility = sum(abs(r - avg_growth) for r in growth_rates) / len(growth_rates)
        
        if avg_growth > 0.05 and growth_volatility < 0.1:
            # Steady, consistent growth (Munger loves this)
            score += 3
            details.append(f"Highly predictable revenue: {avg_growth:.1%} avg growth with low volatility")
        elif avg_growth > 0 and growth_volatility < 0.2:
            # Positive but somewhat volatile growth
            score += 2
            details.append(f"Moderately predictable revenue: {avg_growth:.1%} avg growth with some volatility")
        elif avg_growth > 0:
            # Growing but unpredictable
            score += 1
            details.append(f"Growing but less predictable revenue: {avg_growth:.1%} avg growth with high volatility")
        else:
            details.append(f"Declining or highly unpredictable revenue: {avg_growth:.1%} avg growth")
    else:
        print("MISSING DATA: Insufficient revenue history for predictability analysis")
        details.append("Insufficient revenue history for predictability analysis")
    
    # 2. Operating income stability
    op_income = [item.operating_income for item in financial_line_items 
                if hasattr(item, 'operating_income') and item.operating_income is not None]
    
    if op_income and len(op_income) >= 5:
        # Count positive operating income periods
        positive_periods = sum(1 for income in op_income if income > 0)
        
        if positive_periods == len(op_income):
            # Consistently profitable operations
            score += 3
            details.append("Highly predictable operations: Operating income positive in all periods")
        elif positive_periods >= len(op_income) * 0.8:
            # Mostly profitable operations
            score += 2
            details.append(f"Predictable operations: Operating income positive in {positive_periods}/{len(op_income)} periods")
        elif positive_periods >= len(op_income) * 0.6:
            # Somewhat profitable operations
            score += 1
            details.append(f"Somewhat predictable operations: Operating income positive in {positive_periods}/{len(op_income)} periods")
        else:
            details.append(f"Unpredictable operations: Operating income positive in only {positive_periods}/{len(op_income)} periods")
    else:
        print("MISSING DATA: Insufficient operating income history for predictability analysis")
        details.append("Insufficient operating income history")
    
    # 3. Margin consistency - Munger values stable margins
    op_margins = [item.operating_margin for item in financial_line_items 
                 if hasattr(item, 'operating_margin') and item.operating_margin is not None]
    
    if op_margins and len(op_margins) >= 5:
        # Calculate margin volatility
        avg_margin = sum(op_margins) / len(op_margins)
        margin_volatility = sum(abs(m - avg_margin) for m in op_margins) / len(op_margins)
        
        if margin_volatility < 0.03:  # Very stable margins
            score += 2
            details.append(f"Highly predictable margins: {avg_margin:.1%} avg with minimal volatility")
        elif margin_volatility < 0.07:  # Moderately stable margins
            score += 1
            details.append(f"Moderately predictable margins: {avg_margin:.1%} avg with some volatility")
        else:
            details.append(f"Unpredictable margins: {avg_margin:.1%} avg with high volatility ({margin_volatility:.1%})")
    else:
        print("MISSING DATA: Insufficient margin history for predictability analysis")
        details.append("Insufficient margin history")
    
    # 4. Cash generation reliability
    fcf_values = [item.free_cash_flow for item in financial_line_items 
                 if hasattr(item, 'free_cash_flow') and item.free_cash_flow is not None]
    
    if fcf_values and len(fcf_values) >= 5:
        # Count positive FCF periods
        positive_fcf_periods = sum(1 for fcf in fcf_values if fcf > 0)
        
        if positive_fcf_periods == len(fcf_values):
            # Consistently positive FCF
            score += 2
            details.append("Highly predictable cash generation: Positive FCF in all periods")
        elif positive_fcf_periods >= len(fcf_values) * 0.8:
            # Mostly positive FCF
            score += 1
            details.append(f"Predictable cash generation: Positive FCF in {positive_fcf_periods}/{len(fcf_values)} periods")
        else:
            details.append(f"Unpredictable cash generation: Positive FCF in only {positive_fcf_periods}/{len(fcf_values)} periods")
    else:
        print("MISSING DATA: Insufficient free cash flow history for predictability analysis")
        details.append("Insufficient free cash flow history")
    
    # Scale score to 0-10 range
    # Maximum possible raw score would be 10 (3+3+2+2)
    final_score = min(10, score * 10 / 10)
    
    return {
        "score": final_score,
        "details": "; ".join(details)
    }