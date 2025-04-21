import statistics

def analyze_fisher_growth_quality(financial_line_items: list) -> dict:
    """
    Evaluate growth & quality:
      - Consistent Revenue Growth
      - Consistent EPS Growth
      - R&D as a % of Revenue (if relevant, indicative of future-oriented spending)
    """
    if not financial_line_items or len(financial_line_items) < 2:
        print("MISSING DATA: Insufficient financial data for growth/quality analysis")
        return {
            "score": 0,
            "details": "Insufficient financial data for growth/quality analysis",
        }

    details = []
    raw_score = 0  # up to 9 raw points => scale to 0–10

    # 1. Revenue Growth (YoY)
    revenues = [fi['revenue'] for fi in financial_line_items if fi.get('revenue') is not None]
    if len(revenues) >= 2:
        # We'll look at the earliest vs. latest to gauge multi-year growth if possible
        latest_rev = revenues[0]
        oldest_rev = revenues[-1]
        if oldest_rev > 0:
            rev_growth = (latest_rev - oldest_rev) / abs(oldest_rev)
            if rev_growth > 0.80:
                raw_score += 3
                details.append(f"Very strong multi-period revenue growth: {rev_growth:.1%}")
            elif rev_growth > 0.40:
                raw_score += 2
                details.append(f"Moderate multi-period revenue growth: {rev_growth:.1%}")
            elif rev_growth > 0.10:
                raw_score += 1
                details.append(f"Slight multi-period revenue growth: {rev_growth:.1%}")
            else:
                details.append(f"Minimal or negative multi-period revenue growth: {rev_growth:.1%}")
        else:
            print("MISSING DATA: Oldest revenue is zero/negative; cannot compute growth")
            details.append("Oldest revenue is zero/negative; cannot compute growth.")
    else:
        print(f"MISSING DATA: Not enough revenue data points for growth calculation. Found {len(revenues)} points, need at least 2")
        details.append("Not enough revenue data points for growth calculation.")

    # 2. EPS Growth (YoY)
    eps_values = [fi['eps'] for fi in financial_line_items if fi.get('eps') is not None]
    if len(eps_values) >= 2:
        latest_eps = eps_values[0]
        oldest_eps = eps_values[-1]
        if abs(oldest_eps) > 1e-9:
            eps_growth = (latest_eps - oldest_eps) / abs(oldest_eps)
            if eps_growth > 0.80:
                raw_score += 3
                details.append(f"Very strong multi-period EPS growth: {eps_growth:.1%}")
            elif eps_growth > 0.40:
                raw_score += 2
                details.append(f"Moderate multi-period EPS growth: {eps_growth:.1%}")
            elif eps_growth > 0.10:
                raw_score += 1
                details.append(f"Slight multi-period EPS growth: {eps_growth:.1%}")
            else:
                details.append(f"Minimal or negative multi-period EPS growth: {eps_growth:.1%}")
        else:
            print("MISSING DATA: Oldest EPS near zero; skipping EPS growth calculation")
            details.append("Oldest EPS near zero; skipping EPS growth calculation.")
    else:
        print(f"MISSING DATA: Not enough EPS data points for growth calculation. Found {len(eps_values)} points, need at least 2")
        details.append("Not enough EPS data points for growth calculation.")

    # 3. R&D as % of Revenue (if we have R&D data)
    rnd_values = [fi['research_and_development'] for fi in financial_line_items if fi.get('research_and_development') is not None]
    if rnd_values and revenues and len(rnd_values) == len(revenues):
        # We'll just look at the most recent for a simple measure
        recent_rnd = rnd_values[0]
        recent_rev = revenues[0] if revenues[0] else 1e-9
        rnd_ratio = recent_rnd / recent_rev
        # Generally, Fisher admired companies that invest aggressively in R&D,
        # but it must be appropriate. We'll assume "3%-15%" is healthy, just as an example.
        if 0.03 <= rnd_ratio <= 0.15:
            raw_score += 3
            details.append(f"R&D ratio {rnd_ratio:.1%} indicates significant investment in future growth")
        elif rnd_ratio > 0.15:
            raw_score += 2
            details.append(f"R&D ratio {rnd_ratio:.1%} is very high (could be good if well-managed)")
        elif rnd_ratio > 0.0:
            raw_score += 1
            details.append(f"R&D ratio {rnd_ratio:.1%} is somewhat low but still positive")
        else:
            details.append("No meaningful R&D expense ratio")
    else:
        print("MISSING DATA: Insufficient R&D data to evaluate")
        details.append("Insufficient R&D data to evaluate")

    # scale raw_score (max 9) to 0–10
    final_score = min(10, (raw_score / 9) * 10)
    return {"score": final_score, "details": "; ".join(details)}


def analyze_margins_stability(financial_line_items: list) -> dict:
    """
    Looks at margin consistency (gross/operating margin) and general stability over time.
    """
    if not financial_line_items or len(financial_line_items) < 2:
        print("MISSING DATA: Insufficient data for margin stability analysis")
        return {
            "score": 0,
            "details": "Insufficient data for margin stability analysis",
        }

    details = []
    raw_score = 0  # up to 6 => scale to 0-10

    # 1. Operating Margin Consistency
    op_margins = [fi['operating_margin'] for fi in financial_line_items if fi.get('operating_margin') is not None]
    if len(op_margins) >= 2:
        # Check if margins are stable or improving (comparing oldest to newest)
        oldest_op_margin = op_margins[-1]
        newest_op_margin = op_margins[0]
        if newest_op_margin >= oldest_op_margin > 0:
            raw_score += 2
            details.append(f"Operating margin stable or improving ({oldest_op_margin:.1%} -> {newest_op_margin:.1%})")
        elif newest_op_margin > 0:
            raw_score += 1
            details.append(f"Operating margin positive but slightly declined")
        else:
            details.append(f"Operating margin may be negative or uncertain")
    else:
        print(f"MISSING DATA: Not enough operating margin data points. Found {len(op_margins)} points, need at least 2")
        details.append("Not enough operating margin data points")

    # 2. Gross Margin Level
    gm_values = [fi['gross_margin'] for fi in financial_line_items if fi.get('gross_margin') is not None]
    if gm_values:
        # We'll just take the most recent
        recent_gm = gm_values[0]
        if recent_gm > 0.5:
            raw_score += 2
            details.append(f"Strong gross margin: {recent_gm:.1%}")
        elif recent_gm > 0.3:
            raw_score += 1
            details.append(f"Moderate gross margin: {recent_gm:.1%}")
        else:
            details.append(f"Low gross margin: {recent_gm:.1%}")
    else:
        print("MISSING DATA: No gross margin data available")
        details.append("No gross margin data available")

    # 3. Multi-year Margin Stability
    #   e.g. if we have at least 3 data points, see if standard deviation is low.
    if len(op_margins) >= 3:
        stdev = statistics.pstdev(op_margins)
        if stdev < 0.02:
            raw_score += 2
            details.append("Operating margin extremely stable over multiple years")
        elif stdev < 0.05:
            raw_score += 1
            details.append("Operating margin reasonably stable")
        else:
            details.append("Operating margin volatility is high")
    else:
        print(f"MISSING DATA: Not enough margin data points for volatility check. Found {len(op_margins)} points, need at least 3")
        details.append("Not enough margin data points for volatility check")

    # scale raw_score (max 6) to 0-10
    final_score = min(10, (raw_score / 6) * 10)
    return {"score": final_score, "details": "; ".join(details)}


def analyze_management_efficiency_leverage(financial_line_items: list) -> dict:
    """
    Evaluate management efficiency & leverage:
      - Return on Equity (ROE)
      - Debt-to-Equity ratio
      - Possibly check if free cash flow is consistently positive
    """
    if not financial_line_items:
        print("MISSING DATA: No financial data for management efficiency analysis")
        return {
            "score": 0,
            "details": "No financial data for management efficiency analysis",
        }

    details = []
    raw_score = 0  # up to 6 => scale to 0–10

    # 1. Return on Equity (ROE)
    ni_values = [fi['net_income'] for fi in financial_line_items if fi.get('net_income') is not None]
    eq_values = [fi['shareholders_equity'] for fi in financial_line_items if fi.get('shareholders_equity') is not None]
    if ni_values and eq_values and len(ni_values) == len(eq_values):
        recent_ni = ni_values[0]
        recent_eq = eq_values[0] if eq_values[0] else 1e-9
        if recent_ni > 0:
            roe = recent_ni / recent_eq
            if roe > 0.2:
                raw_score += 3
                details.append(f"High ROE: {roe:.1%}")
            elif roe > 0.1:
                raw_score += 2
                details.append(f"Moderate ROE: {roe:.1%}")
            elif roe > 0:
                raw_score += 1
                details.append(f"Positive but low ROE: {roe:.1%}")
            else:
                details.append(f"ROE is near zero or negative: {roe:.1%}")
        else:
            print("MISSING DATA: Recent net income is zero or negative, hurting ROE")
            details.append("Recent net income is zero or negative, hurting ROE")
    else:
        print("MISSING DATA: Insufficient data for ROE calculation")
        details.append("Insufficient data for ROE calculation")

    # 2. Debt-to-Equity
    debt_values = [fi['total_liabilities'] for fi in financial_line_items if fi.get('total_liabilities') is not None]
    if debt_values and eq_values and len(debt_values) == len(eq_values):
        recent_debt = debt_values[0]
        recent_equity = eq_values[0] if eq_values[0] else 1e-9
        dte = recent_debt / recent_equity
        if dte < 0.3:
            raw_score += 2
            details.append(f"Low debt-to-equity: {dte:.2f}")
        elif dte < 1.0:
            raw_score += 1
            details.append(f"Manageable debt-to-equity: {dte:.2f}")
        else:
            details.append(f"High debt-to-equity: {dte:.2f}")
    else:
        print("MISSING DATA: Insufficient data for debt/equity analysis")
        details.append("Insufficient data for debt/equity analysis")

    # 3. FCF Consistency
    fcf_values = [fi['free_cash_flow'] for fi in financial_line_items if fi.get('free_cash_flow') is not None]
    if fcf_values and len(fcf_values) >= 2:
        # Check if FCF is positive in recent years
        positive_fcf_count = sum(1 for x in fcf_values if x and x > 0)
        # We'll be simplistic: if most are positive, reward
        ratio = positive_fcf_count / len(fcf_values)
        if ratio > 0.8:
            raw_score += 1
            details.append(f"Majority of periods have positive FCF ({positive_fcf_count}/{len(fcf_values)})")
        else:
            details.append(f"Free cash flow is inconsistent or often negative")
    else:
        print("MISSING DATA: Insufficient or no FCF data to check consistency")
        details.append("Insufficient or no FCF data to check consistency")

    final_score = min(10, (raw_score / 6) * 10)
    return {"score": final_score, "details": "; ".join(details)}


def analyze_fisher_valuation(financial_line_items: list, market_cap: float | None) -> dict:
    """
    Phil Fisher is willing to pay for quality and growth, but still checks:
      - P/E
      - P/FCF
      - (Optionally) Enterprise Value metrics, but simpler approach is typical
    We will grant up to 2 points for each of two metrics => max 4 raw => scale to 0–10.
    """
    if not financial_line_items or market_cap is None:
        print("MISSING DATA: Insufficient data to perform valuation")
        return {"score": 0, "details": "Insufficient data to perform valuation"}

    details = []
    raw_score = 0

    # Gather needed data
    net_incomes = [fi['net_income'] for fi in financial_line_items if fi.get('net_income') is not None]
    fcf_values = [fi['free_cash_flow'] for fi in financial_line_items if fi.get('free_cash_flow') is not None]

    # 1) P/E
    if net_incomes and net_incomes[0] > 0:
        pe_ratio = market_cap / net_incomes[0]
        if pe_ratio < 15:
            raw_score += 2
            details.append(f"Attractive P/E ratio: {pe_ratio:.1f}")
        elif pe_ratio < 25:
            raw_score += 1
            details.append(f"Moderate P/E ratio: {pe_ratio:.1f}")
        else:
            details.append(f"High P/E ratio: {pe_ratio:.1f}")
    else:
        print("MISSING DATA: Cannot compute P/E ratio (no positive earnings)")
        details.append("Cannot compute P/E ratio (no positive earnings)")

    # 2) P/FCF
    if fcf_values and fcf_values[0] > 0:
        pfcf_ratio = market_cap / fcf_values[0]
        if pfcf_ratio < 15:
            raw_score += 2
            details.append(f"Attractive P/FCF ratio: {pfcf_ratio:.1f}")
        elif pfcf_ratio < 25:
            raw_score += 1
            details.append(f"Moderate P/FCF ratio: {pfcf_ratio:.1f}")
        else:
            details.append(f"High P/FCF ratio: {pfcf_ratio:.1f}")
    else:
        print("MISSING DATA: Cannot compute P/FCF ratio (no positive FCF)")
        details.append("Cannot compute P/FCF ratio (no positive FCF)")

    # Scale raw_score (max 4) to 0–10
    final_score = min(10, (raw_score / 4) * 10)
    return {"score": final_score, "details": "; ".join(details)}


def analyze_insider_activity(insider_trades: list) -> dict:
    """
    Simple insider-trade analysis:
      - If there's heavy insider buying, we nudge the score up.
      - If there's mostly selling, we reduce it.
      - Otherwise, neutral.
    """
    # Default is neutral (5/10).
    score = 5
    details = []

    if not insider_trades:
        print("MISSING DATA: No insider trades data; defaulting to neutral")
        details.append("No insider trades data; defaulting to neutral")
        return {"score": score, "details": "; ".join(details)}

    buys, sells = 0, 0
    for trade in insider_trades:
        # Check if trade is a dictionary
        if isinstance(trade, dict):
            # Determine if it's a buy or sale based on the 'Text' field
            if 'Text' in trade:
                text = trade['Text'].lower()
                if 'sale' in text:
                    sells += 1
                elif 'purchase' in text or 'buy' in text:
                    buys += 1
            # Fallback to checking 'Shares' key if 'Text' is not available
            elif 'Shares' in trade and trade['Shares'] is not None:
                # Default to counting as a buy if we can't determine from text
                buys += 1
        # Fallback for object-style access if needed
        elif hasattr(trade, 'text'):
            text = trade.text.lower()
            if 'sale' in text:
                sells += 1
            elif 'purchase' in text or 'buy' in text:
                buys += 1
        elif hasattr(trade, 'shares') and trade.shares is not None:
            # Default to counting as a buy if we can't determine from text
            buys += 1

    total = buys + sells
    if total == 0:
        print("MISSING DATA: No buy/sell transactions found; neutral")
        details.append("No buy/sell transactions found; neutral")
        return {"score": score, "details": "; ".join(details)}

    buy_ratio = buys / total
    if buy_ratio > 0.7:
        score = 8
        details.append(f"Heavy insider buying: {buys} buys vs. {sells} sells")
    elif buy_ratio > 0.4:
        score = 6
        details.append(f"Moderate insider buying: {buys} buys vs. {sells} sells")
    else:
        score = 4
        details.append(f"Mostly insider selling: {buys} buys vs. {sells} sells")

    return {"score": score, "details": "; ".join(details)}