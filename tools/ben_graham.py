import math

def analyze_earnings_stability(metrics: list, financial_line_items: list) -> dict:
    """
    Graham wants at least several years of consistently positive earnings (ideally 5+).
    We'll check:
    1. Number of years with positive EPS.
    2. Growth in EPS from first to last period.
    """
    score = 0
    details = []

    if not metrics or not financial_line_items:
        print("MISSING DATA: No metrics or financial_line_items provided for earnings stability analysis")
        return {"score": score, "details": "Insufficient data for earnings stability analysis"}

    eps_vals = []
    for item in metrics:
        if 'eps' in item and item['eps'] is not None:
            eps_vals.append(item['eps'])
        else:
            print(f"MISSING DATA: EPS value missing or None in metrics item: {item}")

    if len(eps_vals) < 2:
        print(f"MISSING DATA: Only {len(eps_vals)} EPS values found, need at least 2 for analysis")
        details.append("Not enough multi-year EPS data.")
        return {"score": score, "details": "; ".join(details)}

    # 1. Consistently positive EPS
    positive_eps_years = sum(1 for e in eps_vals if e > 0)
    total_eps_years = len(eps_vals)
    if positive_eps_years == total_eps_years:
        score += 3
        details.append("EPS was positive in all available periods.")
    elif positive_eps_years >= (total_eps_years * 0.8):
        score += 2
        details.append("EPS was positive in most periods.")
    else:
        details.append("EPS was negative in multiple periods.")

    # 2. EPS growth from earliest to latest
    if eps_vals[-1] > eps_vals[0]:
        score += 1
        details.append("EPS grew from earliest to latest period.")
    else:
        details.append("EPS did not grow from earliest to latest period.")

    return {"score": score, "details": "; ".join(details)}


def analyze_financial_strength(metrics: list, financial_line_items: list) -> dict:
    """
    Graham checks liquidity (current ratio >= 2), manageable debt,
    and dividend record (preferably some history of dividends).
    """
    score = 0
    details = []

    if not financial_line_items:
        print("MISSING DATA: No financial_line_items provided for financial strength analysis")
        return {"score": score, "details": "No data for financial strength analysis"}

    latest_item = financial_line_items[-1]
    
    # Safely extract values with proper type conversion
    try:
        total_assets = float(latest_item.get('total_assets', 0) or 0)
        total_liabilities = float(latest_item.get('total_liabilities', 0) or 0)
        current_assets = float(latest_item.get('current_assets', 0) or 0)
        current_liabilities = float(latest_item.get('current_liabilities', 0) or 0)
        
        # Print missing data warnings
        if 'total_assets' not in latest_item or latest_item['total_assets'] is None:
            print("MISSING DATA: total_assets is missing or None")
        if 'total_liabilities' not in latest_item or latest_item['total_liabilities'] is None:
            print("MISSING DATA: total_liabilities is missing or None")
        if 'current_assets' not in latest_item or latest_item['current_assets'] is None:
            print("MISSING DATA: current_assets is missing or None")
        if 'current_liabilities' not in latest_item or latest_item['current_liabilities'] is None:
            print("MISSING DATA: current_liabilities is missing or None")
            
    except (ValueError, TypeError) as e:
        print(f"MISSING DATA: Error converting financial values to numbers: {e}")
        return {"score": score, "details": "Error converting financial values to numbers"}

    # 1. Current ratio
    if current_liabilities > 0:
        current_ratio = current_assets / current_liabilities
        if current_ratio >= 2.0:
            score += 2
            details.append(f"Current ratio = {current_ratio:.2f} (>=2.0: solid).")
        elif current_ratio >= 1.5:
            score += 1
            details.append(f"Current ratio = {current_ratio:.2f} (moderately strong).")
        else:
            details.append(f"Current ratio = {current_ratio:.2f} (<1.5: weaker liquidity).")
    else:
        print("MISSING DATA: Cannot compute current ratio (missing or zero current_liabilities)")
        details.append("Cannot compute current ratio (missing or zero current_liabilities).")

    # 2. Debt vs. Assets
    if total_assets > 0:
        debt_ratio = total_liabilities / total_assets
        if debt_ratio < 0.5:
            score += 2
            details.append(f"Debt ratio = {debt_ratio:.2f}, under 0.50 (conservative).")
        elif debt_ratio < 0.8:
            score += 1
            details.append(f"Debt ratio = {debt_ratio:.2f}, somewhat high but could be acceptable.")
        else:
            details.append(f"Debt ratio = {debt_ratio:.2f}, quite high by Graham standards.")
    else:
        print("MISSING DATA: Cannot compute debt ratio (missing total_assets)")
        details.append("Cannot compute debt ratio (missing total_assets).")

    # 3. Dividend track record
    try:
        div_periods = [float(item.get('dividend_yield', 0) or 0) for item in financial_line_items if 'dividend_yield' in item]
        if div_periods:
            # In many data feeds, dividend outflow is shown as a negative number
            # (money going out to shareholders). We'll consider any negative as 'paid a dividend'.
            div_paid_years = sum(1 for d in div_periods if d > 0)
            if div_paid_years > 0:
                # e.g. if at least half the periods had dividends
                if div_paid_years >= (len(div_periods) // 2 + 1):
                    score += 1
                    details.append("Company paid dividends in the majority of the reported years.")
                else:
                    details.append("Company has some dividend payments, but not most years.")
            else:
                details.append("Company did not pay dividends in these periods.")
        else:
            print("MISSING DATA: No dividend_yield data available in financial_line_items")
            details.append("No dividend data available to assess payout consistency.")
    except (ValueError, TypeError) as e:
        print(f"MISSING DATA: Error processing dividend data: {e}")
        details.append("Error processing dividend data.")

    return {"score": score, "details": "; ".join(details)}


def analyze_valuation_graham(metrics: list, financial_line_items: list, market_cap: float) -> dict:
    """
    Core Graham approach to valuation:
    1. Net-Net Check: (Current Assets - Total Liabilities) vs. Market Cap
    2. Graham Number: sqrt(22.5 * EPS * Book Value per Share)
    3. Compare per-share price to Graham Number => margin of safety
    """
    if not financial_line_items or not market_cap or market_cap <= 0:
        print("MISSING DATA: No financial_line_items or invalid market_cap for valuation analysis")
        return {"score": 0, "details": "Insufficient data to perform valuation"}

    latest = financial_line_items[-1]
    
    # Safely extract values with proper type conversion
    try:
        current_assets = float(latest.get('current_assets', 0) or 0)
        total_liabilities = float(latest.get('total_liabilities', 0) or 0)
        book_value_ps = float(latest.get('book_value_per_share', 0) or 0)
        eps = float(latest.get('eps', 0) or 0)
        shares_outstanding = float(latest.get('outstanding_shares', 0) or 0)
        
        # Print missing data warnings
        if 'current_assets' not in latest or latest['current_assets'] is None:
            print("MISSING DATA: current_assets is missing or None")
        if 'total_liabilities' not in latest or latest['total_liabilities'] is None:
            print("MISSING DATA: total_liabilities is missing or None")
        if 'book_value_per_share' not in latest or latest['book_value_per_share'] is None:
            print("MISSING DATA: book_value_per_share is missing or None")
        if 'eps' not in latest or latest['eps'] is None:
            print("MISSING DATA: eps is missing or None")
        if 'outstanding_shares' not in latest or latest['outstanding_shares'] is None:
            print("MISSING DATA: outstanding_shares is missing or None")
            
    except (ValueError, TypeError) as e:
        print(f"MISSING DATA: Error converting financial values to numbers: {e}")
        return {"score": 0, "details": "Error converting financial values to numbers"}

    details = []
    score = 0

    # 1. Net-Net Check
    #   NCAV = Current Assets - Total Liabilities
    #   If NCAV > Market Cap => historically a strong buy signal
    net_current_asset_value = current_assets - total_liabilities
    if net_current_asset_value > 0 and shares_outstanding > 0:
        net_current_asset_value_per_share = net_current_asset_value / shares_outstanding
        price_per_share = market_cap / shares_outstanding if shares_outstanding else 0

        details.append(f"Net Current Asset Value = {net_current_asset_value:,.2f}")
        details.append(f"NCAV Per Share = {net_current_asset_value_per_share:,.2f}")
        details.append(f"Price Per Share = {price_per_share:,.2f}")

        if net_current_asset_value > market_cap:
            score += 4  # Very strong Graham signal
            details.append("Net-Net: NCAV > Market Cap (classic Graham deep value).")
        else:
            # For partial net-net discount
            if net_current_asset_value_per_share >= (price_per_share * 0.67):
                score += 2
                details.append("NCAV Per Share >= 2/3 of Price Per Share (moderate net-net discount).")
    else:
        print(f"MISSING DATA: NCAV not exceeding market cap or insufficient data for net-net approach. NCAV: {net_current_asset_value}, Shares: {shares_outstanding}")
        details.append("NCAV not exceeding market cap or insufficient data for net-net approach.")

    # 2. Graham Number
    #   GrahamNumber = sqrt(22.5 * EPS * BVPS).
    #   Compare the result to the current price_per_share
    #   If GrahamNumber >> price, indicates undervaluation
    graham_number = None
    if eps > 0 and book_value_ps > 0:
        graham_number = math.sqrt(22.5 * eps * book_value_ps)
        details.append(f"Graham Number = {graham_number:.2f}")
    else:
        print(f"MISSING DATA: Unable to compute Graham Number. EPS: {eps}, Book Value PS: {book_value_ps}")
        details.append("Unable to compute Graham Number (EPS or Book Value missing/<=0).")

    # 3. Margin of Safety relative to Graham Number
    if graham_number and shares_outstanding > 0:
        current_price = market_cap / shares_outstanding
        if current_price > 0:
            margin_of_safety = (graham_number - current_price) / current_price
            details.append(f"Margin of Safety (Graham Number) = {margin_of_safety:.2%}")
            if margin_of_safety > 0.5:
                score += 3
                details.append("Price is well below Graham Number (>=50% margin).")
            elif margin_of_safety > 0.2:
                score += 1
                details.append("Some margin of safety relative to Graham Number.")
            else:
                details.append("Price close to or above Graham Number, low margin of safety.")
        else:
            print("MISSING DATA: Current price is zero or invalid; can't compute margin of safety")
            details.append("Current price is zero or invalid; can't compute margin of safety.")
    # else: already appended details for missing graham_number

    return {"score": score, "details": "; ".join(details)}
