def latest_line_item(line_items):
    """Return the most recent line item from the list."""
    return line_items[0] if line_items else None

def analyze_value(metrics, line_items, market_cap):
    """Free cash‑flow yield, EV/EBIT, other classic deep‑value metrics."""

    max_score = 6  # 4 pts for FCF‑yield, 2 pts for EV/EBIT
    score = 0
    details: list[str] = []

    # Free‑cash‑flow yield
    latest_item = latest_line_item(line_items)
    fcf = getattr(latest_item, "free_cash_flow", None) if latest_item else None
    if fcf is not None and market_cap:
        fcf_yield = fcf / market_cap
        if fcf_yield >= 0.15:
            score += 4
            details.append(f"Extraordinary FCF yield {fcf_yield:.1%}")
        elif fcf_yield >= 0.12:
            score += 3
            details.append(f"Very high FCF yield {fcf_yield:.1%}")
        elif fcf_yield >= 0.08:
            score += 2
            details.append(f"Respectable FCF yield {fcf_yield:.1%}")
        else:
            details.append(f"Low FCF yield {fcf_yield:.1%}")
    else:
        if fcf is None:
            print("WARNING: Free cash flow data is missing")
        if not market_cap:
            print("WARNING: Market cap data is missing")
        details.append("FCF data unavailable")

    # EV/EBIT (from financial metrics)
    if metrics:
        ev_ebit = getattr(metrics[0], "ev_to_ebit", None)
        if ev_ebit is not None:
            if ev_ebit < 6:
                score += 2
                details.append(f"EV/EBIT {ev_ebit:.1f} (<6)")
            elif ev_ebit < 10:
                score += 1
                details.append(f"EV/EBIT {ev_ebit:.1f} (<10)")
            else:
                details.append(f"High EV/EBIT {ev_ebit:.1f}")
        else:
            print("WARNING: EV/EBIT data is missing")
            details.append("EV/EBIT data unavailable")
    else:
        print("WARNING: Financial metrics data is missing")
        details.append("Financial metrics unavailable")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


# ----- Balance sheet --------------------------------------------------------

def analyze_balance_sheet(metrics, line_items):
    """Leverage and liquidity checks."""

    max_score = 3
    score = 0
    details: list[str] = []

    latest_metrics = metrics[0] if metrics else None
    latest_item = latest_line_item(line_items)

    debt_to_equity = getattr(latest_metrics, "debt_to_equity", None) if latest_metrics else None
    if debt_to_equity is not None:
        if debt_to_equity < 0.5:
            score += 2
            details.append(f"Low D/E {debt_to_equity:.2f}")
        elif debt_to_equity < 1:
            score += 1
            details.append(f"Moderate D/E {debt_to_equity:.2f}")
        else:
            details.append(f"High leverage D/E {debt_to_equity:.2f}")
    else:
        print("WARNING: Debt-to-equity data is missing")
        details.append("Debt‑to‑equity data unavailable")

    # Quick liquidity sanity check (cash vs total debt)
    if latest_item is not None:
        cash = getattr(latest_item, "cash_and_equivalents", None)
        total_debt = getattr(latest_item, "total_debt", None)
        if cash is not None and total_debt is not None:
            if cash > total_debt:
                score += 1
                details.append("Net cash position")
            else:
                details.append("Net debt position")
        else:
            if cash is None:
                print("WARNING: Cash and equivalents data is missing")
            if total_debt is None:
                print("WARNING: Total debt data is missing")
            details.append("Cash/debt data unavailable")
    else:
        print("WARNING: Latest line item data is missing")
        details.append("Cash/debt data unavailable")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


# ----- Insider activity -----------------------------------------------------

def analyze_insider_activity(insider_trades):
    """Net insider buying over the last 12 months acts as a hard catalyst."""

    max_score = 2
    score = 0
    details: list[str] = []

    if not insider_trades:
        print("WARNING: Insider trade data is missing")
        details.append("No insider trade data")
        return {"score": score, "max_score": max_score, "details": "; ".join(details)}

    shares_bought = sum(t.transaction_shares or 0 for t in insider_trades if (t.transaction_shares or 0) > 0)
    shares_sold = abs(sum(t.transaction_shares or 0 for t in insider_trades if (t.transaction_shares or 0) < 0))
    net = shares_bought - shares_sold
    if net > 0:
        score += 2 if net / max(shares_sold, 1) > 1 else 1
        details.append(f"Net insider buying of {net:,} shares")
    else:
        details.append("Net insider selling")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


# ----- Contrarian sentiment -------------------------------------------------

def simple_sentiment_analysis(text):
    """
    Perform basic sentiment analysis using keyword matching.
    Returns 'negative', 'positive', or 'neutral'.
    """
    # Convert to lowercase for case-insensitive matching
    text = text.lower()
    
    # Define keyword lists
    negative_keywords = {
        'decline', 'decrease', 'drop', 'fall', 'loss', 'lose', 'losing', 'lost',
        'risk', 'risky', 'concern', 'worried', 'worry', 'fear', 'afraid',
        'bankruptcy', 'bankrupt', 'default', 'debt', 'trouble', 'problem',
        'weak', 'weaken', 'weakening', 'poor', 'unstable', 'volatile',
        'sell', 'selling', 'sold', 'bearish', 'bear', 'crash', 'crashing',
        'negative', 'negatively', 'down', 'downward', 'downtrend'
    }
    
    positive_keywords = {
        'rise', 'rising', 'increase', 'increasing', 'growth', 'growing',
        'gain', 'gaining', 'profit', 'profitable', 'success', 'successful',
        'strong', 'strengthen', 'strengthening', 'improve', 'improving',
        'buy', 'buying', 'bought', 'bullish', 'bull', 'recovery', 'recovering',
        'positive', 'positively', 'up', 'upward', 'uptrend'
    }
    
    # Count keyword matches
    negative_count = sum(1 for word in negative_keywords if word in text)
    positive_count = sum(1 for word in positive_keywords if word in text)
    
    # Determine sentiment based on keyword counts
    if negative_count > positive_count + 1:  # Require a margin to be considered negative
        return 'negative'
    elif positive_count > negative_count + 1:  # Require a margin to be considered positive
        return 'positive'
    else:
        return 'neutral'

def analyze_contrarian_sentiment(news):
    """Very rough gauge: a wall of recent negative headlines can be a *positive* for a contrarian."""

    max_score = 1
    score = 0
    details: list[str] = []
    if not news:
        print("WARNING: News data is missing")
        details.append("No recent news")
        return {"score": score, "max_score": max_score, "details": "; ".join(details)}

    # Count negative sentiment articles using simple keyword analysis
    sentiment_negative_count = sum(
        1 for n in news if simple_sentiment_analysis(n["content"]['title']) == "negative"
    )
    
    if sentiment_negative_count >= 5:
        score += 1  # The more hated, the better (assuming fundamentals hold up)
        details.append(f"{sentiment_negative_count} negative headlines (contrarian opportunity)")
    else:
        details.append("Limited negative press")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}