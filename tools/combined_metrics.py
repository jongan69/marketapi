import datetime
import logging
from tools.yf_tools import (
    fetch_and_cache_stock_tickers, get_price_data, get_market_cap, stock_price, stock_info, 
    insider_trades, financial_metrics, company_news, 
    prices_to_df, balance_sheet, cash_flow, stock_dividends, 
    calculate_debt_to_equity, calculate_roe, calculate_current_ratio,
    calculate_financial_ratios, get_all_fundamental_metrics,
    calculate_profit_margin, calculate_operating_margin, calculate_gross_margin,
    calculate_earnings_growth, calculate_dividend_yield, get_outstanding_shares,
    calculate_free_cash_flow, get_operating_cash_flow, calculate_ebitda,
    get_research_and_development, get_selling_general_and_admin,
    get_total_assets, get_total_liabilities, get_current_assets,
    get_current_liabilities, get_cash, get_inventory,
    get_accounts_receivable, get_accounts_payable, get_shareholders_equity,
    get_revenue, get_net_income, get_eps, get_gross_profit, get_operating_income,
    prepare_financial_line_items
)
import pandas as pd
import numpy as np

from tools.ben_graham import analyze_earnings_stability, analyze_financial_strength, analyze_valuation_graham
from tools.bill_ackman import analyze_business_quality, analyze_financial_discipline, analyze_activism_potential, analyze_valuation
from tools.cathie_wood import analyze_disruptive_potential, analyze_innovation_growth, analyze_cathie_wood_valuation
from tools.michael_burry import analyze_value, analyze_balance_sheet, analyze_insider_activity, analyze_contrarian_sentiment
from tools.charlie_munger import analyze_moat_strength, analyze_management_quality as munger_analyze_management_quality, analyze_predictability
from tools.peter_lynch import analyze_lynch_growth, analyze_lynch_fundamentals, analyze_lynch_valuation, analyze_sentiment
from tools.phil_fisher import analyze_fisher_growth_quality, analyze_margins_stability, analyze_management_efficiency_leverage, analyze_fisher_valuation, analyze_insider_activity
from tools.stanley_druckenmiller import analyze_growth_and_momentum, analyze_risk_reward, analyze_druckenmiller_valuation
from tools.warren_buffett import analyze_fundamentals, analyze_consistency, analyze_moat, analyze_management_quality, calculate_owner_earnings, calculate_intrinsic_value
from tools.valuation import calculate_intrinsic_value
from tools.technicals import calculate_trend_signals
from tools.utils import df_to_list

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for validation
MIN_MARKET_CAP = 1_000_000  # $1M
MAX_MARKET_CAP = 3_000_000_000_000  # $3T
MIN_POSITIVE_VALUE = 1e-10  # Small positive number to avoid division by zero
MAX_REASONABLE_PE = 1000  # Maximum reasonable P/E ratio
MAX_REASONABLE_EV_RATIO = 100  # Maximum reasonable EV/EBIT or EV/EBITDA

def safe_divide(numerator, denominator, default=None):
    """Safely perform division with validation."""
    # Handle pandas Series
    if isinstance(numerator, pd.Series):
        numerator = numerator.iloc[0] if not numerator.empty else None
    if isinstance(denominator, pd.Series):
        denominator = denominator.iloc[0] if not denominator.empty else None
        
    if denominator is None or (isinstance(denominator, (int, float)) and abs(denominator) < MIN_POSITIVE_VALUE):
        return default
    return numerator / denominator

def validate_financial_ratio(ratio, ratio_name, min_value=None, max_value=None):
    """Validate a financial ratio and return None if invalid."""
    # Handle pandas Series
    if isinstance(ratio, pd.Series):
        ratio = ratio.iloc[0] if not ratio.empty else None
        
    if ratio is None:
        return None
    
    if min_value is not None and ratio < min_value:
        logger.warning(f"{ratio_name} below minimum threshold: {ratio}")
        return None
    
    if max_value is not None and ratio > max_value:
        logger.warning(f"{ratio_name} above maximum threshold: {ratio}")
        return None
    
    return ratio

def calculate_financial_ratios(metrics):
    """Calculate financial ratios with proper validation."""
    print(f"DEBUG: Calculating financial ratios using {metrics}")
    ratios = {}
    
    # Helper function to safely get metric value
    def get_metric_value(metric_name):
        value = metrics.get(metric_name)
        if isinstance(value, pd.Series):
            return value.iloc[0] if not value.empty else None
        return value
    
    # P/E Ratio
    eps = get_metric_value('eps')
    stock_price = get_metric_value('stock_price')
    if eps is not None and stock_price is not None:
        pe_ratio = safe_divide(stock_price, eps)
        pe_ratio = validate_financial_ratio(pe_ratio, "P/E Ratio", min_value=0, max_value=MAX_REASONABLE_PE)
        if pe_ratio is not None:
            ratios['pe_ratio'] = pe_ratio
    
    # P/FCF Ratio
    free_cash_flow = get_metric_value('free_cash_flow')
    outstanding_shares = get_metric_value('outstanding_shares')
    if free_cash_flow is not None and outstanding_shares is not None:
        fcf_per_share = safe_divide(free_cash_flow, outstanding_shares)
        if fcf_per_share is not None and fcf_per_share > 0 and stock_price is not None:
            pfcf_ratio = safe_divide(stock_price, fcf_per_share)
            pfcf_ratio = validate_financial_ratio(pfcf_ratio, "P/FCF Ratio", min_value=0, max_value=MAX_REASONABLE_PE)
            if pfcf_ratio is not None:
                ratios['pfcf_ratio'] = pfcf_ratio
    
    # EV/EBIT
    total_liabilities = get_metric_value('total_liabilities')
    shareholders_equity = get_metric_value('shareholders_equity')
    if total_liabilities is not None and shareholders_equity is not None:
        enterprise_value = total_liabilities + shareholders_equity
        operating_income = get_metric_value('operating_income')
        if operating_income is not None:
            ev_ebit = safe_divide(enterprise_value, operating_income)
            ev_ebit = validate_financial_ratio(ev_ebit, "EV/EBIT", min_value=0, max_value=MAX_REASONABLE_EV_RATIO)
            if ev_ebit is not None:
                ratios['ev_ebit'] = ev_ebit
        
        # EV/EBITDA
        ebitda = get_metric_value('ebitda')
        if ebitda is not None:
            ev_ebitda = safe_divide(enterprise_value, ebitda)
            ev_ebitda = validate_financial_ratio(ev_ebitda, "EV/EBITDA", min_value=0, max_value=MAX_REASONABLE_EV_RATIO)
            if ev_ebitda is not None:
                ratios['ev_ebitda'] = ev_ebitda
    
    # Debt to Equity
    if total_liabilities is not None and shareholders_equity is not None:
        debt_to_equity = safe_divide(total_liabilities, shareholders_equity)
        debt_to_equity = validate_financial_ratio(debt_to_equity, "Debt to Equity", min_value=0)
        if debt_to_equity is not None:
            ratios['debt_to_equity'] = debt_to_equity
    
    # Current Ratio
    current_assets = get_metric_value('current_assets')
    current_liabilities = get_metric_value('current_liabilities')
    if current_assets is not None and current_liabilities is not None:
        current_ratio = safe_divide(current_assets, current_liabilities)
        current_ratio = validate_financial_ratio(current_ratio, "Current Ratio", min_value=0)
        if current_ratio is not None:
            ratios['current_ratio'] = current_ratio
    
    # Return on Equity
    net_income = get_metric_value('net_income')
    if net_income is not None and shareholders_equity is not None:
        roe = safe_divide(net_income, shareholders_equity)
        roe = validate_financial_ratio(roe, "ROE", min_value=-1, max_value=1)
        if roe is not None:
            ratios['roe'] = roe
    
    return ratios

def get_financial_data(ticker):
    """Get financial data for a stock ticker."""
    try:
        # Get fundamental metrics
        metrics = get_all_fundamental_metrics(ticker)
        if not metrics:
            return None
            
        # Calculate and validate financial ratios
        ratios = calculate_financial_ratios(metrics)
        metrics.update(ratios)
        
        # Get balance sheet data directly
        balance_sheet_data = balance_sheet(ticker)
        if balance_sheet_data is not None and not balance_sheet_data.empty:
            # Try to extract missing metrics from balance sheet
            if metrics.get('total_liabilities') is None or (isinstance(metrics['total_liabilities'], pd.Series) and metrics['total_liabilities'].empty):
                try:
                    if 'Total Liab' in balance_sheet_data.index:
                        metrics['total_liabilities'] = balance_sheet_data.loc['Total Liab']
                    elif 'Total Liabilities' in balance_sheet_data.index:
                        metrics['total_liabilities'] = balance_sheet_data.loc['Total Liabilities']
                except Exception as e:
                    logger.error(f"Error getting total liabilities from balance sheet: {str(e)}")
            
            if metrics.get('current_assets') is None or (isinstance(metrics['current_assets'], pd.Series) and metrics['current_assets'].empty):
                try:
                    if 'Total Current Assets' in balance_sheet_data.index:
                        metrics['current_assets'] = balance_sheet_data.loc['Total Current Assets']
                    elif 'Current Assets' in balance_sheet_data.index:
                        metrics['current_assets'] = balance_sheet_data.loc['Current Assets']
                except Exception as e:
                    logger.error(f"Error getting current assets from balance sheet: {str(e)}")
            
            if metrics.get('current_liabilities') is None or (isinstance(metrics['current_liabilities'], pd.Series) and metrics['current_liabilities'].empty):
                try:
                    if 'Total Current Liabilities' in balance_sheet_data.index:
                        metrics['current_liabilities'] = balance_sheet_data.loc['Total Current Liabilities']
                    elif 'Current Liabilities' in balance_sheet_data.index:
                        metrics['current_liabilities'] = balance_sheet_data.loc['Current Liabilities']
                except Exception as e:
                    logger.error(f"Error getting current liabilities from balance sheet: {str(e)}")
            
            if metrics.get('cash') is None or (isinstance(metrics['cash'], pd.Series) and metrics['cash'].empty):
                try:
                    if 'Cash' in balance_sheet_data.index:
                        metrics['cash'] = balance_sheet_data.loc['Cash']
                    elif 'Cash And Cash Equivalents' in balance_sheet_data.index:
                        metrics['cash'] = balance_sheet_data.loc['Cash And Cash Equivalents']
                except Exception as e:
                    logger.error(f"Error getting cash from balance sheet: {str(e)}")
            
            if metrics.get('accounts_receivable') is None or (isinstance(metrics['accounts_receivable'], pd.Series) and metrics['accounts_receivable'].empty):
                try:
                    if 'Net Receivables' in balance_sheet_data.index:
                        metrics['accounts_receivable'] = balance_sheet_data.loc['Net Receivables']
                    elif 'Accounts Receivable' in balance_sheet_data.index:
                        metrics['accounts_receivable'] = balance_sheet_data.loc['Accounts Receivable']
                except Exception as e:
                    logger.error(f"Error getting accounts receivable from balance sheet: {str(e)}")
            
            if metrics.get('shareholders_equity') is None or (isinstance(metrics['shareholders_equity'], pd.Series) and metrics['shareholders_equity'].empty):
                try:
                    if 'Total Stockholder Equity' in balance_sheet_data.index:
                        metrics['shareholders_equity'] = balance_sheet_data.loc['Total Stockholder Equity']
                    elif 'Stockholders Equity' in balance_sheet_data.index:
                        metrics['shareholders_equity'] = balance_sheet_data.loc['Stockholders Equity']
                except Exception as e:
                    logger.error(f"Error getting shareholders equity from balance sheet: {str(e)}")
            
            # Try to get depreciation and amortization
            try:
                depreciation_keys = ['Depreciation', 'Depreciation And Amortization', 'Depreciation & Amortization', 'D&A']
                for key in depreciation_keys:
                    if key in balance_sheet_data.index:
                        metrics['depreciation'] = balance_sheet_data.loc[key]
                        break
                
                # Try to get maintenance capex
                capex_keys = ['Capital Expenditure', 'Capital Expenditures', 'CapEx']
                for key in capex_keys:
                    if key in balance_sheet_data.index:
                        metrics['maintenance_capex'] = balance_sheet_data.loc[key]
                        break
            except Exception as e:
                logger.error(f"Error getting depreciation/amortization: {str(e)}")
            
            # If still missing, try the individual getter functions as fallback
            if metrics.get('total_assets') is None or (isinstance(metrics['total_assets'], pd.Series) and metrics['total_assets'].empty):
                metrics['total_assets'] = get_total_assets(ticker)
                
            if metrics.get('total_liabilities') is None or (isinstance(metrics['total_liabilities'], pd.Series) and metrics['total_liabilities'].empty):
                metrics['total_liabilities'] = get_total_liabilities(ticker)
                
            if metrics.get('current_assets') is None or (isinstance(metrics['current_assets'], pd.Series) and metrics['current_assets'].empty):
                metrics['current_assets'] = get_current_assets(ticker)
                
            if metrics.get('current_liabilities') is None or (isinstance(metrics['current_liabilities'], pd.Series) and metrics['current_liabilities'].empty):
                metrics['current_liabilities'] = get_current_liabilities(ticker)
                
            if metrics.get('cash') is None or (isinstance(metrics['cash'], pd.Series) and metrics['cash'].empty):
                metrics['cash'] = get_cash(ticker)
                
            if metrics.get('accounts_receivable') is None or (isinstance(metrics['accounts_receivable'], pd.Series) and metrics['accounts_receivable'].empty):
                metrics['accounts_receivable'] = get_accounts_receivable(ticker)
                
            if metrics.get('shareholders_equity') is None or (isinstance(metrics['shareholders_equity'], pd.Series) and metrics['shareholders_equity'].empty):
                metrics['shareholders_equity'] = get_shareholders_equity(ticker)
                
            if metrics.get('return_on_equity') is None or (isinstance(metrics['return_on_equity'], pd.Series) and metrics['return_on_equity'].empty):
                metrics['return_on_equity'] = calculate_roe(ticker)
            
            if metrics.get('debt_to_equity') is None or (isinstance(metrics['debt_to_equity'], pd.Series) and metrics['debt_to_equity'].empty):
                metrics['debt_to_equity'] = calculate_debt_to_equity(ticker)
            
            if metrics.get('research_and_development') is None or (isinstance(metrics['research_and_development'], pd.Series) and metrics['research_and_development'].empty):
                metrics['research_and_development'] = get_research_and_development(ticker)
        
        # Get cash flow data for depreciation and capex
        cash_flow_data = cash_flow(ticker)
        if cash_flow_data is not None and not cash_flow_data.empty:
            try:
                # Try to get depreciation from cash flow statement if not found in balance sheet
                if metrics.get('depreciation') is None or (isinstance(metrics['depreciation'], pd.Series) and metrics['depreciation'].empty):
                    depreciation_keys = ['Depreciation', 'Depreciation And Amortization', 'Depreciation & Amortization', 'D&A']
                    for key in depreciation_keys:
                        if key in cash_flow_data.index:
                            metrics['depreciation'] = cash_flow_data.loc[key]
                            break
                
                # Try to get maintenance capex from cash flow statement if not found in balance sheet
                if metrics.get('maintenance_capex') is None or (isinstance(metrics['maintenance_capex'], pd.Series) and metrics['maintenance_capex'].empty):
                    capex_keys = ['Capital Expenditure', 'Capital Expenditures', 'CapEx']
                    for key in capex_keys:
                        if key in cash_flow_data.index:
                            metrics['maintenance_capex'] = cash_flow_data.loc[key]
                            break
            except Exception as e:
                logger.error(f"Error getting cash flow data: {str(e)}")
        
        # Extract the latest values from pandas Series
        latest_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, pd.Series):
                # Get the first non-NaN value
                non_nan_values = value.dropna()
                if not non_nan_values.empty:
                    latest_metrics[key] = float(non_nan_values.iloc[0])
                else:
                    latest_metrics[key] = None
            else:
                # For non-Series values (like outstanding_shares)
                latest_metrics[key] = value
        
        # Create historical metrics by extracting values from each Series
        historical_metrics = []
        if metrics:
            # Get all dates from the first Series
            first_series = next((v for v in metrics.values() if isinstance(v, pd.Series)), None)
            if first_series is not None:
                dates = first_series.index
                for date in dates:
                    period_metrics = {}
                    for key, value in metrics.items():
                        if isinstance(value, pd.Series) and date in value.index:
                            period_metrics[key] = float(value[date]) if pd.notna(value[date]) else None
                        elif not isinstance(value, pd.Series):
                            period_metrics[key] = value
                    historical_metrics.append(period_metrics)
        
        return {
            "latest_metrics": latest_metrics,
            "historical_metrics": historical_metrics
        }
    except Exception as e:
        logger.error(f"Error getting financial data for {ticker}: {str(e)}")
        return None


def calculate_earnings_growth_rate(historical_metrics):
    """Calculate the earnings growth rate from historical metrics."""
    if not historical_metrics or len(historical_metrics) < 2:
        return None, None
    
    # Extract net income values in chronological order
    net_incomes = []
    for metric in historical_metrics:
        if 'net_income' in metric and metric['net_income'] is not None:
            net_incomes.append(metric['net_income'])
    
    if len(net_incomes) < 2:
        return None, None
    
    # Calculate year-over-year growth rates
    growth_rates = []
    for i in range(1, len(net_incomes)):
        if net_incomes[i-1] != 0:  # Avoid division by zero
            growth_rate = (net_incomes[i] - net_incomes[i-1]) / abs(net_incomes[i-1])
            growth_rates.append(growth_rate)
    
    if not growth_rates:
        return None, None
    
    # Calculate average growth rate
    avg_growth_rate = sum(growth_rates) / len(growth_rates)
    
    # Check if growth is consistent (all positive or all negative)
    is_consistent = all(rate > 0 for rate in growth_rates) or all(rate < 0 for rate in growth_rates)
    
    return avg_growth_rate, is_consistent

def analyze_consistency(historical_metrics):
    """Analyze earnings consistency with improved logic."""
    if not historical_metrics:
        return 0.0
    
    growth_rate, is_consistent = calculate_earnings_growth_rate(historical_metrics)
    if growth_rate is None:
        return 0.0
    
    score = 0.0
    
    # Award points for consistent growth/decline
    if is_consistent:
        score += 3.0
        logger.debug("Added 3 points for consistent earnings trend")
    
    # Award points based on growth rate magnitude
    if abs(growth_rate) > 0.5:  # 50% growth/decline
        score += 2.0
        logger.debug(f"Added 2 points for significant growth rate: {growth_rate:.2%}")
    elif abs(growth_rate) > 0.2:  # 20% growth/decline
        score += 1.0
        logger.debug(f"Added 1 point for moderate growth rate: {growth_rate:.2%}")
    
    # Normalize score to 0-1 range
    normalized_score = min(score / 5.0, 1.0)
    logger.debug(f"Final consistency score: {score}, Normalized: {normalized_score:.2f}")
    
    return normalized_score

def analyze_operating_margins(historical_metrics):
    """Analyze operating margins with improved logic."""
    if not historical_metrics:
        return {"score": 0.0, "reasons": ["No historical data available"]}
    
    margins = []
    for metric in historical_metrics:
        if 'operating_margin' in metric and metric['operating_margin'] is not None:
            margins.append(metric['operating_margin'])
    
    if not margins:
        return {"score": 0.0, "reasons": ["No operating margin data available"]}
    
    # Calculate margin stability
    margin_changes = [margins[i] - margins[i-1] for i in range(1, len(margins))]
    is_improving = all(change >= 0 for change in margin_changes)
    is_stable = all(abs(change) < 0.05 for change in margin_changes)  # Less than 5% change
    
    # Calculate average margin
    avg_margin = sum(margins) / len(margins)
    latest_margin = margins[-1]
    
    score = 0.0
    reasons = []
    
    # Score based on margin level
    if latest_margin > 0.15:  # 15% margin
        score += 3.0
        reasons.append("Strong operating margins (>15%)")
    elif latest_margin > 0.10:  # 10% margin
        score += 2.0
        reasons.append("Good operating margins (10-15%)")
    elif latest_margin > 0.05:  # 5% margin
        score += 1.0
        reasons.append("Moderate operating margins (5-10%)")
    elif latest_margin > 0:
        reasons.append("Weak but positive operating margins")
    else:
        reasons.append(f"Negative operating margins: {latest_margin:.1%}")
    
    # Score based on trend
    if is_improving:
        score += 2.0
        reasons.append("Consistently improving margins")
    elif is_stable:
        score += 1.0
        reasons.append("Stable margins")
    else:
        reasons.append("Unstable margins")
    
    # Normalize score to 0-1 range
    normalized_score = min(score / 5.0, 1.0)
    
    return {
        "score": normalized_score,
        "reasons": reasons,
        "latest_margin": latest_margin,
        "avg_margin": avg_margin,
        "is_improving": is_improving,
        "is_stable": is_stable
    }

def get_insider_activity_data(ticker):
    """Get insider activity data once and cache it."""
    try:
        insider_data = df_to_list(insider_trades(ticker))
        if not insider_data:
            logger.warning(f"No insider trading data available for {ticker}")
            return None
        return insider_data
    except Exception as e:
        logger.error(f"Error getting insider trading data for {ticker}: {str(e)}")
        return None

def analyze_stock(ticker):
    """Analyze a stock using multiple metrics and return a combined score."""
    logger.info(f"Starting analysis for {ticker}")
    
    # Get financial data
    financial_data = get_financial_data(ticker)
    if not financial_data:
        logger.warning(f"No financial data available for {ticker}")
        return None
    
    # Get latest metrics
    latest_metrics = financial_data.get("latest_metrics", {})
    if not latest_metrics:
        logger.warning(f"No latest metrics available for {ticker}")
        return None
    
    # Get historical metrics
    historical_metrics = financial_data.get("historical_metrics", [])
    if not historical_metrics:
        logger.warning(f"No historical metrics available for {ticker}")
        return None
    
    # Validate market cap
    market_cap = latest_metrics.get('market_cap', 0)
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    if market_cap == 0:
        market_cap = get_market_cap(ticker)
    
    if not MIN_MARKET_CAP <= market_cap <= MAX_MARKET_CAP:
        logger.warning(f"Suspicious market cap for {ticker}: {market_cap}")
    
    logger.info(f"Market cap for {ticker} on {today}: {market_cap}")
    
    # Get insider activity data once
    insider_data = get_insider_activity_data(ticker)
    
    # Get financial_line_items
    # financial_line_items = prepare_financial_line_items(ticker)
    
    # Calculate Warren Buffett metrics
    fundamentals_score, fundamentals_reasons = analyze_fundamentals(latest_metrics)
    consistency_score = analyze_consistency(historical_metrics)
    moat_result = analyze_moat(historical_metrics)
    management_result = analyze_management_quality(historical_metrics)
    owner_earnings = calculate_owner_earnings(latest_metrics)
    intrinsic_value = calculate_intrinsic_value(latest_metrics)
    
    # Calculate Ben Graham metrics
    earnings_stability = analyze_earnings_stability(historical_metrics, historical_metrics)
    financial_strength = analyze_financial_strength(historical_metrics, historical_metrics)
    graham_valuation = analyze_valuation_graham(historical_metrics, historical_metrics, market_cap)
    
    # Calculate Bill Ackman metrics
    business_quality = analyze_business_quality(historical_metrics, historical_metrics)
    financial_discipline = analyze_financial_discipline(historical_metrics, historical_metrics)
    activism_potential = analyze_activism_potential(historical_metrics, historical_metrics)
    ackman_valuation = analyze_valuation(historical_metrics, historical_metrics)
    
    # Calculate Cathie Wood metrics
    disruptive_potential = analyze_disruptive_potential(historical_metrics, historical_metrics)
    innovation_growth = analyze_innovation_growth(historical_metrics, historical_metrics)
    wood_valuation = analyze_cathie_wood_valuation(historical_metrics, market_cap)
    
    # Calculate Michael Burry metrics
    value_score = analyze_value(historical_metrics, historical_metrics, market_cap)
    balance_sheet_health = analyze_balance_sheet(historical_metrics, historical_metrics)
    insider_activity = analyze_insider_activity(insider_data) if insider_data else {"score": 0.0, "reasons": ["No insider data"]}
    contrarian_sentiment = analyze_contrarian_sentiment(company_news(ticker))
    
    # Calculate Charlie Munger metrics
    moat_strength = analyze_moat_strength(historical_metrics, historical_metrics)
    munger_management = munger_analyze_management_quality(historical_metrics, insider_data)
    predictability = analyze_predictability(historical_metrics)
    
    # Calculate Peter Lynch metrics
    lynch_growth = analyze_lynch_growth(historical_metrics)
    lynch_fundamentals = analyze_lynch_fundamentals(historical_metrics)
    lynch_valuation = analyze_lynch_valuation(historical_metrics, market_cap)
    sentiment_score = analyze_sentiment(historical_metrics)
    
    # Calculate Phil Fisher metrics
    fisher_growth_quality = analyze_fisher_growth_quality(historical_metrics)
    margins_stability = analyze_margins_stability(historical_metrics)
    management_efficiency = analyze_management_efficiency_leverage(historical_metrics)
    fisher_valuation = analyze_fisher_valuation(historical_metrics, market_cap)
    fisher_insider = insider_activity
    
    # Calculate Stanley Druckenmiller metrics
    price_data = df_to_list(get_price_data(ticker))
    growth_momentum = analyze_growth_and_momentum(historical_metrics, price_data)
    risk_reward = analyze_risk_reward(historical_metrics, market_cap, price_data)
    druckenmiller_valuation = analyze_druckenmiller_valuation(historical_metrics, market_cap)
    
    # Calculate technical signals
    prices_df = pd.DataFrame(price_data).rename(columns={
        'Close': 'close',
        'Volume': 'volume',
        'High': 'high',
        'Low': 'low',
        'Open': 'open'
    })
    technical_signals = calculate_trend_signals(prices_df)
    
    # Combine scores with weighted average
    combined_score = (
        fundamentals_score * 0.15 +
        consistency_score * 0.10 +
        moat_result["score"] * 0.10 +
        management_result["score"] * 0.10 +
        earnings_stability["score"] * 0.05 +
        financial_strength["score"] * 0.05 +
        business_quality["score"] * 0.05 +
        financial_discipline["score"] * 0.05 +
        disruptive_potential["score"] * 0.05 +
        innovation_growth["score"] * 0.05 +
        value_score["score"] * 0.05 +
        balance_sheet_health["score"] * 0.05 +
        moat_strength["score"] * 0.05 +
        predictability["score"] * 0.05 +
        lynch_growth["score"] * 0.05
    )
    
    # Prepare detailed analysis
    analysis = {
        "ticker": ticker,
        "combined_score": combined_score,
        "warren_buffett": {
            "fundamentals": {
                "score": fundamentals_score,
                "reasons": fundamentals_reasons
            },
            "consistency": {
                "score": consistency_score
            },
            "moat": moat_result,
            "management": management_result,
            "owner_earnings": owner_earnings,
            "intrinsic_value": intrinsic_value
        },
        "ben_graham": {
            "earnings_stability": earnings_stability,
            "financial_strength": financial_strength,
            "valuation": graham_valuation
        },
        "bill_ackman": {
            "business_quality": business_quality,
            "financial_discipline": financial_discipline,
            "activism_potential": activism_potential,
            "valuation": ackman_valuation
        },
        "cathie_wood": {
            "disruptive_potential": disruptive_potential,
            "innovation_growth": innovation_growth,
            "valuation": wood_valuation
        },
        "michael_burry": {
            "value_score": value_score,
            "balance_sheet_health": balance_sheet_health,
            "insider_activity": insider_activity,
            "contrarian_sentiment": contrarian_sentiment
        },
        "charlie_munger": {
            "moat_strength": moat_strength,
            "management_quality": munger_management,
            "predictability": predictability
        },
        "peter_lynch": {
            "growth": lynch_growth,
            "fundamentals": lynch_fundamentals,
            "valuation": lynch_valuation,
            "sentiment": sentiment_score
        },
        "phil_fisher": {
            "growth_quality": fisher_growth_quality,
            "margins_stability": margins_stability,
            "management_efficiency": management_efficiency,
            "valuation": fisher_valuation,
            "insider_activity": fisher_insider
        },
        "stanley_druckenmiller": {
            "growth_momentum": growth_momentum,
            "risk_reward": risk_reward,
            "valuation": druckenmiller_valuation
        },
        "technical_analysis": technical_signals
    }
    
    logger.info(f"Completed analysis for {ticker}")
    return analysis


if __name__ == "__main__":
    print(analyze_stock("NVDA"))