import datetime
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
    get_revenue, get_net_income, get_eps, get_gross_profit, get_operating_income
)
import pandas as pd

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


def get_financial_data(ticker):
    """Get financial data for a stock ticker."""
    try:
        # Get fundamental metrics
        metrics = get_all_fundamental_metrics(ticker)
        if not metrics:
            return None
            
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
                    print(f"Error getting total liabilities from balance sheet: {str(e)}")
            
            if metrics.get('current_assets') is None or (isinstance(metrics['current_assets'], pd.Series) and metrics['current_assets'].empty):
                try:
                    if 'Total Current Assets' in balance_sheet_data.index:
                        metrics['current_assets'] = balance_sheet_data.loc['Total Current Assets']
                    elif 'Current Assets' in balance_sheet_data.index:
                        metrics['current_assets'] = balance_sheet_data.loc['Current Assets']
                except Exception as e:
                    print(f"Error getting current assets from balance sheet: {str(e)}")
            
            if metrics.get('current_liabilities') is None or (isinstance(metrics['current_liabilities'], pd.Series) and metrics['current_liabilities'].empty):
                try:
                    if 'Total Current Liabilities' in balance_sheet_data.index:
                        metrics['current_liabilities'] = balance_sheet_data.loc['Total Current Liabilities']
                    elif 'Current Liabilities' in balance_sheet_data.index:
                        metrics['current_liabilities'] = balance_sheet_data.loc['Current Liabilities']
                except Exception as e:
                    print(f"Error getting current liabilities from balance sheet: {str(e)}")
            
            if metrics.get('cash') is None or (isinstance(metrics['cash'], pd.Series) and metrics['cash'].empty):
                try:
                    if 'Cash' in balance_sheet_data.index:
                        metrics['cash'] = balance_sheet_data.loc['Cash']
                    elif 'Cash And Cash Equivalents' in balance_sheet_data.index:
                        metrics['cash'] = balance_sheet_data.loc['Cash And Cash Equivalents']
                except Exception as e:
                    print(f"Error getting cash from balance sheet: {str(e)}")
            
            if metrics.get('accounts_receivable') is None or (isinstance(metrics['accounts_receivable'], pd.Series) and metrics['accounts_receivable'].empty):
                try:
                    if 'Net Receivables' in balance_sheet_data.index:
                        metrics['accounts_receivable'] = balance_sheet_data.loc['Net Receivables']
                    elif 'Accounts Receivable' in balance_sheet_data.index:
                        metrics['accounts_receivable'] = balance_sheet_data.loc['Accounts Receivable']
                except Exception as e:
                    print(f"Error getting accounts receivable from balance sheet: {str(e)}")
            
            if metrics.get('shareholders_equity') is None or (isinstance(metrics['shareholders_equity'], pd.Series) and metrics['shareholders_equity'].empty):
                try:
                    if 'Total Stockholder Equity' in balance_sheet_data.index:
                        metrics['shareholders_equity'] = balance_sheet_data.loc['Total Stockholder Equity']
                    elif 'Stockholders Equity' in balance_sheet_data.index:
                        metrics['shareholders_equity'] = balance_sheet_data.loc['Stockholders Equity']
                except Exception as e:
                    print(f"Error getting shareholders equity from balance sheet: {str(e)}")
            
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
                print(f"Error getting depreciation/amortization: {str(e)}")
            
            # If still missing, try the individual getter functions as fallback
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
                print(f"Error getting cash flow data: {str(e)}")
        
        # Calculate financial ratios using yf_tools functions
        try:
            # Get all financial ratios at once
            financial_ratios = calculate_financial_ratios(ticker)
            for ratio_name, ratio_value in financial_ratios.items():
                if isinstance(ratio_value, pd.Series) and not ratio_value.empty:
                    metrics[ratio_name.lower().replace(' ', '_')] = ratio_value
            
            # Get individual ratios as backup
            roe = calculate_roe(ticker)
            if isinstance(roe, pd.Series) and not roe.empty:
                metrics['roe'] = roe
            
            debt_to_equity = calculate_debt_to_equity(ticker)
            if isinstance(debt_to_equity, pd.Series) and not debt_to_equity.empty:
                metrics['debt_to_equity'] = debt_to_equity
            
            current_ratio = calculate_current_ratio(ticker)
            if isinstance(current_ratio, pd.Series) and not current_ratio.empty:
                metrics['current_ratio'] = current_ratio
        except Exception as e:
            print(f"Error calculating financial ratios: {str(e)}")
        
        # Calculate earnings growth if missing
        if metrics.get('earnings_growth') is None and metrics.get('net_income') is not None:
            if isinstance(metrics['net_income'], pd.Series):
                metrics['earnings_growth'] = metrics['net_income'].pct_change()
        
        # Fix dividend yield calculation if it seems incorrect
        if metrics.get('dividend_yield') is not None:
            if isinstance(metrics['dividend_yield'], pd.Series):
                if not metrics['dividend_yield'].empty and metrics['dividend_yield'].iloc[0] > 100:
                    metrics['dividend_yield'] = metrics['dividend_yield'] / 100
            elif metrics['dividend_yield'] > 100:
                metrics['dividend_yield'] = metrics['dividend_yield'] / 100
        
        # Calculate P/E ratio if we have the data
        if metrics.get('eps') is not None and metrics.get('stock_price') is not None:
            if isinstance(metrics['eps'], pd.Series):
                if not metrics['eps'].empty and metrics['eps'].iloc[0] > 0:
                    metrics['pe_ratio'] = metrics['stock_price'] / metrics['eps']
            elif metrics['eps'] > 0:
                metrics['pe_ratio'] = metrics['stock_price'] / metrics['eps']
        
        # Calculate P/FCF ratio if we have the data
        if metrics.get('free_cash_flow') is not None and metrics.get('outstanding_shares') is not None:
            if isinstance(metrics['free_cash_flow'], pd.Series):
                if not metrics['free_cash_flow'].empty and metrics['free_cash_flow'].iloc[0] > 0:
                    fcf_per_share = metrics['free_cash_flow'] / metrics['outstanding_shares']
                    if fcf_per_share.iloc[0] > 0 and metrics.get('stock_price') is not None:
                        metrics['pfcf_ratio'] = metrics['stock_price'] / fcf_per_share
            elif metrics['free_cash_flow'] > 0:
                fcf_per_share = metrics['free_cash_flow'] / metrics['outstanding_shares']
                if fcf_per_share > 0 and metrics.get('stock_price') is not None:
                    metrics['pfcf_ratio'] = metrics['stock_price'] / fcf_per_share
        
        # Calculate EV/EBIT and EV/EBITDA if we have the data
        if metrics.get('total_liabilities') is not None and metrics.get('shareholders_equity') is not None:
            enterprise_value = metrics['total_liabilities'] + metrics['shareholders_equity']
            if metrics.get('operating_income') is not None:
                if isinstance(metrics['operating_income'], pd.Series):
                    if not metrics['operating_income'].empty and metrics['operating_income'].iloc[0] > 0:
                        metrics['ev_ebit'] = enterprise_value / metrics['operating_income']
                elif metrics['operating_income'] > 0:
                    metrics['ev_ebit'] = enterprise_value / metrics['operating_income']
            
            if metrics.get('ebitda') is not None:
                if isinstance(metrics['ebitda'], pd.Series):
                    if not metrics['ebitda'].empty and metrics['ebitda'].iloc[0] > 0:
                        metrics['ev_ebitda'] = enterprise_value / metrics['ebitda']
                elif metrics['ebitda'] > 0:
                    metrics['ev_ebitda'] = enterprise_value / metrics['ebitda']
        
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
        print(f"Error getting financial data for {ticker}: {str(e)}")
        return None


def analyze_stock(ticker):
    """Analyze a stock using multiple metrics and return a combined score."""
    print(f"DEBUG: Starting analysis for {ticker}")
    
    # Get financial data
    financial_data = get_financial_data(ticker)
    if not financial_data:
        print(f"DEBUG: No financial data available for {ticker}")
        return None
    
    # Get latest metrics
    latest_metrics = financial_data.get("latest_metrics", {})
    print(f"DEBUG: Latest metrics for {ticker}: {latest_metrics}")
    if not latest_metrics:
        print(f"DEBUG: No latest metrics available for {ticker}")
        return None
    
    # Get historical metrics
    historical_metrics = financial_data.get("historical_metrics", [])
    if not historical_metrics:
        print(f"DEBUG: No historical metrics available for {ticker}")
        return None
    
    # Extract market cap from latest metrics if available
    market_cap = latest_metrics.get('market_cap', 0)
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    if market_cap == 0:
        market_cap = get_market_cap(ticker)
    print(f"DEBUG: Market cap for {ticker} on {today}: {market_cap}")
    
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
    insider_activity = analyze_insider_activity(df_to_list(insider_trades(ticker)))
    contrarian_sentiment = analyze_contrarian_sentiment(company_news(ticker))
    
    # Calculate Charlie Munger metrics
    moat_strength = analyze_moat_strength(historical_metrics, historical_metrics)
    munger_management = munger_analyze_management_quality(historical_metrics, [])  # Empty list for insider trades
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
    fisher_insider = analyze_insider_activity(df_to_list(insider_trades(ticker)))
    
    # Calculate Stanley Druckenmiller metric
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
        fundamentals_score * 0.15 +  # Buffett
        consistency_score * 0.10 +   # Buffett
        moat_result["score"] * 0.10 +  # Buffett/Munger
        management_result["score"] * 0.10 +  # Buffett/Munger
        earnings_stability["score"] * 0.05 +  # Graham
        financial_strength["score"] * 0.05 +  # Graham
        business_quality["score"] * 0.05 +    # Ackman
        financial_discipline["score"] * 0.05 +  # Ackman
        disruptive_potential["score"] * 0.05 +  # Wood
        innovation_growth["score"] * 0.05 +    # Wood
        value_score["score"] * 0.05 +         # Burry
        balance_sheet_health["score"] * 0.05 +  # Burry
        moat_strength["score"] * 0.05 +       # Munger
        predictability["score"] * 0.05 +      # Munger
        lynch_growth["score"] * 0.05          # Lynch
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
    
    print(f"DEBUG: Completed analysis for {ticker}")
    return analysis


if __name__ == "__main__":
    print(analyze_stock("NVDA"))