import tools.yf_tools as yf_tools
import tools.phil_fisher as phil_fisher

line_items = yf_tools.prepare_financial_line_items("AAPL")
print(phil_fisher.analyze_fisher_growth_quality(line_items))