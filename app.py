import streamlit as st
import pandas as pd
import json
import re

# Wrap imports in try-except for better error handling
try:
    from analysis_utils import get_investment_crew, run_investment_analysis, PortfolioAllocation
except ImportError as e:
    st.error(f"Failed to import analysis_utils: {e}")
    st.error("Please ensure analysis_utils.py is in the same directory and all dependencies are installed.")
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="Investment Portfolio Analysis Agent",
    page_icon="🤖",
    layout="wide"
)


# --- Caching Wrapper Function ---
@st.cache_data(ttl=1800)  # Cache for 30 minutes (better for volatile markets)
def run_cached_analysis(investment_amount, matured_value, days, risk, stocks_tuple):
    """Wrapper function to cache the complex analysis."""
    print("--- CACHE MISS: Running full analysis ---")
    
    llm, guardrail_crew, investment_crew = get_investment_crew()  # Note: 3 return values now
    
    if llm is None or guardrail_crew is None or investment_crew is None:
        return {"error": "Could not initialize AI Crews."}

    inputs = {
        "investment_amount": investment_amount,
        "matured_value_expectation": matured_value,
        "investment_days": days,
        "risk_tolerance": risk,
        "stocks_to_analyze": list(stocks_tuple)
    }

    try:
        result_data = run_investment_analysis(llm, guardrail_crew, investment_crew, inputs)
        return result_data
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}


# --- Streamlit UI ---

st.title("🤖 Investment Portfolio Analysis Crew")
st.markdown("Enter your investment goals and up to 10 BSE/NSE stock tickers to get a portfolio analysis.")

# Add cache control in sidebar
with st.sidebar:
    st.subheader("⚙️ Settings")
    if st.button("🗑️ Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared successfully!")
        st.rerun()
    
    st.info("📊 Cache TTL: 10 minutes\n\nResults are cached to improve performance. Clear cache to force fresh analysis.")

# --- 1. Get User Inputs ---

col1, col2 = st.columns(2)

with col1:
    st.subheader("Investment Goals")
    investment_amount = st.number_input(
        "Initial Investment Amount (INR)", 
        min_value=1000.0, 
        value=50000.0, 
        step=1000.0
    )
    matured_value_expectation = st.number_input(
        "Target Value (INR)", 
        min_value=investment_amount + 100.0,  # Dynamic validation restored
        value=max(55000.0, investment_amount + 5000.0),  # Ensure default is always valid
        step=1000.0,
        help="Must be greater than initial investment amount"
    )
    investment_days = st.number_input(
        "Investment Horizon (in days)", 
        min_value=30, 
        value=365, 
        step=1
    )
    risk_tolerance = st.selectbox(
        "Risk Tolerance", 
        ["low", "medium", "high"], 
        index=1
    )

with col2:
    st.subheader("Stocks to Analyze (Max 10)")
    stocks_input = st.text_area(
        "Enter BSE/NSE stock tickers, separated by commas",
        "RELIANCE.BO, TCS.BO, INFY.NS, HDFCBANK.BO",
        height=150,
        help="Use .BO for BSE stocks and .NS for NSE stocks"
    )
    
    # Helper text for tickers
    st.info("✅ Valid formats:\n- BSE: TICKER.BO (e.g., RELIANCE.BO)\n- NSE: TICKER.NS (e.g., INFY.NS)")

# --- 2. Analysis Button and Logic ---

if st.button("🚀 Analyze Portfolio", use_container_width=True):
    
    # --- 2.A. Validate Inputs ---
    
    # Clean and parse stock tickers
    stocks_to_analyze = [
        ticker.strip().upper() for ticker in re.split(r'[,\s\n]+', stocks_input) if ticker.strip()
    ]
    
    valid_inputs = True
    
    # Validation 1: Check if stocks are provided
    if not stocks_to_analyze:
        st.error("❌ Please enter at least one stock ticker.")
        valid_inputs = False
    
    # Validation 2: Check ticker format
    invalid_tickers = [
        ticker for ticker in stocks_to_analyze 
        if not (ticker.endswith('.BO') or ticker.endswith('.NS'))
    ]
    if invalid_tickers:
        st.error(f"❌ Invalid ticker format: {', '.join(invalid_tickers)}")
        st.error("All tickers must end with .BO (BSE) or .NS (NSE)")
        valid_inputs = False
    
    # Validation 3: Check maximum stocks
    if len(stocks_to_analyze) > 10:
        st.error(f"❌ You entered {len(stocks_to_analyze)} tickers. Maximum allowed is 10.")
        valid_inputs = False
    
    # Validation 4: Check target value (this check is technically redundant now)
    if matured_value_expectation <= investment_amount:
        st.error("❌ Target Value must be greater than Initial Investment Amount.")
        valid_inputs = False

    if valid_inputs:
        with st.spinner("🔄 Your AI crew is analyzing the market... (This may take a moment)"):
            try:
                # --- 2.B. Run Cached Analysis ---
                
                # Create a hashable key (keep original order by using tuple directly)
                stocks_tuple = tuple(stocks_to_analyze)
                
                # Call the cached function
                result_data = run_cached_analysis(
                    investment_amount,
                    matured_value_expectation,
                    investment_days,
                    risk_tolerance,
                    stocks_tuple
                )

                # --- 2.C. Handle Results or Errors ---
                
                # Check for errors
                if isinstance(result_data, dict) and "error" in result_data:
                    st.error(f"❌ {result_data['error']}")
                    result_data = None

                # --- 2.D. Display Results ---
                if result_data:
                    st.success("✅ Analysis Complete!")
                    st.caption("💡 Results are cached for 10 minutes. Use sidebar to clear cache for fresh analysis.")
                    
                    res_col1, res_col2 = st.columns(2)
                    
                    with res_col1:
                        st.metric(
                            label="Expected Annual Return",
                            value=f"{result_data.get('expected_annual_return', 0) * 100:.2f}%"
                        )
                    with res_col2:
                        st.metric(
                            label="Expected Annual Volatility",
                            value=f"{result_data.get('expected_annual_volatility', 0) * 100:.2f}%"
                        )

                    st.subheader("📈 Portfolio Allocation")

                    # Create a DataFrame for the allocation table
                    alloc_data = []
                    
                    total_investment = investment_amount
                    allocations_percent = result_data.get("allocations_percent", {})

                    for ticker, percent in allocations_percent.items():
                        # Calculate the INR amount based on percentage
                        inr_amount = total_investment * (percent / 100.0)
                        
                        alloc_data.append({
                            "Stock Ticker": ticker,
                            "Allocation (%)": f"{percent:.2f}%",
                            "Allocation (INR)": f"₹{inr_amount:,.2f}",
                            "Rationale": result_data.get("stock_rationales", {}).get(ticker, "N/A")
                        })
                    
                    if alloc_data:
                        alloc_df = pd.DataFrame(alloc_data)
                        st.dataframe(
                            alloc_df, 
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Rationale": st.column_config.TextColumn("Rationale", width="large")
                            }
                        )
                    else:
                        st.warning("⚠️ No allocation data available.")
                    
                    st.subheader("📋 Overall Rationale")
                    st.markdown(result_data.get('overall_rationale', 'No rationale provided.'))

                    # Optional: Display raw JSON
                    with st.expander("🔍 Show Raw JSON Output"):
                        st.json(result_data)

                else:
                    st.error("❌ Analysis failed. The crew could not generate a portfolio.")
                    st.info("This might be due to:")
                    st.markdown("""
                    - Invalid stock tickers
                    - API rate limits or errors
                    - Network connectivity issues
                    - Compliance blocks from the AI model
                    """)

            except Exception as e:
                st.error(f"❌ An unexpected error occurred: {str(e)}")
                with st.expander("🐛 Show Error Details"):
                    st.exception(e)

# Footer
st.markdown("---")
st.caption("⚠️ Disclaimer: This tool is for educational purposes only. Not financial advice. Consult a financial advisor before investing.")