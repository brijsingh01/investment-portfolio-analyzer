# Investment Portfolio Analyzer

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor
[![CrewAI](https://img.shields.io/badge/CrewAI-2C3E50?style=for-the-badge&logo=crewai&logoColorA multi-agent AI-powered Streamlit app for analyzing investment portfolios using Indian stocks (BSE/NSE). Leveraging CrewAI agents for data fetching, news sentiment analysis, quantitative metrics, and ethical compliance checks, this tool generates optimized portfolio allocations based on your goals and risk tolerance. Built with guardrails for security and regulatory adherence, it's designed for educational and exploratory purposes.

> **⚠️ Disclaimer:** This app is for informational and learning purposes only. It does not constitute financial, investment, or legal advice. All outputs are based on historical data and AI-generated insights, which may not predict future performance. Investments involve risks, including loss of principal. Always consult a qualified financial advisor before making decisions.

## Features

- **Input Validation**: AI agent scans for PII, injections, and invalid data (e.g., non-.BO/.NS tickers, unrealistic amounts).
- **Goal Feasibility Analysis**: Assesses if your target returns are feasible, ambitious, or unrealistic given risk tolerance.
- **Market News & Sentiment**: Fetches latest news and analyst views via Tavily for provided stocks.
- **Quantitative Analysis**: Pulls historical prices, returns, volatility, PE ratios, sectors, and correlations using yfinance.
- **Portfolio Optimization**: Synthesizes data into percentage and INR allocations with rationales, expected return/volatility.
- **Compliance Guardrails**: AI reviews outputs for prohibited language (e.g., "guaranteed returns") and adds disclaimers.
- **User-Friendly Interface**: Streamlit-based UI with graceful error handling—always provides guidance, never raw exceptions.
- **Extensible**: Easy to add more agents/tools for advanced features like risk simulations.

## Prerequisites

- **Python 3.9+**: Ensure Python is installed (check with `python --version`).
- **Git**: For cloning the repo and deployment.
- **API Keys**:
  - [Google Gemini API Key](https://aistudio.google.com/app/apikey): For the LLM powering CrewAI agents.
  - [Tavily API Key](https://app.tavily.com/home): For news and sentiment search (free tier available, but limits apply).
- **Hardware**: At least 4GB RAM for local runs (CrewAI + yfinance can be compute-intensive); internet connection required for data fetching.

No additional hardware like GPUs needed—runs on CPU.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/brijsingh01/investment-portfolio-analyzer.git
   cd investment-portfolio-analyzer
   ```


2. **Create a Virtual Environment** (Recommended to isolate dependencies):
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   Create a `requirements.txt` file with the following content (copy-paste into your project root):

   ```
   streamlit==1.43.2
   crewai==0.51.1
   yfinance==0.2.43
   pandas==2.2.3
   numpy==1.26.4
   pydantic==2.9.2
   tavily-python==0.6.0
   google-generativeai==0.8.3
   ```

   Then install:
   ```bash
   pip install -r requirements.txt
   ```

   - **Why these versions?** Tested for compatibility with CrewAI and Streamlit as of November 2025. Update if needed, but check for breaking changes.
   - If issues arise (e.g., with CrewAI tools), run `pip install --upgrade crewai` separately.

## Configuration

1. **Set Up API Keys**:
   - Create a `.env` file in your project root (add to `.gitignore` to avoid committing secrets):
     ```
     GEMINI_API_KEY=your_gemini_api_key_here
     TAVILY_API_KEY=your_tavily_api_key_here
     ```
   - The app falls back to environment variables for deployment. For local runs, ensure they are set:
     ```bash
     # On Windows (Command Prompt)
     set GEMINI_API_KEY=your_key

     # On macOS/Linux
     export GEMINI_API_KEY=your_key
     export TAVILY_API_KEY=your_key
     ```

   - **Obtaining Keys**:
     - Gemini: Sign up at [Google AI Studio](https://aistudio.google.com), generate an API key (free tier: 15 RPM).
     - Tavily: Register at [Tavily](https://app.tavily.com), get a key (free: 1000 queries/month).

2. **Project Structure** (After setup):
   ```
   investment-portfolio-analyzer/
   ├── app.py                  # Main Streamlit app (run with: streamlit run app.py)
   ├── analysis_utils.py       # Multi Agent Logic
   ├── requirements.txt        # Dependencies
   ├── .streamlit              # Secrets (gitignored)
   └── README.md               # This file
   ```
   - The core code (e.g., agents, tools, models) should be in `app.py` or a separate `core.py`. Copy the full code from previous interactions into `app.py`.


## Running Locally

1. **Start the App**:
   ```bash
   streamlit run app.py
   ```
   - This launches the app at `http://localhost:8501`. Open in your browser.

2. **Interact with the App**:
   - Enter investment details: Amount (INR), Target Value (INR), Days Horizon, Risk Tolerance (e.g., "low", "medium", "high"), and Stock Tickers (e.g., "RELIANCE.BO,TCS.NS").
   - Click "Analyze Portfolio" to run the CrewAI workflow.
   - View results: Feasibility verdict, news summaries, metrics, and allocations (with disclaimers).

3. **Expected Output**:
   - JSON-formatted portfolio with percentages, INR values, rationales, and expected metrics.
   - If issues occur, you'll see user-friendly guidance (e.g., "Refine your inputs") with precautions.

## Usage Examples

- **Sample Input**:
  - Amount: 100000
  - Target: 120000
  - Days: 365
  - Risk: "medium"
  - Tickers: ["RELIANCE.BO", "TCS.NS", "HDFCBANK.NS"]

- **Output Preview** (Simplified):
  ```json
  {
    "allocations_percent": {"RELIANCE.BO": 0.4, "TCS.NS": 0.3, "HDFCBANK.NS": 0.3},
    "allocations_inr": {"RELIANCE.BO": 40000, ...},
    "expected_annual_return": 0.15,
    "expected_annual_volatility": 0.20,
    "overall_rationale": "Balanced for medium risk... **Disclaimer:** Past performance not indicative of future results.",
    "stock_rationales": {"RELIANCE.BO": "Strong sector growth..."}
  }
  ```

- **Customization**: Edit agent goals in `app.py` to tweak behavior (e.g., add more stocks or tools).

## Important Disclaimers

- **Not Financial Advice**: Outputs are AI-generated simulations. No guarantees on accuracy or returns. Markets are volatile.
- **Data Sources**: Relies on yfinance (historical) and Tavily (news)—delays or inaccuracies possible.
- **Limitations**: Designed for BSE/NSE stocks; not for crypto/forex. Free APIs have quotas.
- **Ethical Use**: The app blocks unsafe inputs and flags non-compliant outputs. Use responsibly.
- **Liability**: Developers not liable for decisions based on this tool.

For real investments, use certified platforms and professionals.

## Troubleshooting

- **API Errors**: Check keys in env. Test with `curl` or Postman.
- **CrewAI Fails**: Lower LLM temperature or increase `max_iter=20` in crew init. Ensure internet for yfinance/Tavily.
- **Streamlit Issues**: Run `streamlit hello` to test install. For deployment, check platform logs.
- **No Data for Tickers**: Use valid .BO (BSE) or .NS (NSE) suffixes; e.g., "INFY.NS".
- **Performance Slow**: Limit tickers to 5; cache with `@st.cache_resource`.
- **Common Fixes**:
  - `ModuleNotFoundError`: Re-run `pip install -r requirements.txt`.
  - JSON Parsing: Update code with extraction helpers if LLM outputs verbose text.
  - Deployment Errors: Ensure `app.py` has no syntax issues; configure ports correctly (e.g., $PORT on Heroku).

If stuck, open an issue on GitHub with logs (sanitize keys!).

## Contributing

Contributions welcome! Fork the repo, make changes, and submit a PR:
1. Add features (e.g., more tools, UI tweaks).
2. Fix bugs in guardrails or tools.
3. Improve docs or add tests.

Follow PEP 8 for code. Test locally before PRs. Thanks for contributing to ethical AI in finance!

## License

MIT License – Feel free to use, modify, and distribute. See [LICENSE](LICENSE) for details.

***

**Questions?** File an issue or discuss on GitHub. Happy investing (responsibly)! ���


