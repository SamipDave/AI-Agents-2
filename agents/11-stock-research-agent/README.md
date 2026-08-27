# Stock Research Agent

Fetches real-time stock data via Yahoo Finance and generates an AI-powered investment analysis.

**Framework**: LangChain  
**LLM**: GPT-4o-mini  
**Data**: Yahoo Finance (free, no API key needed)

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

## Run

```bash
python agent.py --ticker AAPL
python agent.py --ticker NVDA
python agent.py --ticker TSLA

# Fetch market data without an OpenAI call (useful for rate-limit troubleshooting)
python agent.py --ticker TSLA --data-only
```

## Output

Real-time fundamentals + AI analysis covering investment thesis, strengths, risks, valuation, and verdict.

Yahoo Finance can temporarily throttle its fundamentals endpoint with HTTP 429. When that happens, the
agent automatically falls back to price-history-backed fields (`price`, market cap, and 52-week range),
marks unavailable fundamentals as `N/A`, and continues without fabricating values. If Yahoo cannot return
even a usable price, the command exits cleanly with an actionable error instead of calling the LLM.

Use `--data-only` to verify Yahoo data fetching without an OpenAI API key or an AI-analysis cost.

## Tests

From the repository root, run the deterministic offline suite:

```bash
python -B -m unittest discover -s tests -p "test_stock_research_agent.py" -v
```
