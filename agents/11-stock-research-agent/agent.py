"""
Stock Research Agent using Agno + Yahoo Finance.

Provides comprehensive stock analysis: price data, financials,
analyst ratings, and AI-powered investment summary.

Usage:
    python agent.py --ticker AAPL
    python agent.py --ticker NVDA
"""

import argparse
import math
import sys
from numbers import Real

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

try:
    import yfinance as yf

    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

MISSING = "N/A"


def _clean_value(value, default=MISSING):
    """Return a printable value, replacing null and non-finite numbers."""
    if value is None:
        return default
    if isinstance(value, Real) and not math.isfinite(float(value)):
        return default
    return value


def _safe_fast_value(fast_info, key):
    """Read one lazily fetched fast-info field without aborting the request."""
    try:
        return _clean_value(fast_info.get(key))
    except Exception:  # noqa: BLE001 - yfinance errors vary by version
        # yfinance may fetch each fast-info field independently. A failure in
        # market cap, for example, must not hide an otherwise usable price.
        return MISSING


def _usable_price(value) -> bool:
    """Whether a value can be used as the current stock price."""
    try:
        return _clean_value(value) != MISSING and float(value) > 0
    except (TypeError, ValueError):
        return False


def _fast_info_fallback(stock) -> dict:
    """Fetch fields backed by price history when quote fundamentals fail."""
    try:
        fast_info = stock.fast_info
    except Exception:  # noqa: BLE001 - yfinance errors vary by version
        return {}

    values = {
        "price": _safe_fast_value(fast_info, "last_price"),
        "market_cap": _safe_fast_value(fast_info, "market_cap"),
        "52w_high": _safe_fast_value(fast_info, "year_high"),
        "52w_low": _safe_fast_value(fast_info, "year_low"),
    }
    return {key: value for key, value in values.items() if value != MISSING}


def get_stock_data(ticker: str) -> dict:
    symbol = ticker.strip().upper()
    empty_data = {
        "ticker": symbol,
        "name": symbol,
        "sector": MISSING,
        "industry": MISSING,
        "price": MISSING,
        "market_cap": MISSING,
        "pe_ratio": MISSING,
        "forward_pe": MISSING,
        "peg_ratio": MISSING,
        "revenue_growth": MISSING,
        "profit_margin": MISSING,
        "dividend_yield": MISSING,
        "52w_high": MISSING,
        "52w_low": MISSING,
        "analyst_rating": MISSING,
        "target_price": MISSING,
        "description": MISSING,
    }

    if not HAS_YFINANCE:
        return {
            **empty_data,
            "error": "yfinance is not installed. Run: pip install -r requirements.txt",
        }

    stock = yf.Ticker(symbol)
    info_error = None
    try:
        info = stock.get_info() or {}
    except Exception as exc:  # noqa: BLE001 - yfinance errors vary by version
        # Yahoo's quoteSummary endpoint commonly responds with HTTP 429. Keep
        # the agent usable by switching to the price-history-backed fast_info
        # surface instead of failing the entire report.
        info = {}
        info_error = exc

    data = {
        **empty_data,
        "name": _clean_value(info.get("longName"), symbol),
        "sector": _clean_value(info.get("sector")),
        "industry": _clean_value(info.get("industry")),
        "price": _clean_value(info.get("currentPrice", info.get("regularMarketPrice"))),
        "market_cap": _clean_value(info.get("marketCap")),
        "pe_ratio": _clean_value(info.get("trailingPE")),
        "forward_pe": _clean_value(info.get("forwardPE")),
        "peg_ratio": _clean_value(info.get("pegRatio")),
        "revenue_growth": _clean_value(info.get("revenueGrowth")),
        "profit_margin": _clean_value(info.get("profitMargins")),
        "dividend_yield": _clean_value(info.get("dividendYield")),
        "52w_high": _clean_value(info.get("fiftyTwoWeekHigh")),
        "52w_low": _clean_value(info.get("fiftyTwoWeekLow")),
        "analyst_rating": _clean_value(info.get("recommendationKey")),
        "target_price": _clean_value(info.get("targetMeanPrice")),
        "description": _clean_value(info.get("longBusinessSummary"), MISSING),
    }

    if info_error or not _usable_price(data["price"]):
        fallback = _fast_info_fallback(stock)
        for key, value in fallback.items():
            if data[key] == MISSING:
                data[key] = value

    if not _usable_price(data["price"]):
        reason = (
            type(info_error).__name__
            if info_error
            else "incomplete Yahoo Finance response"
        )
        return {
            **data,
            "error": f"Yahoo Finance did not return a usable price for {symbol} ({reason}). Retry later.",
        }

    if info_error:
        data["data_source"] = "Yahoo Finance price history (limited fundamentals)"
        data["warning"] = (
            "Yahoo Finance full fundamentals were unavailable; using limited price data "
            f"after {type(info_error).__name__}."
        )
    elif any(
        data[key] == MISSING
        for key in ("sector", "industry", "pe_ratio", "target_price")
    ):
        data["data_source"] = "Yahoo Finance fundamentals with price-history fallback"
        data["warning"] = (
            "Some Yahoo Finance fundamentals were unavailable; missing values are shown as N/A."
        )
    else:
        data["data_source"] = "Yahoo Finance fundamentals"

    if data["description"] != MISSING:
        data["description"] = str(data["description"])[:500]
    return data


def analyze_stock(data: dict) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    stock_info = "\n".join(
        f"{k}: {v}"
        for k, v in data.items()
        if k not in {"description", "warning", "error", "data_source"}
    )

    messages = [
        SystemMessage(
            content="You are a financial analyst. Provide a concise stock analysis covering: Investment Thesis (2-3 sentences), Key Strengths (3 bullets), Key Risks (3 bullets), Valuation Assessment, and a Verdict (Buy/Hold/Sell with brief reasoning). Keep it under 300 words. Use only the supplied data; never invent values for fields marked N/A."
        ),
        HumanMessage(
            content=(
                f"Analyze this stock:\n{stock_info}\n\n"
                f"Data source: {data.get('data_source', MISSING)}\n"
                f"Availability note: {data.get('warning', 'All requested fields available.')}\n"
                f"Company description: {data.get('description', MISSING)}"
            )
        ),
    ]

    response = llm.invoke(messages)
    return response.content


def format_number(n) -> str:
    if n is None:
        return MISSING
    if isinstance(n, Real) and math.isfinite(float(n)):
        if n >= 1e12:
            return f"${n / 1e12:.2f}T"
        if n >= 1e9:
            return f"${n / 1e9:.2f}B"
        if n >= 1e6:
            return f"${n / 1e6:.2f}M"
        return f"${n:.2f}"
    return str(n)


def format_price(value) -> str:
    """Format a price without turning an unavailable value into $0.00."""
    if isinstance(value, Real) and math.isfinite(float(value)):
        return f"${value:,.2f}"
    return MISSING


def _configure_output_encoding() -> None:
    """Keep the emoji-based CLI usable on Windows' legacy console encoding."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8")
            except (AttributeError, ValueError):
                pass


def main():
    _configure_output_encoding()
    parser = argparse.ArgumentParser(description="Stock Research Agent")
    parser.add_argument(
        "--ticker", required=True, help="Stock ticker symbol (e.g., AAPL)"
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Print market data without calling OpenAI",
    )
    args = parser.parse_args()
    symbol = args.ticker.strip().upper()

    print(f"\n📈 Researching {symbol}...\n")

    data = get_stock_data(symbol)
    if data.get("error"):
        print(f"❌ {data['error']}", file=sys.stderr)
        return 1

    if data.get("warning"):
        print(f"⚠️  {data['warning']}\n")

    print("=" * 60)
    print(f"📊 {data.get('name', symbol)} ({symbol})")
    print("=" * 60)
    print(
        f"Price: {format_price(data.get('price'))}  |  Market Cap: {format_number(data.get('market_cap'))}"
    )
    print(f"Sector: {data.get('sector')}  |  Industry: {data.get('industry')}")
    print(
        f"P/E: {data.get('pe_ratio')}  |  Forward P/E: {data.get('forward_pe')}  |  PEG: {data.get('peg_ratio')}"
    )
    print(
        f"52W Range: {format_price(data.get('52w_low'))} - {format_price(data.get('52w_high'))}"
    )
    analyst_rating = data.get("analyst_rating") or "N/A"
    print(
        f"Analyst: {str(analyst_rating).upper()}  |  Target: {format_price(data.get('target_price'))}"
    )

    if args.data_only:
        print("\nℹ️  Data-only mode: skipped AI analysis.")
        return 0

    print("\n🤖 AI Analysis:")
    print("-" * 40)
    analysis = analyze_stock(data)
    print(analysis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
