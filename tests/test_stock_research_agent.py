import importlib.util
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

AGENT_PATH = (
    Path(__file__).parents[1] / "agents" / "11-stock-research-agent" / "agent.py"
)
SPEC = importlib.util.spec_from_file_location("stock_research_agent", AGENT_PATH)
agent = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(agent)


class MappingFastInfo:
    def __init__(self, values, failing_keys=()):
        self.values = values
        self.failing_keys = set(failing_keys)

    def get(self, key, default=None):
        if key in self.failing_keys:
            raise RuntimeError(f"could not fetch {key}")
        return self.values.get(key, default)


class FakeTicker:
    def __init__(
        self, info=None, info_error=None, fast_info=None, fast_info_error=None
    ):
        self.info = info or {}
        self.info_error = info_error
        self._fast_info = fast_info
        self.fast_info_error = fast_info_error

    def get_info(self):
        if self.info_error:
            raise self.info_error
        return self.info

    @property
    def fast_info(self):
        if self.fast_info_error:
            raise self.fast_info_error
        return self._fast_info


def ticker_factory(ticker):
    return SimpleNamespace(Ticker=lambda symbol: ticker)


class StockResearchAgentTests(unittest.TestCase):
    def test_full_fundamentals_are_mapped(self):
        ticker = FakeTicker(
            info={
                "longName": "Tesla, Inc.",
                "sector": "Consumer Cyclical",
                "industry": "Auto Manufacturers",
                "currentPrice": 321.5,
                "marketCap": 1_000_000_000,
                "trailingPE": 100.0,
                "forwardPE": 80.0,
                "pegRatio": 2.0,
                "revenueGrowth": 0.2,
                "profitMargins": 0.1,
                "dividendYield": None,
                "fiftyTwoWeekHigh": 400.0,
                "fiftyTwoWeekLow": 150.0,
                "recommendationKey": "buy",
                "targetMeanPrice": 350.0,
                "longBusinessSummary": "A company summary.",
            }
        )

        with patch.object(agent, "yf", ticker_factory(ticker)):
            data = agent.get_stock_data(" tsla ")

        self.assertEqual(data["ticker"], "TSLA")
        self.assertEqual(data["name"], "Tesla, Inc.")
        self.assertEqual(data["price"], 321.5)
        self.assertEqual(data["target_price"], 350.0)
        self.assertEqual(data["data_source"], "Yahoo Finance fundamentals")
        self.assertNotIn("warning", data)
        self.assertNotIn("error", data)

    def test_rate_limit_uses_price_history_fallback(self):
        ticker = FakeTicker(
            info_error=RuntimeError("HTTP 429: Too Many Requests"),
            fast_info=MappingFastInfo(
                {
                    "last_price": 321.5,
                    "market_cap": 1_000_000_000,
                    "year_high": 400.0,
                    "year_low": 150.0,
                }
            ),
        )

        with patch.object(agent, "yf", ticker_factory(ticker)):
            data = agent.get_stock_data("nvda")

        self.assertEqual(data["ticker"], "NVDA")
        self.assertEqual(data["price"], 321.5)
        self.assertEqual(data["market_cap"], 1_000_000_000)
        self.assertEqual(data["52w_low"], 150.0)
        self.assertIn("limited price data", data["warning"])
        self.assertEqual(
            data["data_source"], "Yahoo Finance price history (limited fundamentals)"
        )
        self.assertEqual(data["pe_ratio"], "N/A")
        self.assertNotIn("error", data)

    def test_one_fallback_field_failure_does_not_hide_price(self):
        ticker = FakeTicker(
            info_error=RuntimeError("rate limited"),
            fast_info=MappingFastInfo(
                {"last_price": 99.5, "year_high": 120.0, "year_low": 80.0},
                failing_keys={"market_cap"},
            ),
        )

        with patch.object(agent, "yf", ticker_factory(ticker)):
            data = agent.get_stock_data("aapl")

        self.assertEqual(data["price"], 99.5)
        self.assertEqual(data["market_cap"], "N/A")
        self.assertNotIn("error", data)

    def test_complete_yahoo_failure_is_truthful(self):
        ticker = FakeTicker(
            info_error=RuntimeError("rate limited"),
            fast_info_error=RuntimeError("price history unavailable"),
        )

        with patch.object(agent, "yf", ticker_factory(ticker)):
            data = agent.get_stock_data("tsla")

        self.assertEqual(data["price"], "N/A")
        self.assertIn("usable price", data["error"])
        self.assertNotIn("mock", data)

    def test_missing_yfinance_returns_actionable_error(self):
        with patch.object(agent, "HAS_YFINANCE", False):
            data = agent.get_stock_data("tsla")

        self.assertIn("pip install", data["error"])
        self.assertNotIn("mock", data)

    def test_formatters_preserve_unavailable_values(self):
        self.assertEqual(agent.format_number("N/A"), "N/A")
        self.assertEqual(agent.format_number(None), "N/A")
        self.assertEqual(agent.format_number(1_250_000_000), "$1.25B")
        self.assertEqual(agent.format_price("N/A"), "N/A")
        self.assertEqual(agent.format_price(12.5), "$12.50")

    def test_data_only_mode_skips_openai(self):
        data = {
            "ticker": "TSLA",
            "name": "Tesla, Inc.",
            "sector": "Automotive",
            "industry": "Auto Manufacturers",
            "price": 321.5,
            "market_cap": 1_000_000_000,
            "pe_ratio": "N/A",
            "forward_pe": "N/A",
            "peg_ratio": "N/A",
            "52w_high": 400.0,
            "52w_low": 150.0,
            "analyst_rating": "N/A",
            "target_price": "N/A",
        }
        with (
            patch.object(agent, "get_stock_data", return_value=data),
            patch.object(agent, "analyze_stock") as analyze,
            patch.object(sys, "argv", ["agent.py", "--ticker", "tsla", "--data-only"]),
        ):
            output = io.StringIO()
            with patch("sys.stdout", output):
                result = agent.main()

        self.assertEqual(result, 0)
        analyze.assert_not_called()
        self.assertIn("Data-only mode", output.getvalue())

    def test_data_error_skips_openai_and_returns_failure(self):
        with (
            patch.object(
                agent, "get_stock_data", return_value={"error": "retry later"}
            ),
            patch.object(agent, "analyze_stock") as analyze,
            patch.object(sys, "argv", ["agent.py", "--ticker", "tsla"]),
        ):
            error = io.StringIO()
            with patch("sys.stdout", io.StringIO()), patch("sys.stderr", error):
                result = agent.main()

        self.assertEqual(result, 1)
        analyze.assert_not_called()
        self.assertIn("retry later", error.getvalue())

    def test_normal_mode_analyzes_successful_data(self):
        data = {
            "ticker": "AAPL",
            "name": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "price": 200.0,
            "market_cap": 2_000_000_000,
            "pe_ratio": 30.0,
            "forward_pe": 28.0,
            "peg_ratio": 2.0,
            "52w_high": 220.0,
            "52w_low": 150.0,
            "analyst_rating": "buy",
            "target_price": 215.0,
        }
        with (
            patch.object(agent, "get_stock_data", return_value=data),
            patch.object(agent, "analyze_stock", return_value="analysis") as analyze,
            patch.object(sys, "argv", ["agent.py", "--ticker", "aapl"]),
            patch("sys.stdout", io.StringIO()),
        ):
            result = agent.main()

        self.assertEqual(result, 0)
        analyze.assert_called_once_with(data)


if __name__ == "__main__":
    unittest.main()
