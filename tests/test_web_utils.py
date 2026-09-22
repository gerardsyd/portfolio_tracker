import pandas as pd

from utils import web_utils


def test_portfolio_display_labels_use_dynamic_base_currency():
    labels = web_utils.portfolio_display_labels("USD")

    assert labels["LastPrice"] == "Last Price†"
    assert labels["CurrVal"] == "Current Value*"
    assert labels["AvgCost"] == "Average Cost*"
    assert labels["Cost"] == "Cost Base*"
    assert labels["%UnRlGain"] == "Unrealised Gain (%)"
    assert web_utils.portfolio_currency_legend("USD") == (
        "* Converted to portfolio base currency (USD). "
        "† Displayed in the security's local trading currency."
    )


def test_pandas_table_styler_renders_labels_without_changing_data_columns():
    frame = pd.DataFrame(
        [{
            "Ticker": "GOOGL",
            "LastPrice": 354.97,
            "CurrVal": 18442.33,
            "AvgCost": -511.00,
            "%UnRlGain": -0.0246,
        }]
    )

    html = web_utils.pandas_table_styler(
        frame,
        neg_cols=["%UnRlGain"],
        left_align_cols=["Ticker"],
        ticker_links=False,
        uuid="labels",
        base_currency="AUD",
    )

    assert "Last Price†" in html
    assert "Current Value*" in html
    assert "Average Cost*" in html
    assert "Unrealised Gain (%)" in html
    assert "AUD" not in html
    assert list(frame.columns) == ["Ticker", "LastPrice", "CurrVal", "AvgCost", "%UnRlGain"]
