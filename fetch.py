# fetch.py
# Stock Market Analysis Tool (Yahoo fallback + Stooq support)

import random
import time
from datetime import datetime, date

import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr

import dash
from dash import dcc, html
import plotly.graph_objects as go

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from rich.console import Console
from rich.table import Table

# ---- Optional caching for fewer rate-limit issues ----
try:
    import requests_cache

    requests_cache.install_cache(
        cache_name="yf_cache",
        backend="sqlite",
        expire_after=60 * 30
    )

    from urllib3.util.retry import Retry
    from requests.adapters import HTTPAdapter
    import requests

    def make_polite_session():
        s = requests_cache.CachedSession("yf_cache", expire_after=60 * 30)
        s.headers["User-Agent"] = "Mozilla/5.0 (StockTool)"
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"]
        )
        s.mount("https://", HTTPAdapter(max_retries=retries))
        return s

    POLITE_SESSION = make_polite_session()
except Exception:
    POLITE_SESSION = None

# ---- Date parsing ----
try:
    from dateutil import parser as dateparser
except Exception:
    dateparser = None

console = Console()


def parse_date(s):
    if not s or not s.strip():
        return None
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt)
        except:
            pass
    if dateparser:
        return dateparser.parse(s, dayfirst=True)
    raise ValueError("Invalid date. Use YYYY-MM-DD or DD/MM/YYYY")


def normalize_dates(s, e):
    start = parse_date(s) if s else None
    end = parse_date(e) if e else None
    today = datetime.combine(date.today(), datetime.min.time())
    if end and end > today:
        end = today
    if start and end and start > end:
        raise ValueError("Start date must be before end date.")
    return start, end


def normalize_ohlcv(df, ticker):
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        cols = {}
        for field in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            matches = [c for c in df.columns if c[0] == field]
            if matches:
                cols[field] = df[matches[0]]
        df = pd.DataFrame(cols, index=df.index)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Close"])

    if pd.api.types.is_datetime64_any_dtype(df.index):
        try:
            df.index = df.index.tz_localize(None)
        except:
            pass

    return df


# ---- Stooq fallback ----
def stooq_fetch(ticker, start, end):
    tickers = [ticker, f"{ticker}.US"] if not ticker.endswith(".US") else [ticker]
    for sym in tickers:
        try:
            df = pdr.DataReader(sym, "stooq")
            df = df.sort_index()
            if start or end:
                df = df.loc[
                    (df.index >= (start or df.index.min())) &
                    (df.index <= (end or df.index.max()))
                ]
            df = df.rename(columns={"Close": "Close", "Volume": "Volume"})
            return df
        except:
            continue
    raise RuntimeError("Stooq fallback failed.")


def fetch_stock_data(ticker, s, e, tries=5):
    console.print(f"\n[bold green]Fetching data for {ticker}...[/bold green]")
    ticker = ticker.upper().strip()

    start, end = normalize_dates(s, e)

    use_period = not s and not e  # User pressed enter → use lighter request

    last_err = None
    for i in range(tries):
        try:
            if use_period:
                df = yf.download(
                    ticker,
                    period="6mo",
                    interval="1d",
                    auto_adjust=True,
                    threads=False,
                    progress=False,
                    session=POLITE_SESSION
                )
            else:
                df = yf.download(
                    ticker,
                    start=start.strftime("%Y-%m-%d") if start else None,
                    end=end.strftime("%Y-%m-%d") if end else None,
                    interval="1d",
                    auto_adjust=True,
                    threads=False,
                    progress=False,
                    session=POLITE_SESSION
                )
            if df is not None and not df.empty:
                return normalize_ohlcv(df, ticker)

        except Exception as e:
            last_err = e

        time.sleep((2 * (i + 1)) * random.uniform(0.5, 1.5))

    console.print("[yellow]Yahoo is throttling → trying Stooq...[/yellow]")

    try:
        return normalize_ohlcv(stooq_fetch(ticker, start, end), ticker)
    except Exception as e:
        console.print(f"[bold red]Fetch failed: {e}[/bold red]")
        return pd.DataFrame()


# ---- Indicators ----
def indicators(df):
    df = df.copy()
    df["EMA_20"] = df["Close"].ewm(span=20).mean()
    df["EMA_50"] = df["Close"].ewm(span=50).mean()
    mean = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["Upper_Band"] = mean + std * 2
    df["Lower_Band"] = mean - std * 2
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = (100 - (100 / (1 + rs))).bfill()
    return df


# ---- Insights ----
def insights(df):
    c = df["Close"].astype(float)
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Avg Close", f"${c.mean():.2f}")
    table.add_row("High Close", f"${c.max():.2f}")
    table.add_row("Low Close", f"${c.min():.2f}")
    table.add_row("Volatility (σ)", f"{c.std():.2f}")
    r = df["RSI"].iloc[-1]
    if r > 70: state = "Overbought"
    elif r < 30: state = "Oversold"
    else: state = "Neutral"
    table.add_row("RSI", f"{r:.2f} ({state})")
    console.print(table)


# ---- NN Model ----
class Model(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.l1 = nn.Linear(n, 128)
        self.l2 = nn.Linear(128, 1)

    def forward(self, x):
        return self.l2(torch.relu(self.l1(x)))


def predict(df):
    df = df.copy()
    df["Days"] = (df.index - df.index[0]).days
    df["Pct_Change"] = df["Close"].pct_change()
    df["Momentum"] = df["Close"] - df["Close"].shift(10)
    df = df.dropna()

    X_cols = ["Days", "EMA_20", "EMA_50", "RSI", "Upper_Band", "Lower_Band", "Pct_Change", "Momentum"]
    y_col = "Close"

    scX, scY = MinMaxScaler(), MinMaxScaler()
    X = scX.fit_transform(df[X_cols])
    y = scY.fit_transform(df[[y_col]])

    # ✅ FIXED
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]

    model = Model(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for _ in range(300):
        opt.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        opt.step()

    last = df.iloc[-1]
    future_days = np.arange(last["Days"] + 1, last["Days"] + 31).reshape(-1, 1)
    future_static = np.tile(last[X_cols[1:]].values, (30, 1))
    future = np.hstack([future_days, future_static])
    future = scX.transform(future)

    future_tensor = torch.tensor(future, dtype=torch.float32)
    preds_scaled = model(future_tensor).detach().numpy()
    preds = scY.inverse_transform(preds_scaled)

    dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30)
    return pd.DataFrame({"Date": dates, "Predicted": preds.ravel()})


# ---- Dashboard ----
def dashboard(df, ticker, pred):
    app = dash.Dash(__name__)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA_20"], name="EMA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Upper_Band"], name="UpperBand"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Lower_Band"], name="LowerBand"))

    fig2 = go.Figure()
    if not pred.empty:
        fig2.add_trace(go.Scatter(x=pred["Date"], y=pred["Predicted"], name="Forecast"))

    app.layout = html.Div([
        html.H1(f"{ticker} Analysis Dashboard"),
        dcc.Graph(figure=fig),
        html.H2("Next 30 Days Prediction"),
        dcc.Graph(figure=fig2)
    ])

    app.run_server(debug=True, use_reloader=False)


# ---- Main ----
def main():
    console.print("\n📈 Stock Analysis Tool\n")

    ticker = input("Enter ticker: ").upper()
    s = input("Start date (optional): ")
    e = input("End date (optional): ")

    df = fetch_stock_data(ticker, s, e)
    if df.empty:
        console.print("[red]No data, exiting.[/red]")
        return

    df = indicators(df)
    insights(df)
    pred = predict(df)

    if input("Export CSV? (yes/no): ").lower() == "yes":
        df.to_csv(f"{ticker}_analysis.csv")
        console.print("[green]Saved.[/green]")

    console.print("\nLaunching dashboard...")
    dashboard(df, ticker, pred)


if __name__ == "__main__":
    main()
