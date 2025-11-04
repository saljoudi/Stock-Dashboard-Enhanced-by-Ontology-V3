import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from yahooquery import Ticker
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import pandas as pd, numpy as np, ta, warnings
from functools import lru_cache
from datetime import datetime
warnings.filterwarnings("ignore")

class EnhancedStockAnalysisOntology:
    """
    Fully optimized ontology-based analysis engine.
    Encapsulates trend, momentum, volume, volatility, and candle reasoning.
    """

    def __init__(self, debug=False):
        self.debug = debug
        self.version = "3.0-optimized"

        # Indicator registry for reference/export
        self.indicators = {
            "trend": ["SMA", "EMA", "MACD", "ADX", "Ichimoku"],
            "momentum": ["RSI", "Stochastic", "CCI", "Williams_R"],
            "volume": ["OBV", "VWAP", "CMF", "MFI", "ADL"],
            "volatility": ["Bollinger_Bands", "ATR", "Keltner_Channel"],
            "support_resistance": ["Pivot_Points", "Fibonacci", "Volume_Profile"]
        }

        # Simple explanation map
        self.explanations = {
            "Golden Cross": "SMA(50) crosses above SMA(200) – bullish trend continuation.",
            "Death Cross": "SMA(50) crosses below SMA(200) – bearish trend continuation.",
            "RSI Oversold": "RSI < 30 indicates possible rebound potential.",
            "RSI Overbought": "RSI > 70 indicates potential price exhaustion.",
            "Doji": "Indecision candle suggesting reversal or pause.",
            "Hammer": "Bullish reversal candle after downtrend.",
        }

    # ------------------------------
    # Helper: Safe get
    # ------------------------------
    def _safe(self, df, col, default=np.nan):
        return df[col].iloc[-1] if col in df.columns and not df[col].empty else default

    # ------------------------------
    # Candle Pattern Detection
    # ------------------------------
    def detect_candle_patterns(self, df, lookback=10):
        if df is None or len(df) < 5:
            return [], 0
        df = df.tail(lookback).copy()
        df["body"] = abs(df["close"] - df["open"])
        df["range"] = df["high"] - df["low"]
        df["upper"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower"] = df[["open", "close"]].min(axis=1) - df["low"]

        patterns, conf = [], 0
        for _, r in df.iterrows():
            if r["range"] == 0:
                continue
            body, upper, lower = r["body"]/r["range"], r["upper"]/r["range"], r["lower"]/r["range"]

            if body < 0.1:
                patterns.append({"pattern": "Doji", "type": "neutral", "confidence": 0.6})
                conf += 0.3
            if lower > 0.5 and upper < 0.3:
                if r["close"] > r["open"]:
                    patterns.append({"pattern": "Hammer", "type": "bullish", "confidence": 0.7})
                    conf += 0.5
                else:
                    patterns.append({"pattern": "Hanging Man", "type": "bearish", "confidence": 0.7})
                    conf -= 0.5
            if upper > 0.5 and lower < 0.3:
                patterns.append({"pattern": "Shooting Star", "type": "bearish", "confidence": 0.7})
                conf -= 0.5
            if body > 0.9:
                if r["close"] > r["open"]:
                    patterns.append({"pattern": "Bullish Marubozu", "type": "bullish", "confidence": 0.8})
                    conf += 0.6
                else:
                    patterns.append({"pattern": "Bearish Marubozu", "type": "bearish", "confidence": 0.8})
                    conf -= 0.6

        patterns = sorted(patterns, key=lambda x: x["confidence"], reverse=True)[:3]
        if self.debug:
            print(f"[DEBUG] Candle patterns detected: {patterns}")
        return patterns, conf

    # ------------------------------
    # Trend Analysis
    # ------------------------------
    def enhanced_trend_analysis(self, df):
        if len(df) < 50:
            return ["Insufficient data"], 0
        s, sc = [], 0
        cur = self._safe(df, "close")
        for p in [20, 50, 200]:
            col = f"SMA_{p}"
            if col in df.columns:
                if cur > df[col].iloc[-1]:
                    s.append(f"🟢 Above SMA {p}")
                    sc += 1
                else:
                    s.append(f"🔴 Below SMA {p}")
                    sc -= 1
        adx = self._safe(df, "ADX")
        if not np.isnan(adx):
            if adx > 25:
                s.append("📈 Strong Trend (ADX>25)")
                sc += 2
            else:
                s.append("⚪ Weak Trend (ADX<25)")
        sc = max(min(sc + 5, 15), 0)
        if self.debug:
            print(f"[DEBUG] Trend score={sc}")
        return s, sc

    # ------------------------------
    # Momentum Analysis
    # ------------------------------
    def enhanced_momentum_analysis(self, df):
        s, sc = [], 0
        rsi = self._safe(df, "RSI")
        if not np.isnan(rsi):
            if rsi < 30:
                s.append(f"🎯 RSI Oversold ({rsi:.1f})")
                sc += 1
            elif rsi > 70:
                s.append(f"🎯 RSI Overbought ({rsi:.1f})")
                sc -= 1
            else:
                s.append(f"⚖️ RSI Neutral ({rsi:.1f})")
        macd, sig = self._safe(df, "MACD"), self._safe(df, "MACD_Signal")
        if not np.isnan(macd) and not np.isnan(sig):
            if macd > sig:
                s.append("📊 MACD Bullish")
                sc += 1
            else:
                s.append("📊 MACD Bearish")
                sc -= 1
        sc = max(min(sc + 4, 8), 0)
        if self.debug:
            print(f"[DEBUG] Momentum score={sc}")
        return s, sc

    # ------------------------------
    # Volume Analysis
    # ------------------------------
    def enhanced_volume_analysis(self, df):
        if "OBV" not in df.columns:
            return ["Insufficient data"], 0
        rising = df["OBV"].iloc[-1] > df["OBV"].iloc[-5] if len(df) > 5 else False
        s, sc = [], 0
        if rising:
            s.append("💰 OBV Rising – Volume confirms trend")
            sc += 2
        else:
            s.append("📉 OBV Falling – Weak volume confirmation")
            sc -= 2
        sc = max(min(sc + 3, 6), 0)
        if self.debug:
            print(f"[DEBUG] Volume score={sc}")
        return s, sc

    # ------------------------------
    # Volatility Analysis
    # ------------------------------
    def volatility_analysis(self, df):
        atr, close = self._safe(df, "ATR"), self._safe(df, "close")
        if np.isnan(atr) or np.isnan(close):
            return ["Insufficient data"], 0
        ratio = atr / close * 100
        if ratio > 5:
            return [f"🌪️ High Volatility ({ratio:.1f}%)"], 3
        elif ratio < 2:
            return [f"🍃 Low Volatility ({ratio:.1f}%)"], 1
        return [f"⚖️ Moderate Volatility ({ratio:.1f}%)"], 2

    # ------------------------------
    # Support / Resistance
    # ------------------------------
    def support_resistance_analysis(self, df):
        if len(df) < 20:
            return ["Insufficient data"], 0
        price, high, low = df["close"].iloc[-1], df["high"].tail(20).max(), df["low"].tail(20).min()
        dist_high, dist_low = (high - price)/high*100, (price - low)/price*100
        s, sc = [], 0
        if dist_high < 2:
            s.append("🏔️ Near Resistance")
            sc -= 1
        elif dist_low < 2:
            s.append("🛟 Near Support")
            sc += 1
        if self.debug:
            print(f"[DEBUG] Support/Resistance score={sc}")
        return s, sc

    # ------------------------------
    # Bias / Confidence Reasoning
    # ------------------------------
    def determine_overall_bias(self, t, m, v, vol, sr, c):
        total = t*0.3 + m*0.2 + v*0.15 + vol*0.1 + sr*0.1 + c*0.15
        maxp = 15*0.3 + 8*0.2 + 6*0.15 + 3*0.1 + 2*0.1 + 5*0.15
        conf = max(0, min(100, (total/maxp)*100))
        if total > maxp*0.7: bias = "🟢 STRONG BULLISH"
        elif total > maxp*0.5: bias = "🟡 MODERATE BULLISH"
        elif total > maxp*0.3: bias = "⚪ NEUTRAL"
        elif total > maxp*0.1: bias = "🟠 MODERATE BEARISH"
        else: bias = "🔴 STRONG BEARISH"
        if self.debug:
            print(f"[DEBUG] Bias={bias}, confidence={conf:.1f}%")
        return bias, conf

    # ------------------------------
    # Master Summary
    # ------------------------------
    def generate_advanced_summary(self, df):
        t_s, t = self.enhanced_trend_analysis(df)
        m_s, m = self.enhanced_momentum_analysis(df)
        v_s, v = self.enhanced_volume_analysis(df)
        vol_s, vol = self.volatility_analysis(df)
        sr_s, sr = self.support_resistance_analysis(df)
        c_s, c = self.detect_candle_patterns(df)
        c_s = self._normalize_candles_for_summary(c_s)
        bias, conf = self.determine_overall_bias(t, m, v, vol, sr, c)
        risks = self.risk_assessment(df, {"overall_bias": bias, "confidence_score": conf})
        if self.debug:
            print("[DEBUG] Summary complete with risk assessment.")

        return {
            "trend": t_s,
            "momentum": m_s,
            "volume": v_s,
            "volatility": vol_s,
            "support_resistance": sr_s,
            "candle_patterns": c_s,
            "overall_bias": bias,
            "confidence_score": conf,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "detailed_scores": {
                "trend": t, "momentum": m, "volume": v,
                "volatility": vol, "support_resistance": sr, "candle_patterns": c
            },
            "risk_assessment": risks,
            "reasoning_trace": [
                f"Trend={t}, Momentum={m}, Volume={v}, Volatility={vol}, SR={sr}, Candle={c}",
                f"→ {bias} ({conf:.1f}% confidence)"
            ],
        }
    # ------------------------------
    # Trading Recommendations
    # ------------------------------
    def get_trading_recommendations(self, summary):
        """
        Produce human-readable trading suggestions based on ontology bias,
        confidence, and major signals.  Output: list[str]
        """
        recs = []
        bias = summary.get("overall_bias", "")
        conf = summary.get("confidence_score", 0)
        rsi_line = next((x for x in summary["momentum"] if "RSI" in x), "")
        atr_line = next((x for x in summary["volatility"] if "Volatility" in x), "")

        # --- Bias-driven suggestions ---
        if "BULLISH" in bias:
            recs.append("✅ Consider long positions or scaling into uptrend opportunities.")
            if "STRONG" in bias:
                recs.append("📈 Strong confirmation — trend momentum aligned across indicators.")
        elif "BEARISH" in bias:
            recs.append("⚠️ Bias bearish — protect profits or tighten stops on longs.")
            recs.append("📉 Consider short setups or wait for reversal confirmation.")
        else:
            recs.append("⚪ Neutral conditions — stand aside or trade range boundaries.")

        # --- RSI based ---
        if "Oversold" in rsi_line:
            recs.append("🎯 RSI oversold — watch for bullish reversal signals.")
        elif "Overbought" in rsi_line:
            recs.append("💢 RSI overbought — risk of pullback or profit taking.")

        # --- Volatility based ---
        if "High" in atr_line:
            recs.append("🌪️ High volatility — use wider stops and reduce position size.")
        elif "Low" in atr_line:
            recs.append("🍃 Low volatility — expect range-bound price action.")

        # --- Candle patterns ---
        for p in summary.get("candle_patterns", []):
            name = p.get("pattern", "")
            if p.get("type") == "bullish":
                recs.append(f"🟢 { name } supports bullish setup.")
            elif p.get("type") == "bearish":
                recs.append(f"🔴 { name } warns of bearish pressure.")

        recs.append(f"ℹ️ Confidence level: {conf:.1f}% (based on multi-factor ontology scoring).")
        return recs


    # ------------------------------
    # Risk Assessment
    # ------------------------------
    def risk_assessment(self, df, summary):
        """
        Evaluate immediate risk environment: volatility, volume, and trend strength.
        Returns list[str]
        """
        risks = []
        atr = self._safe(df, "ATR")
        close = self._safe(df, "close")
        adx = self._safe(df, "ADX")

        if not np.isnan(atr) and not np.isnan(close):
            atr_pct = atr / close * 100
            if atr_pct > 6:
                risks.append(f"⚠️ Extreme volatility ({atr_pct:.1f} %) — avoid over-leverage.")
            elif atr_pct > 3:
                risks.append(f"⚠️ Elevated volatility ({atr_pct:.1f} %) — moderate risk exposure.")
            else:
                risks.append(f"✅ Stable volatility ({atr_pct:.1f} %) — risk conditions normal.")

        if not np.isnan(adx):
            if adx > 35:
                risks.append("⚡ Very strong trend — trades against trend risky.")
            elif adx < 20:
                risks.append("💤 Weak trend — range trading preferred.")
            else:
                risks.append("📊 Moderate trend strength — balanced risk environment.")

        if not risks:
            risks.append("ℹ️ Risk data insufficient to evaluate.")
        return risks


    # ------------------------------
    # Fix: normalize candle patterns structure
    # ------------------------------
    def _normalize_candles_for_summary(self, patterns):
        """Guarantee consistent list[dict] structure for summary use."""
        normalized = []
        for p in patterns:
            if isinstance(p, dict):
                normalized.append(p)
            elif isinstance(p, str):
                normalized.append({"pattern": p, "type": "unknown", "confidence": 0.5})
        return normalized

#!/usr/bin/env python
# coding: utf-8

# import dash
# from dash import dcc, html
# from dash.dependencies import Input, Output, State
# from yahooquery import Ticker
# import plotly.graph_objs as go
# import pandas as pd
# import numpy as np
# import ta
# import dash_bootstrap_components as dbc
# import warnings
# from functools import lru_cache
# from datetime import datetime
# warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Setup Dash App
# ─────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SOLAR], suppress_callback_exceptions=True)
server = app.server

# Instantiate our ontology engine (enable debug by passing debug=True if desired)
ontology = EnhancedStockAnalysisOntology(debug=False)

# ─────────────────────────────────────────────
# Cache data fetch function
# ─────────────────────────────────────────────
@lru_cache(maxsize=32)
def fetch_data_cached(ticker: str, period: str, interval: str):
    if ticker.isdigit():
        ticker_mod = f"{ticker}.SR"
    else:
        ticker_mod = ticker
    tq = Ticker(ticker_mod)
    df = tq.history(period=period, interval=interval)
    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.get_level_values('date')
    df = df.dropna(subset=['close'])  # ensure close exists
    return df

# ─────────────────────────────────────────────
# Layout (unchanged IDs, structure)
# ─────────────────────────────────────────────
app.layout = dbc.Container([
    dbc.NavbarSimple(brand="Advanced Stock Dashboard with Enhanced Ontology", color="dark brown", dark=True),

    # Symbol input
    dbc.Row([dbc.Col(dbc.Input(id='stock-input', placeholder='Enter stock symbol', value='AAPL'), width=4)], justify='center', className="my-3"),

    # Time-range selector
    dbc.Row([dbc.Col(dcc.Dropdown(
        id='time-range',
        options=[
            {'label':'6 months','value':'6mo'},
            {'label':'1 year','value':'1y'},
            {'label':'2 years','value':'2y'},
            {'label':'3 years','value':'3y'},
            {'label':'4 years','value':'4y'},
            {'label':'5 years','value':'5y'},
            {'label':'All','value':'max'}
        ],
        value='1y',
        clearable=False
    ), width=4)], justify='center', className="my-3"),

    # Interval selector
    dbc.Row([dbc.Col(dcc.Dropdown(
        id='interval',
        options=[
            {'label':'Daily','value':'1d'},
            {'label':'Weekly','value':'1wk'},
            {'label':'Monthly','value':'1mo'}
        ],
        value='1d',
        clearable=False
    ), width=4)], justify='center', className="my-3"),

    # Analysis mode toggle
    dbc.Row([dbc.Col(dcc.RadioItems(
        id='analysis-mode',
        options=[
            {'label':'📊 Standard Analysis (Charts Only)','value':'standard'},
            {'label':'🧠 Enhanced Ontology Analysis','value':'ontology'}
        ],
        value='ontology',
        inline=True
    ), width=8)], justify='center', className="my-3"),

    # Analyze button
    dbc.Row([dbc.Col(dbc.Button(id='analyze-button', n_clicks=0, color="primary"), width="auto")], justify='center', className="my-3"),

    # Ontology insights
    dbc.Row([dbc.Col(dbc.Card([dbc.CardHeader("Advanced Ontology-Based Analysis", className="bg-primary text-white"),
                               dbc.CardBody([html.Div(id="ontology-insights"),
                                             html.Div(id="trading-signals"),
                                             html.Div(id="risk-assessment"),
                                             html.Div(id="trading-recommendations")])
                              ]), width=12)], className="mb-4"),

    # Chart rows (all 18 charts)
    dbc.Row([dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='candlestick-chart'))), width=12)], className="mb-4"),
    dbc.Row([dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='sma-ema-chart'))), width=12)], className="mb-4"),

    dbc.Row([dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='support-resistance-chart'))), width=6),
             dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='rsi-chart'))), width=6)], className="mb-4"),

    dbc.Row([dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='bollinger-bands-chart'))), width=6),
             dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='macd-chart'))), width=6)], className="mb-4"),

    dbc.Row([dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='stochastic-oscillator-chart'))), width=6),
             dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='obv-chart'))), width=6)], className="mb-4"),

    dbc.Row([dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='atr-chart'))), width=6),
             dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='cci-chart'))), width=6)], className="mb-4"),

    dbc.Row([dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='mfi-chart'))), width=6),
             dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='cmf-chart'))), width=6)], className="mb-4"),

    dbc.Row([dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='fi-chart'))), width=6),
             dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='fibonacci-retracement-chart'))), width=6)], className="mb-4"),

    dbc.Row([dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='ichimoku-cloud-chart'))), width=6),
             dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='vwap-chart'))), width=6)], className="mb-4"),

    dbc.Row([dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='adl-chart'))), width=6),
             dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='adx-di-chart'))), width=6)], className="mb-4"),

    # Footer
    dbc.Row([dbc.Col(html.Footer("Advanced Stock Dashboard with Enhanced Ontology ©2025 by Abu Sanad", className="text-center text-muted"))], className="mt-4")
], fluid=True)

# ─────────────────────────────────────────────
# Callback for the scores toggle and button label
# ─────────────────────────────────────────────
@app.callback(
    Output("scores-collapse", "is_open"),
    [Input("scores-toggle", "n_clicks")],
    [State("scores-collapse", "is_open")]
)
def toggle_scores(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

@app.callback(
    Output('analyze-button', 'children'),
    Input('analysis-mode', 'value')
)
def update_button_text(mode):
    if mode == 'ontology':
        return "🧠 Analyze with Enhanced Ontology"
    return "📊 Analyze Stock (Standard)"

# ─────────────────────────────────────────────
# Main analysis callback (returns 22 outputs)
# ─────────────────────────────────────────────
@app.callback(
    [Output('candlestick-chart', 'figure'),
     Output('sma-ema-chart', 'figure'),
     Output('support-resistance-chart', 'figure'),
     Output('rsi-chart', 'figure'),
     Output('bollinger-bands-chart', 'figure'),
     Output('macd-chart', 'figure'),
     Output('stochastic-oscillator-chart', 'figure'),
     Output('obv-chart', 'figure'),
     Output('atr-chart', 'figure'),
     Output('cci-chart', 'figure'),
     Output('mfi-chart', 'figure'),
     Output('cmf-chart', 'figure'),
     Output('fi-chart', 'figure'),
     Output('fibonacci-retracement-chart', 'figure'),
     Output('ichimoku-cloud-chart', 'figure'),
     Output('vwap-chart', 'figure'),
     Output('adl-chart', 'figure'),
     Output('adx-di-chart', 'figure'),
     Output('ontology-insights', 'children'),
     Output('trading-signals', 'children'),
     Output('risk-assessment', 'children'),
     Output('trading-recommendations', 'children')],
    Input('analyze-button', 'n_clicks'),
    State('stock-input', 'value'),
    State('time-range', 'value'),
    State('interval', 'value'),
    State('analysis-mode', 'value')
)
def update_graphs(n_clicks, ticker, time_range, interval, analysis_mode):
    # Initial blank state
    if not n_clicks:
        empty_fig = go.Figure().update_layout(title="Click 'Analyze' to display analysis", template='plotly_dark')
        empty_ins = html.Div("Click analyze to see insights")
        return (empty_fig,)*18 + (empty_ins, html.Div(), html.Div(), html.Div())

    # Fetch data
    try:
        df = fetch_data_cached(ticker, time_range, interval)
        if df.empty:
            raise ValueError("No data returned for symbol")
    except Exception as e:
        fig_err = go.Figure().update_layout(title=f"Error fetching data: {e}", template='plotly_dark')
        ins_err = html.Div(f"⚠️ Error fetching data for {ticker}: {e}")
        return (fig_err,)*18 + (ins_err, html.Div(), html.Div(), html.Div())

    # Indicator calculations safely
    try:
        # MAs
        df['SMA_20'] = df['close'].rolling(20).mean()
        df['SMA_50'] = df['close'].rolling(50).mean()
        df['SMA_200'] = df['close'].rolling(200).mean()
        df['EMA_8'] = df['close'].ewm(span=8, adjust=False).mean()
        df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['EMA_200'] = df['close'].ewm(span=200, adjust=False).mean()

        # Pivot Points
        pivot = (df['high'] + df['low'] + df['close']) / 3
        df['Pivot_Point'] = pivot
        df['Support_1'] = 2*pivot - df['high']
        df['Resistance_1'] = 2*pivot - df['low']
        df['Support_2'] = pivot - (df['high'] - df['low'])
        df['Resistance_2'] = pivot + (df['high'] - df['low'])

        # RSI
        df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

        # Bollinger Bands
        ma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        df['Upper_band'] = ma20 + 2*std20
        df['Lower_band'] = ma20 - 2*std20

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['%K'] = stoch.stoch()
        df['%D'] = stoch.stoch_signal()

        # Volume & VWAP
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df['VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

        # Volatility & other indicators
        df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        df['CCI'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
        df['ADL'] = ta.volume.AccDistIndexIndicator(df['high'], df['low'], df['close'], df['volume']).acc_dist_index()
        df['MFI'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
        df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()
        df['FI'] = ta.volume.ForceIndexIndicator(df['close'], df['volume']).force_index()
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['ADX'] = adx.adx()
        df['DI+'] = adx.adx_pos()
        df['DI-'] = adx.adx_neg()

        # Fibonacci retracement levels
        high_p = df['high'].max()
        low_p = df['low'].min()
        diff = high_p - low_p
        fib = {
            '0.0%': high_p,
            '23.6%': high_p - 0.236*diff,
            '38.2%': high_p - 0.382*diff,
            '50.0%': high_p - 0.5*diff,
            '61.8%': high_p - 0.618*diff,
            '100.0%': low_p
        }

        # Ichimoku Calculations
        df['Tenkan_sen'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min())/2
        df['Kijun_sen']  = (df['high'].rolling(26).max() + df['low'].rolling(26).min())/2
        df['Senkou_span_a'] = ((df['Tenkan_sen'] + df['Kijun_sen'])/2).shift(26)
        df['Senkou_span_b'] = ((df['high'].rolling(52).max() + df['low'].rolling(52).min())/2).shift(26)
        df['Chikou_span'] = df['close'].shift(-26)

    except Exception as e:
        err_fig = go.Figure().update_layout(title=f"Error computing indicators: {e}", template='plotly_dark')
        err_ins = html.Div(f"⚠️ Error computing indicators: {e}")
        return (err_fig,)*18 + (err_ins, html.Div(), html.Div(), html.Div())

    # Ontology analysis
    if analysis_mode == 'ontology':
        summary = ontology.generate_advanced_summary(df)
        recommendations = ontology.get_trading_recommendations(summary)

        # Build insights content
        insights_content = html.Div([
            html.H4(f"{summary['overall_bias']} (Confidence: {summary['confidence_score']:.1f}%)"),
            html.Div(f"Market Regime: {', '.join(summary.get('market_regime', []))}"),
            html.Hr(),
            html.H5("📍 Candle Patterns Detected:"),
            html.Ul([html.Li(f"{p['pattern']} ({p['type']}) – conf {p['confidence']:.2f}") for p in summary.get('candle_patterns', [])]),
            html.H5("📈 Trend Signals:"),
            html.Ul([html.Li(s) for s in summary.get('trend', [])]),
            html.H5("⚡ Momentum Signals:"),
            html.Ul([html.Li(s) for s in summary.get('momentum', [])]),
        ])
        signals_content = html.Div([html.H5("📌 Key Signals"), html.P("See above sections for details.")])
        risk_content = html.Div([html.H5("🛡️ Risk Assessment"), html.Ul([html.Li(r) for r in summary.get('risk_assessment', [])])])
        recommendations_content = html.Div([html.H5("💡 Trading Recommendations"), html.Ul([html.Li(r) for r in recommendations])])
    else:
        insights_content = html.Div([
            html.H4("📊 Standard Technical Analysis Mode"),
            html.P(f"Symbol: {ticker}"),
            html.P(f"Period: {time_range}, Interval: {interval}"),
            html.P(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        ])
        signals_content = html.Div()
        risk_content = html.Div()
        recommendations_content = html.Div()

    # Generate all figures
    try:
        # Candlestick + volume
        fig_candle = go.Figure(go.Candlestick(x=df.index,
                                              open=df['open'], high=df['high'],
                                              low=df['low'], close=df['close'],
                                              name='Candlestick'))
        fig_candle.update_layout(title=f"{ticker} Candlestick", template='plotly_dark')

        # SMA/EMA chart
        fig_sma_ema = go.Figure()
        fig_sma_ema.add_trace(go.Scatter(x=df.index, y=df['close'], name='Close'))
        for col in ['SMA_20','SMA_50','SMA_200','EMA_8','EMA_20','EMA_50','EMA_200']:
            if col in df.columns:
                fig_sma_ema.add_trace(go.Scatter(x=df.index, y=df[col], name=col))
        fig_sma_ema.update_layout(title=f"{ticker} SMA & EMA", template='plotly_dark')

        # Support & Resistance chart
        fig_sr = go.Figure()
        for col, dashstyle in [('Pivot_Point','dash'), ('Support_1','dot'),
                               ('Resistance_1','dot'), ('Support_2','dot'),
                               ('Resistance_2','dot')]:
            if col in df.columns:
                fig_sr.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(dash=dashstyle)))
        fig_sr.update_layout(title=f"{ticker} Support & Resistance", template='plotly_dark')

        # RSI
        fig_rsi = go.Figure(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
        fig_rsi.add_shape(type='line', x0=df.index[0], x1=df.index[-1], y0=70, y1=70, line=dict(color='Red', dash='dash'))
        fig_rsi.add_shape(type='line', x0=df.index[0], x1=df.index[-1], y0=30, y1=30, line=dict(color='Green', dash='dash'))
        fig_rsi.update_layout(title=f"{ticker} RSI", template='plotly_dark')

        # Bollinger Bands
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=df.index, y=df['close'], name='Close'))
        fig_bb.add_trace(go.Scatter(x=df.index, y=df['Upper_band'], name='Upper Band'))
        fig_bb.add_trace(go.Scatter(x=df.index, y=df['Lower_band'], name='Lower Band'))
        fig_bb.update_layout(title=f"{ticker} Bollinger Bands", template='plotly_dark')

        # MACD
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'))
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal'))
        fig_macd.update_layout(title=f"{ticker} MACD", template='plotly_dark')

        # Stochastic
        fig_sto = go.Figure()
        fig_sto.add_trace(go.Scatter(x=df.index, y=df['%K'], name='%K'))
        fig_sto.add_trace(go.Scatter(x=df.index, y=df['%D'], name='%D'))
        fig_sto.update_layout(title=f"{ticker} Stochastic Oscillator", template='plotly_dark')

        # OBV
        fig_obv = go.Figure(go.Scatter(x=df.index, y=df['OBV'], name='OBV'))
        fig_obv.update_layout(title=f"{ticker} OBV", template='plotly_dark')

        # ATR
        fig_atr = go.Figure(go.Scatter(x=df.index, y=df['ATR'], name='ATR'))
        fig_atr.update_layout(title=f"{ticker} ATR", template='plotly_dark')

        # CCI
        fig_cci = go.Figure(go.Scatter(x=df.index, y=df['CCI'], name='CCI'))
        fig_cci.add_shape(type='line', x0=df.index[0], x1=df.index[-1], y0=100, y1=100, line=dict(color='Red', dash='dash'))
        fig_cci.add_shape(type='line', x0=df.index[0], x1=df.index[-1], y0=-100, y1=-100, line=dict(color='Green', dash='dash'))
        fig_cci.update_layout(title=f"{ticker} CCI", template='plotly_dark')

        # MFI
        fig_mfi = go.Figure(go.Scatter(x=df.index, y=df['MFI'], name='MFI'))
        fig_mfi.add_shape(type='line', x0=df.index[0], x1=df.index[-1], y0=80, y1=80, line=dict(color='Red', dash='dash'))
        fig_mfi.add_shape(type='line', x0=df.index[0], x1=df.index[-1], y0=20, y1=20, line=dict(color='Green', dash='dash'))
        fig_mfi.update_layout(title=f"{ticker} MFI", template='plotly_dark')

        # CMF
        fig_cmf = go.Figure(go.Scatter(x=df.index, y=df['CMF'], name='CMF'))
        fig_cmf.add_shape(type='line', x0=df.index[0], x1=df.index[-1], y0=0, y1=0, line=dict(color='Red', dash='dash'))
        fig_cmf.update_layout(title=f"{ticker} CMF", template='plotly_dark')

        # FI
        fig_fi = go.Figure(go.Scatter(x=df.index, y=df['FI'], name='Force Index'))
        fig_fi.add_shape(type='line', x0=df.index[0], x1=df.index[-1], y0=0, y1=0, line=dict(color='Red', dash='dash'))
        fig_fi.update_layout(title=f"{ticker} Force Index", template='plotly_dark')

        # Fibonacci Retracement
        fig_fib = go.Figure(go.Scatter(x=df.index, y=df['close'], name='Close'))
        for label, price in fib.items():
            fig_fib.add_trace(go.Scatter(x=[df.index[0], df.index[-1]], y=[price, price], name=f'Fib {label}', line=dict(dash='dash')))
        fig_fib.update_layout(title=f"{ticker} Fibonacci Retracement", template='plotly_dark')

        # Ichimoku Cloud
        fig_ich = go.Figure()
        fig_ich.add_trace(go.Scatter(x=df.index, y=df['close'], name='Close'))
        for col in ['Tenkan_sen', 'Kijun_sen', 'Senkou_span_a', 'Senkou_span_b', 'Chikou_span']:
            if col in df.columns:
                fig_ich.add_trace(go.Scatter(x=df.index, y=df[col], name=col))
        fig_ich.update_layout(title=f"{ticker} Ichimoku Cloud", template='plotly_dark')

        # VWAP
        fig_vwap = go.Figure()
        fig_vwap.add_trace(go.Scatter(x=df.index, y=df['close'], name='Close'))
        fig_vwap.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name='VWAP'))
        fig_vwap.update_layout(title=f"{ticker} VWAP", template='plotly_dark')

        # ADL
        fig_adl = go.Figure(go.Scatter(x=df.index, y=df['ADL'], name='ADL'))
        fig_adl.update_layout(title=f"{ticker} ADL", template='plotly_dark')

        # ADX & DI
        fig_adx = go.Figure()
        fig_adx.add_trace(go.Scatter(x=df.index, y=df['ADX'], name='ADX'))
        fig_adx.add_trace(go.Scatter(x=df.index, y=df['DI-'], name='DI-'))
        fig_adx.add_trace(go.Scatter(x=df.index, y=df['DI+'], name='DI+'))
        fig_adx.update_layout(title=f"{ticker} ADX & DI", template='plotly_dark')

    except Exception as e:
        errf = go.Figure().update_layout(title=f"Error in chart generation: {e}", template='plotly_dark')
        return (errf,)*18 + (insights_content, signals_content, risk_content, recommendations_content)

    # Return all 22 outputs
    return (fig_candle, fig_sma_ema, fig_sr, fig_rsi, fig_bb, fig_macd,
            fig_sto, fig_obv, fig_atr, fig_cci, fig_mfi, fig_cmf,
            fig_fi, fig_fib, fig_ich, fig_vwap, fig_adl, fig_adx,
            insights_content, signals_content, risk_content, recommendations_content)

# ─────────────────────────────────────────────
# Run app
# ─────────────────────────────────────────────
if __name__ == '__main__':
    app.run_server(debug=False)
