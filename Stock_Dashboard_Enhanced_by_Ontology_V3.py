# ============================================================
# PART 1: IMPORTS & SETUP
# ============================================================

import os
import dash
import ta
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from dash import dcc, html
from dash.dependencies import Input, Output, State
from yahooquery import Ticker
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Any
from joblib import Memory

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Persistent Disk Cache Setup
# ─────────────────────────────────────────────
CACHE_DIR = "./cache_dir"
os.makedirs(CACHE_DIR, exist_ok=True)
memory = Memory(location=CACHE_DIR, verbose=0)

# ─────────────────────────────────────────────
# Console Logger
# ─────────────────────────────────────────────
def log_step(message: str):
    """Lightweight console logger with timestamps."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

# ============================================================
# PART 2: ENHANCED ONTOLOGY ENGINE (FULL, OPTIMIZED)
# ============================================================

class MarketState(Enum):
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    VOLATILE_BREAKOUT = "volatile_breakout"
    LOW_VOLATILITY = "low_volatility"

class TrendDirection(Enum):
    STRONG_UP = "strong_up"
    MODERATE_UP = "moderate_up"
    NEUTRAL = "neutral"
    MODERATE_DOWN = "moderate_down"
    STRONG_DOWN = "strong_down"

class RiskLevel(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class MarketContext:
    market_state: MarketState
    trend_direction: TrendDirection
    risk_level: RiskLevel
    volatility_regime: str
    volume_profile: str
    support_levels: List[float]
    resistance_levels: List[float]

class EnhancedStockAnalysisOntology:
    """
    Optimized version of the full ontology reasoning engine.
    Uses semantic inference with caching and efficient vectorized indicator reuse.
    """

    def __init__(self, debug=False):
        self.debug = debug
        self.version = "4.2-optimized"
        self._context_cache: Dict[str, MarketContext] = {}

    # ─────────────────────────────────────────────
    # MAIN INFERENCE ENTRYPOINT
    # ─────────────────────────────────────────────
    def infer_market_context(self, df: pd.DataFrame) -> MarketContext:
        """Main ontological reasoning method with caching."""
        cache_key = f"{len(df)}_{round(df['close'].iloc[-1], 2)}"
        if cache_key in self._context_cache:
            return self._context_cache[cache_key]

        if len(df) < 50:
            ctx = self._default_context()
            self._context_cache[cache_key] = ctx
            return ctx

        trend_entities = self._extract_trend_entities(df)
        momentum_entities = self._extract_momentum_entities(df)
        volume_entities = self._extract_volume_entities(df)
        volatility_entities = self._extract_volatility_entities(df)

        market_state = self._infer_market_state(trend_entities, momentum_entities, volume_entities, volatility_entities)
        trend_direction = self._infer_trend_direction(trend_entities, momentum_entities)
        risk_level = self._infer_risk_level(volatility_entities, trend_entities)
        support_levels = self._calculate_support_levels(df)
        resistance_levels = self._calculate_resistance_levels(df)

        ctx = MarketContext(
            market_state,
            trend_direction,
            risk_level,
            self._classify_volatility_regime(volatility_entities),
            self._classify_volume_profile(volume_entities),
            support_levels,
            resistance_levels
        )
        self._context_cache[cache_key] = ctx
        return ctx

    # ─────────────────────────────────────────────
    # ENTITY EXTRACTION (OPTIMIZED)
    # ─────────────────────────────────────────────
    def _extract_trend_entities(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract trend entities with optimized reuse."""
        entities = {}
        closes = df["close"]
        for period in [20, 50, 200]:
            sma = closes.rolling(period).mean()
            current = closes.iloc[-1]
            avg = sma.iloc[-1]
            if current > avg * 1.02:
                entities[f"sma_{period}"] = "strong_above"
            elif current > avg:
                entities[f"sma_{period}"] = "above"
            elif current < avg * 0.98:
                entities[f"sma_{period}"] = "strong_below"
            else:
                entities[f"sma_{period}"] = "below"

        adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"]).adx().iloc[-1]
        entities["trend_strength"] = (
            "very_strong" if adx > 40 else
            "strong" if adx > 25 else
            "moderate" if adx > 20 else
            "weak"
        )
        return entities

    def _extract_momentum_entities(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract momentum entities with semantic labels."""
        entities = {}
        rsi_val = ta.momentum.RSIIndicator(df["close"]).rsi().iloc[-1]
        if rsi_val < 20:
            entities["rsi"] = "extremely_oversold"
        elif rsi_val < 30:
            entities["rsi"] = "oversold"
        elif rsi_val > 80:
            entities["rsi"] = "extremely_overbought"
        elif rsi_val > 70:
            entities["rsi"] = "overbought"
        else:
            entities["rsi"] = "neutral"

        exp1, exp2 = df["close"].ewm(span=12).mean(), df["close"].ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        m, s = macd.iloc[-1], signal.iloc[-1]
        if m > s and m > 0:
            entities["macd"] = "strong_bullish"
        elif m > s:
            entities["macd"] = "bullish"
        elif m < s and m < 0:
            entities["macd"] = "strong_bearish"
        else:
            entities["macd"] = "bearish"
        return entities

    def _extract_volume_entities(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract volume entities with OBV semantics."""
        entities = {}
        obv = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
        now, prev = obv.iloc[-1], obv.iloc[-5] if len(df) > 5 else obv.iloc[-1]
        if now > prev * 1.05:
            entities["volume_trend"] = "strong_accumulation"
        elif now > prev:
            entities["volume_trend"] = "accumulation"
        elif now < prev * 0.95:
            entities["volume_trend"] = "strong_distribution"
        else:
            entities["volume_trend"] = "distribution"
        return entities

    def _extract_volatility_entities(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract volatility entities with regime semantics."""
        entities = {}
        atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
        atr_pct = (atr.iloc[-1] / df["close"].iloc[-1]) * 100
        if atr_pct > 5:
            entities["volatility"] = "high"
        elif atr_pct < 2:
            entities["volatility"] = "low"
        else:
            entities["volatility"] = "medium"
        return entities

    # ─────────────────────────────────────────────
    # INFERENCE LOGIC
    # ─────────────────────────────────────────────
    def _infer_market_state(self, t, m, v, vol):
        if (t.get("sma_20") in ["above", "strong_above"]
            and t.get("sma_50") in ["above", "strong_above"]
            and m.get("macd") in ["bullish", "strong_bullish"]
            and t.get("trend_strength") in ["strong", "very_strong"]):
            return MarketState.BULL_TREND

        if (t.get("sma_20") in ["below", "strong_below"]
            and t.get("sma_50") in ["below", "strong_below"]
            and m.get("macd") in ["bearish", "strong_bearish"]
            and t.get("trend_strength") in ["strong", "very_strong"]):
            return MarketState.BEAR_TREND

        if vol.get("volatility") == "high":
            return MarketState.VOLATILE_BREAKOUT
        return MarketState.SIDEWAYS

    def _infer_trend_direction(self, t, m):
        b = sum([
            t.get("sma_20") in ["above", "strong_above"],
            t.get("sma_50") in ["above", "strong_above"],
            m.get("macd") in ["bullish", "strong_bullish"]
        ])
        s = sum([
            t.get("sma_20") in ["below", "strong_below"],
            t.get("sma_50") in ["below", "strong_below"],
            m.get("macd") in ["bearish", "strong_bearish"]
        ])
        if b >= 3: return TrendDirection.STRONG_UP
        if b == 2: return TrendDirection.MODERATE_UP
        if s >= 3: return TrendDirection.STRONG_DOWN
        if s == 2: return TrendDirection.MODERATE_DOWN
        return TrendDirection.NEUTRAL

    def _infer_risk_level(self, vol, t):
        if vol.get("volatility") == "high":
            return RiskLevel.HIGH
        if vol.get("volatility") == "medium":
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    # ─────────────────────────────────────────────
    # SUPPORT / RESISTANCE AND CLASSIFICATION
    # ─────────────────────────────────────────────
    def _calculate_support_levels(self, df):
        lows = df["low"].tail(50)
        return [float(x) for x in [lows.min(), lows.quantile(0.25), lows.quantile(0.1)] if not np.isnan(x)]

    def _calculate_resistance_levels(self, df):
        highs = df["high"].tail(50)
        return [float(x) for x in [highs.max(), highs.quantile(0.75), highs.quantile(0.9)] if not np.isnan(x)]

    def _classify_volatility_regime(self, v): return f"{v.get('volatility', 'medium')}_volatility"
    def _classify_volume_profile(self, v): return v.get("volume_trend", "neutral")

    def _default_context(self):
        return MarketContext(MarketState.SIDEWAYS, TrendDirection.NEUTRAL, RiskLevel.MEDIUM, "unknown", "unknown", [], [])

    # ─────────────────────────────────────────────
    # PUBLIC SUMMARY OUTPUT
    # ─────────────────────────────────────────────
    def generate_summary(self, df):
        ctx = self.infer_market_context(df)
        return {
            "market_context": {
                "state": ctx.market_state.value,
                "trend": ctx.trend_direction.value,
                "risk": ctx.risk_level.value,
                "volatility_regime": ctx.volatility_regime,
                "volume_profile": ctx.volume_profile,
                "support_levels": ctx.support_levels,
                "resistance_levels": ctx.resistance_levels
            }
        }
# ============================================================
# PART 3: DATA FETCHING & DASH LAYOUT
# ============================================================

# ─────────────────────────────────────────────
# Cached Data Fetch Function
# ─────────────────────────────────────────────
@memory.cache
def fetch_data_cached(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Fetch data using yahooquery with persistent caching.
    Automatically handles multi-index and empty results.
    """
    log_step(f"Fetching data for {ticker} | Period: {period} | Interval: {interval}")
    tq = Ticker(ticker)
    df = tq.history(period=period, interval=interval)
    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.get_level_values('date')
    df = df.dropna(subset=['close'])
    log_step(f"Data retrieved successfully: {len(df)} rows.")
    return df

# ─────────────────────────────────────────────
# Technical Indicator Precomputation (Optimized)
# ─────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Efficiently compute all required technical indicators.
    This avoids recomputing overlapping rolling/EMA windows.
    """
    log_step("Computing indicators...")
    closes = df['close']
    highs, lows, vols = df['high'], df['low'], df['volume']

    # Moving Averages
    for w in [8, 20, 50, 200]:
        df[f'SMA_{w}'] = closes.rolling(w).mean()
        df[f'EMA_{w}'] = closes.ewm(span=w, adjust=False).mean()

    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(closes, window=14).rsi()

    # Bollinger Bands
    ma20, std20 = closes.rolling(20).mean(), closes.rolling(20).std()
    df['Upper_band'] = ma20 + 2 * std20
    df['Lower_band'] = ma20 - 2 * std20

    # MACD
    exp1, exp2 = closes.ewm(span=12).mean(), closes.ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(highs, lows, closes)
    df['%K'], df['%D'] = stoch.stoch(), stoch.stoch_signal()

    # Volume indicators
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(closes, vols).on_balance_volume()
    df['VWAP'] = (closes * vols).cumsum() / vols.cumsum()

    # Volatility
    df['ATR'] = ta.volatility.AverageTrueRange(highs, lows, closes).average_true_range()
    df['CCI'] = ta.trend.CCIIndicator(highs, lows, closes).cci()
    df['ADL'] = ta.volume.AccDistIndexIndicator(highs, lows, closes, vols).acc_dist_index()
    df['MFI'] = ta.volume.MFIIndicator(highs, lows, closes, vols).money_flow_index()
    df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(highs, lows, closes, vols).chaikin_money_flow()
    df['FI'] = ta.volume.ForceIndexIndicator(closes, vols).force_index()

    # ADX & DI
    adx = ta.trend.ADXIndicator(highs, lows, closes)
    df['ADX'], df['DI+'], df['DI-'] = adx.adx(), adx.adx_pos(), adx.adx_neg()

    # Ichimoku Cloud
    df['Tenkan_sen'] = (highs.rolling(9).max() + lows.rolling(9).min()) / 2
    df['Kijun_sen'] = (highs.rolling(26).max() + lows.rolling(26).min()) / 2
    df['Senkou_span_a'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)
    df['Senkou_span_b'] = ((highs.rolling(52).max() + lows.rolling(52).min()) / 2).shift(26)
    df['Chikou_span'] = closes.shift(-26)

    log_step("Indicators computed successfully.")
    return df

# ─────────────────────────────────────────────
# Dash App Initialization
# ─────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.SOLAR],
    suppress_callback_exceptions=True
)
server = app.server
ontology = EnhancedStockAnalysisOntology(debug=False)

# ─────────────────────────────────────────────
# Dash Layout (IDENTICAL to Original)
# ─────────────────────────────────────────────
app.layout = dbc.Container([
    dbc.NavbarSimple(
        brand="Advanced Stock Dashboard with Enhanced Ontology",
        color="dark brown", dark=True
    ),

    # Stock Input
    dbc.Row([
        dbc.Col(dbc.Input(
            id='stock-input',
            placeholder='Enter stock symbol',
            value='AAPL'
        ), width=4)
    ], justify='center', className="my-3"),

    # Time Range
    dbc.Row([
        dbc.Col(dcc.Dropdown(
            id='time-range',
            options=[
                {'label': '6 months', 'value': '6mo'},
                {'label': '1 year', 'value': '1y'},
                {'label': '2 years', 'value': '2y'},
                {'label': '3 years', 'value': '3y'},
                {'label': '4 years', 'value': '4y'},
                {'label': '5 years', 'value': '5y'},
                {'label': 'All', 'value': 'max'}
            ],
            value='1y',
            clearable=False
        ), width=4)
    ], justify='center', className="my-3"),

    # Interval
    dbc.Row([
        dbc.Col(dcc.Dropdown(
            id='interval',
            options=[
                {'label': 'Daily', 'value': '1d'},
                {'label': 'Weekly', 'value': '1wk'},
                {'label': 'Monthly', 'value': '1mo'}
            ],
            value='1d',
            clearable=False
        ), width=4)
    ], justify='center', className="my-3"),

    # Analysis Mode
    dbc.Row([
        dbc.Col(dcc.RadioItems(
            id='analysis-mode',
            options=[
                {'label': '📊 Standard Analysis (Charts Only)', 'value': 'standard'},
                {'label': '🧠 Enhanced Ontology Analysis', 'value': 'ontology'}
            ],
            value='ontology',
            inline=True
        ), width=8)
    ], justify='center', className="my-3"),

    # Analyze Button
    dbc.Row([
        dbc.Col(dbc.Button(
            id='analyze-button',
            n_clicks=0,
            children="🧠 Analyze with Enhanced Ontology",
            color="primary"
        ), width="auto")
    ], justify='center', className="my-3"),

    # Ontology Insights Section
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Advanced Ontology-Based Analysis", className="bg-primary text-white"),
            dbc.CardBody([
                html.Div(id="ontology-insights"),
                html.Div(id="trading-signals"),
                html.Div(id="risk-assessment"),
                html.Div(id="trading-recommendations")
            ])
        ]), width=12)
    ], className="mb-4"),

    # Chart Rows (ALL 18 PRESERVED)
    dbc.Row([dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='candlestick-chart'))), width=12)], className="mb-4"),
    dbc.Row([dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='sma-ema-chart'))), width=12)], className="mb-4"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='support-resistance-chart'))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='rsi-chart'))), width=6)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='bollinger-bands-chart'))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='macd-chart'))), width=6)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='stochastic-oscillator-chart'))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='obv-chart'))), width=6)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='atr-chart'))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='cci-chart'))), width=6)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='mfi-chart'))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='cmf-chart'))), width=6)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='fi-chart'))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='fibonacci-retracement-chart'))), width=6)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='ichimoku-cloud-chart'))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='vwap-chart'))), width=6)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='adl-chart'))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='adx-di-chart'))), width=6)
    ], className="mb-4"),

    # Footer
    dbc.Row([
        dbc.Col(html.Footer(
            "Advanced Stock Dashboard with Enhanced Ontology ©2025 by Abu Sanad",
            className="text-center text-muted"
        ))
    ], className="mt-4")
], fluid=True)
# ============================================================
# PART 4: CALLBACKS & APP EXECUTION
# ============================================================

# ─────────────────────────────────────────────
# Update analyze button text dynamically
# ─────────────────────────────────────────────
@app.callback(
    Output('analyze-button', 'children'),
    Input('analysis-mode', 'value')
)
def update_button_text(mode):
    return "🧠 Analyze with Enhanced Ontology" if mode == 'ontology' else "📊 Analyze Stock (Standard)"

# ─────────────────────────────────────────────
# Main Callback (22 outputs, identical to original)
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
    # Initial State
    if not n_clicks:
        empty_fig = go.Figure().update_layout(title="Click 'Analyze' to display analysis", template='plotly_dark')
        empty_ins = html.Div("Click analyze to see insights")
        return (empty_fig,) * 18 + (empty_ins, html.Div(), html.Div(), html.Div())

    try:
        log_step("Starting analysis pipeline...")
        df = fetch_data_cached(ticker, time_range, interval)
        if df.empty:
            raise ValueError("No data returned for symbol")

        # Compute Indicators (optimized)
        df = compute_indicators(df)

    except Exception as e:
        log_step(f"Error fetching or computing data: {e}")
        fig_err = go.Figure().update_layout(title=f"Error fetching data: {e}", template='plotly_dark')
        ins_err = html.Div(f"⚠️ Error fetching data for {ticker}: {e}")
        return (fig_err,) * 18 + (ins_err, html.Div(), html.Div(), html.Div())

    # ─────────────────────────────────────────────
    # Ontology Analysis (only if mode == ontology)
    # ─────────────────────────────────────────────
    if analysis_mode == "ontology":
        log_step("Running ontology reasoning...")
        summary = ontology.generate_summary(df)
        log_step("Ontology reasoning complete.")

        insights_content = html.Div([
            html.H4("🧠 Ontological Market Analysis"),
            html.H5(f"Market State: {summary['market_context']['state'].upper()}"),
            html.H5(f"Trend Direction: {summary['market_context']['trend'].upper()}"),
            html.H5(f"Risk Level: {summary['market_context']['risk'].upper()}"),
            html.Hr(),
            html.H5("📊 Market Context:"),
            html.Ul([
                html.Li(f"Volatility Regime: {summary['market_context']['volatility_regime']}"),
                html.Li(f"Volume Profile: {summary['market_context']['volume_profile']}"),
                html.Li(f"Support Levels: {[f'{lvl:.2f}' for lvl in summary['market_context']['support_levels']]}"),
                html.Li(f"Resistance Levels: {[f'{lvl:.2f}' for lvl in summary['market_context']['resistance_levels']]}"),
            ])
        ])

        # Basic heuristic signals from context
        signals_content = html.Div([
            html.H5("📌 Trading Signals"),
            html.Ul([
                html.Li("🚀 Bullish Bias" if summary['market_context']['state'] == 'bull_trend' else "🔻 Bearish Bias" 
                        if summary['market_context']['state'] == 'bear_trend' else "⚖️ Neutral")
            ])
        ])

        risk_content = html.Div([
            html.H5("🛡️ Risk Assessment"),
            html.P(f"Current Risk Level: {summary['market_context']['risk']}"),
            html.P(f"Market Conditions: {summary['market_context']['state']}"),
            html.P(f"Volatility: {summary['market_context']['volatility_regime']}")
        ])

        recs = []
        risk = summary['market_context']['risk']
        state = summary['market_context']['state']
        if state == 'bull_trend' and risk in ['low', 'very_low']:
            recs.append("🟢 Buy on pullbacks to support levels.")
        elif state == 'bear_trend' and risk in ['high', 'very_high']:
            recs.append("🔴 Sell rallies or hedge positions.")
        elif state == 'volatile_breakout':
            recs.append("⚡ Trade smaller position sizes; confirm breakout.")
        elif risk in ['high', 'very_high']:
            recs.append("⚠️ High-risk market — reduce exposure.")
        else:
            recs.append("⚖️ Market neutral — wait for confirmation.")

        recommendations_content = html.Div([
            html.H5("💡 Trading Recommendations"),
            html.Ul([html.Li(r) for r in recs])
        ])

    else:
        log_step("Standard analysis mode (no ontology).")
        insights_content = html.Div([
            html.H4("📊 Standard Technical Analysis Mode"),
            html.P(f"Symbol: {ticker}"),
            html.P(f"Period: {time_range}, Interval: {interval}"),
            html.P(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        ])
        signals_content = html.Div()
        risk_content = html.Div()
        recommendations_content = html.Div()

    # ─────────────────────────────────────────────
    # Chart Generation (UNCHANGED)
    # ─────────────────────────────────────────────
    log_step("Generating charts...")
    try:
        fig_candle = go.Figure(go.Candlestick(
            x=df.index, open=df['open'], high=df['high'],
            low=df['low'], close=df['close'], name='Candlestick'
        )).update_layout(title=f"{ticker} Candlestick", template='plotly_dark')

        fig_sma_ema = go.Figure()
        fig_sma_ema.add_trace(go.Scatter(x=df.index, y=df['close'], name='Close'))
        for col in ['SMA_20', 'SMA_50', 'SMA_200', 'EMA_8', 'EMA_20', 'EMA_50', 'EMA_200']:
            fig_sma_ema.add_trace(go.Scatter(x=df.index, y=df[col], name=col))
        fig_sma_ema.update_layout(title=f"{ticker} SMA & EMA", template='plotly_dark')

        # Support & Resistance
        pivot = (df['high'] + df['low'] + df['close']) / 3
        df['Support_1'] = 2 * pivot - df['high']
        df['Resistance_1'] = 2 * pivot - df['low']
        df['Support_2'] = pivot - (df['high'] - df['low'])
        df['Resistance_2'] = pivot + (df['high'] - df['low'])
        fig_sr = go.Figure()
        for col in ['Support_1', 'Resistance_1', 'Support_2', 'Resistance_2']:
            fig_sr.add_trace(go.Scatter(x=df.index, y=df[col], name=col))
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
        high_p, low_p = df['high'].max(), df['low'].min()
        diff = high_p - low_p
        fib_levels = {
            '0.0%': high_p,
            '23.6%': high_p - 0.236 * diff,
            '38.2%': high_p - 0.382 * diff,
            '50.0%': high_p - 0.5 * diff,
            '61.8%': high_p - 0.618 * diff,
            '100.0%': low_p
        }
        fig_fib = go.Figure(go.Scatter(x=df.index, y=df['close'], name='Close'))
        for label, price in fib_levels.items():
            fig_fib.add_trace(go.Scatter(x=[df.index[0], df.index[-1]], y=[price, price], name=f'Fib {label}', line=dict(dash='dash')))
        fig_fib.update_layout(title=f"{ticker} Fibonacci Retracement", template='plotly_dark')

        # Ichimoku Cloud
        fig_ich = go.Figure()
        fig_ich.add_trace(go.Scatter(x=df.index, y=df['close'], name='Close'))
        for col in ['Tenkan_sen', 'Kijun_sen', 'Senkou_span_a', 'Senkou_span_b', 'Chikou_span']:
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

        log_step("Charts generated successfully.")

    except Exception as e:
        log_step(f"Error generating charts: {e}")
        err_fig = go.Figure().update_layout(title=f"Error in chart generation: {e}", template='plotly_dark')
        return (err_fig,) * 18 + (insights_content, signals_content, risk_content, recommendations_content)

    log_step("Analysis pipeline complete.")
    return (
        fig_candle, fig_sma_ema, fig_sr, fig_rsi, fig_bb, fig_macd,
        fig_sto, fig_obv, fig_atr, fig_cci, fig_mfi, fig_cmf,
        fig_fi, fig_fib, fig_ich, fig_vwap, fig_adl, fig_adx,
        insights_content, signals_content, risk_content, recommendations_content
    )

# ─────────────────────────────────────────────
# Run the Application
# ─────────────────────────────────────────────
if __name__ == '__main__':
    log_step("Launching Enhanced Stock Dashboard with Optimized Ontology...")
    app.run_server(debug=False, port=8055)
