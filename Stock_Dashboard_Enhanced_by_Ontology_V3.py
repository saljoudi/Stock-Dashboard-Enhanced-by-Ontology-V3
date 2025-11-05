# ============================================================
# PART 1: SETUP, CACHING, LOGGING, AND ONTOLOGY BASE CLASSES
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
from rdflib import Graph, Namespace, RDF, RDFS, Literal, URIRef

# ─────────────────────────────────────────────
# GLOBAL SETTINGS
# ─────────────────────────────────────────────
warnings.filterwarnings("ignore")
pd.options.display.float_format = "{:.4f}".format

# ─────────────────────────────────────────────
# PERSISTENT DISK CACHE (Performance Enhancement)
# ─────────────────────────────────────────────
CACHE_DIR = "./cache_dir"
os.makedirs(CACHE_DIR, exist_ok=True)
memory = Memory(location=CACHE_DIR, verbose=0)

# ─────────────────────────────────────────────
# STRUCTURED LOGGER
# ─────────────────────────────────────────────
def log_step(message: str):
    """
    Console logger with timestamps and standardized prefix.
    Creates an auditable trace of computational events — 
    crucial for patent documentation and reproducibility.
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

# ─────────────────────────────────────────────
# ONTOLOGY VOCABULARY (RDF Namespaces)
# ─────────────────────────────────────────────
STOCK = Namespace("http://example.org/stock#")
TECH = Namespace("http://example.org/technical#")
MARKET = Namespace("http://example.org/market#")

# ─────────────────────────────────────────────
# RDF GRAPH BUILDER
# ─────────────────────────────────────────────
class StockOntologyGraph:
    """
    Constructs and manages RDF graphs for stock analysis indicators,
    semantic relationships, and inferred market states.

    This class forms the patentable "knowledge layer" — encoding 
    technical indicators (EMA, RSI, MACD, etc.) as ontology instances 
    connected through logical and causal relationships.
    """

    def __init__(self):
        self.g = Graph()
        self._define_schema()
        log_step("Ontology graph schema initialized.")

    # ─────────────────────────────────────────────
    # SCHEMA DEFINITION
    # ─────────────────────────────────────────────
    def _define_schema(self):
        """Defines ontology classes and properties."""
        self.g.bind("stock", STOCK)
        self.g.bind("tech", TECH)
        self.g.bind("market", MARKET)

        # Core classes
        self.g.add((STOCK.StockEntity, RDF.type, RDFS.Class))
        self.g.add((STOCK.Indicator, RDF.type, RDFS.Class))
        self.g.add((MARKET.MarketState, RDF.type, RDFS.Class))
        self.g.add((MARKET.RiskLevel, RDF.type, RDFS.Class))

        # Object/Data properties
        self.g.add((STOCK.hasIndicator, RDF.type, RDF.Property))
        self.g.add((STOCK.hasValue, RDF.type, RDF.Property))
        self.g.add((STOCK.hasSignal, RDF.type, RDF.Property))
        self.g.add((STOCK.impliesState, RDF.type, RDF.Property))
        self.g.add((MARKET.hasRisk, RDF.type, RDF.Property))
        self.g.add((MARKET.hasTrend, RDF.type, RDF.Property))
        self.g.add((MARKET.hasVolatility, RDF.type, RDF.Property))
        self.g.add((MARKET.hasVolumeProfile, RDF.type, RDF.Property))

    # ─────────────────────────────────────────────
    # INDICATOR ADDITION
    # ─────────────────────────────────────────────
    def add_indicator(self, symbol: str, name: str, value: float, signal: str):
        """
        Adds an individual indicator node (e.g., RSI, EMA) to the RDF graph.

        Args:
            symbol: Stock ticker symbol.
            name: Indicator name.
            value: Computed numeric value.
            signal: Interpreted signal (e.g., 'bullish', 'bearish').
        """
        ind_uri = URIRef(f"{STOCK}{symbol}_{name}")
        self.g.add((ind_uri, RDF.type, STOCK.Indicator))
        self.g.add((ind_uri, STOCK.hasValue, Literal(round(value, 4))))
        self.g.add((ind_uri, STOCK.hasSignal, Literal(signal)))
        log_step(f"Indicator node added: {symbol}_{name} ({signal})")
        return ind_uri

    # ─────────────────────────────────────────────
    # RELATIONSHIP LINKING
    # ─────────────────────────────────────────────
    def link_state(self, indicator_uri: URIRef, state: str):
        """
        Links indicator nodes to inferred market states.
        Enables semantic chaining such as:
            EMA_50 → impliesState → BullTrend
        """
        state_uri = URIRef(f"{MARKET}{state}")
        self.g.add((indicator_uri, STOCK.impliesState, state_uri))
        log_step(f"Linked {indicator_uri} → impliesState → {state_uri}")

    # ─────────────────────────────────────────────
    # GRAPH SERIALIZATION
    # ─────────────────────────────────────────────
    def serialize(self, format: str = "turtle") -> str:
        """
        Serializes the RDF graph to Turtle format for:
        - Patent appendices
        - Data audits
        - Ontology explainability
        """
        turtle_data = self.g.serialize(format=format)
        log_step(f"Ontology graph serialized ({len(turtle_data)} bytes).")
        return turtle_data

# ─────────────────────────────────────────────
# TEST INITIALIZATION (Optional during development)
# ─────────────────────────────────────────────
# if __name__ == "__main__":
#     test_graph = StockOntologyGraph()
#     test_graph.add_indicator("AAPL", "RSI", 63.42, "bullish")
#     test_graph.link_state(URIRef(f"{STOCK}AAPL_RSI"), "bull_trend")
#     print(test_graph.serialize()[:600])
# ============================================================
# PART 2: ONTOLOGY-DRIVEN INFERENCE ENGINE (PATENT-GRADE)
# ============================================================

class MarketState(Enum):
    """Enumerated ontology concepts describing global market regime."""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    VOLATILE_BREAKOUT = "volatile_breakout"
    LOW_VOLATILITY = "low_volatility"


class TrendDirection(Enum):
    """Fine-grained directional semantics for trend interpretation."""
    STRONG_UP = "strong_up"
    MODERATE_UP = "moderate_up"
    NEUTRAL = "neutral"
    MODERATE_DOWN = "moderate_down"
    STRONG_DOWN = "strong_down"


class RiskLevel(Enum):
    """Ontology enumeration for capital-exposure risk semantics."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


# ─────────────────────────────────────────────
# STRUCTURED MARKET CONTEXT (Semantic Container)
# ─────────────────────────────────────────────
@dataclass
class MarketContext:
    """
    Container for all inferred semantic relationships.
    This object is cached and serialized for explainability.
    """
    market_state: MarketState
    trend_direction: TrendDirection
    risk_level: RiskLevel
    volatility_regime: str
    volume_profile: str
    support_levels: List[float]
    resistance_levels: List[float]
    ontology_graph: str
    reasoning_chain: List[str]


# ─────────────────────────────────────────────
# ONTOLOGY-DRIVEN ANALYSIS ENGINE
# ─────────────────────────────────────────────
class EnhancedStockAnalysisOntology:
    """
    Patent-grade reasoning engine that synthesizes numerical indicators
    into ontology relationships and produces an interpretable 'MarketContext'.

    This component transforms the numerical layer (TA indicators)
    into a knowledge layer (RDF + causal reasoning).
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.version = "5.0-ontology-driven"
        self._context_cache: Dict[str, MarketContext] = {}

    # ============================================================
    # MAIN ENTRYPOINT
    # ============================================================
    def infer_market_context(self, symbol: str, df: pd.DataFrame) -> MarketContext:
        """
        Central pipeline:
        1. Build RDF graph
        2. Extract indicator entities
        3. Infer MarketState / TrendDirection / RiskLevel
        4. Generate reasoning chain (explainable audit trace)
        """
        cache_key = f"{symbol}_{len(df)}"
        if cache_key in self._context_cache:
            return self._context_cache[cache_key]

        if len(df) < 50:
            return self._default_context()

        # Step 1: initialize ontology graph
        graph = StockOntologyGraph()

        # Step 2: extract entities for each indicator class
        trend_entities = self._extract_trend_entities(symbol, df, graph)
        momentum_entities = self._extract_momentum_entities(symbol, df, graph)
        volume_entities = self._extract_volume_entities(symbol, df, graph)
        volatility_entities = self._extract_volatility_entities(symbol, df, graph)

        # Step 3: perform semantic inference
        market_state = self._infer_market_state(trend_entities, momentum_entities, volume_entities, volatility_entities)
        trend_direction = self._infer_trend_direction(trend_entities, momentum_entities)
        risk_level = self._infer_risk_level(volatility_entities, trend_entities)
        support_levels = self._calculate_support_levels(df)
        resistance_levels = self._calculate_resistance_levels(df)

        # Step 4: link ontology relationships
        for ent_dict in [trend_entities, momentum_entities, volume_entities, volatility_entities]:
            for uri in ent_dict.get("uris", []):
                graph.link_state(uri, market_state.value)

        # Step 5: generate human-readable reasoning trace
        reasoning_chain = self._build_reasoning_chain(
            symbol, market_state, trend_direction, risk_level,
            trend_entities, momentum_entities, volume_entities,
            volatility_entities, support_levels, resistance_levels
        )

        # Step 6: package as semantic MarketContext
        context = MarketContext(
            market_state=market_state,
            trend_direction=trend_direction,
            risk_level=risk_level,
            volatility_regime=self._classify_volatility_regime(volatility_entities),
            volume_profile=self._classify_volume_profile(volume_entities),
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            ontology_graph=graph.serialize(),
            reasoning_chain=reasoning_chain
        )

        self._context_cache[cache_key] = context
        return context

    # ============================================================
    # INDICATOR ENTITY EXTRACTORS
    # ============================================================
    def _extract_trend_entities(self, symbol, df, graph):
        """Encodes trend-based signals (SMA, ADX) into ontology triples."""
        entities = {"uris": []}
        closes = df["close"]

        # Moving-average relationships
        for period in [20, 50, 200]:
            sma = closes.rolling(period).mean()
            current, avg = closes.iloc[-1], sma.iloc[-1]
            if current > avg * 1.02:
                signal = "strong_above"
            elif current > avg:
                signal = "above"
            elif current < avg * 0.98:
                signal = "strong_below"
            else:
                signal = "below"
            uri = graph.add_indicator(symbol, f"SMA_{period}", avg, signal)
            entities["uris"].append(uri)

        # ADX Strength
        adx_value = ta.trend.ADXIndicator(df["high"], df["low"], df["close"]).adx().iloc[-1]
        if adx_value > 40:
            strength = "very_strong"
        elif adx_value > 25:
            strength = "strong"
        elif adx_value > 20:
            strength = "moderate"
        else:
            strength = "weak"
        uri = graph.add_indicator(symbol, "ADX", adx_value, strength)
        entities["uris"].append(uri)
        entities["trend_strength"] = strength
        return entities

    def _extract_momentum_entities(self, symbol, df, graph):
        """Encodes momentum indicators (RSI, MACD) into ontology triples."""
        entities = {"uris": []}

        # RSI logic
        rsi_val = ta.momentum.RSIIndicator(df["close"]).rsi().iloc[-1]
        if rsi_val < 20:
            rsi_signal = "extremely_oversold"
        elif rsi_val < 30:
            rsi_signal = "oversold"
        elif rsi_val > 80:
            rsi_signal = "extremely_overbought"
        elif rsi_val > 70:
            rsi_signal = "overbought"
        else:
            rsi_signal = "neutral"
        uri = graph.add_indicator(symbol, "RSI", rsi_val, rsi_signal)
        entities["uris"].append(uri)

        # MACD logic
        exp1, exp2 = df["close"].ewm(span=12).mean(), df["close"].ewm(span=26).mean()
        macd, signal_line = exp1 - exp2, (exp1 - exp2).ewm(span=9).mean()
        macd_val, sig_val = macd.iloc[-1], signal_line.iloc[-1]
        if macd_val > sig_val and macd_val > 0:
            macd_signal = "strong_bullish"
        elif macd_val > sig_val:
            macd_signal = "bullish"
        elif macd_val < sig_val and macd_val < 0:
            macd_signal = "strong_bearish"
        else:
            macd_signal = "bearish"
        uri = graph.add_indicator(symbol, "MACD", macd_val, macd_signal)
        entities["uris"].append(uri)
        entities["macd"] = macd_signal
        return entities

    def _extract_volume_entities(self, symbol, df, graph):
        """Encodes volume-based accumulation/distribution indicators."""
        entities = {"uris": []}
        obv = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
        now, prev = obv.iloc[-1], obv.iloc[-5] if len(df) > 5 else obv.iloc[-1]
        if now > prev * 1.05:
            vol_signal = "strong_accumulation"
        elif now > prev:
            vol_signal = "accumulation"
        elif now < prev * 0.95:
            vol_signal = "strong_distribution"
        else:
            vol_signal = "distribution"
        uri = graph.add_indicator(symbol, "OBV", now, vol_signal)
        entities["uris"].append(uri)
        entities["volume_trend"] = vol_signal
        return entities

    def _extract_volatility_entities(self, symbol, df, graph):
        """Encodes volatility regime (ATR%) indicators."""
        entities = {"uris": []}
        atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
        atr_pct = (atr.iloc[-1] / df["close"].iloc[-1]) * 100
        if atr_pct > 5:
            vol_signal = "high"
        elif atr_pct < 2:
            vol_signal = "low"
        else:
            vol_signal = "medium"
        uri = graph.add_indicator(symbol, "ATR%", atr_pct, vol_signal)
        entities["uris"].append(uri)
        entities["volatility"] = vol_signal
        return entities

    # ============================================================
    # SEMANTIC INFERENCE RULES
    # ============================================================
    def _infer_market_state(self, t, m, v, vol):
        if (
            t.get("trend_strength") in ["strong", "very_strong"]
            and m.get("macd") in ["bullish", "strong_bullish"]
            and v.get("volume_trend") in ["accumulation", "strong_accumulation"]
        ):
            return MarketState.BULL_TREND
        if (
            t.get("trend_strength") in ["strong", "very_strong"]
            and m.get("macd") in ["bearish", "strong_bearish"]
            and v.get("volume_trend") in ["distribution", "strong_distribution"]
        ):
            return MarketState.BEAR_TREND
        if vol.get("volatility") == "high":
            return MarketState.VOLATILE_BREAKOUT
        return MarketState.SIDEWAYS

    def _infer_trend_direction(self, t, m):
        bullish_score = sum([
            t.get("trend_strength") in ["strong", "very_strong"],
            m.get("macd") in ["bullish", "strong_bullish"]
        ])
        bearish_score = sum([
            m.get("macd") in ["bearish", "strong_bearish"],
            t.get("trend_strength") == "weak"
        ])
        if bullish_score >= 2:
            return TrendDirection.STRONG_UP
        if bullish_score == 1:
            return TrendDirection.MODERATE_UP
        if bearish_score >= 2:
            return TrendDirection.STRONG_DOWN
        if bearish_score == 1:
            return TrendDirection.MODERATE_DOWN
        return TrendDirection.NEUTRAL

    def _infer_risk_level(self, vol, t):
        if vol.get("volatility") == "high":
            return RiskLevel.HIGH
        if vol.get("volatility") == "medium":
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    # ============================================================
    # SUPPORT / RESISTANCE DETECTION
    # ============================================================
    def _calculate_support_levels(self, df):
        lows = df["low"].tail(50)
        return [float(x) for x in [lows.min(), lows.quantile(0.25), lows.quantile(0.1)] if not np.isnan(x)]

    def _calculate_resistance_levels(self, df):
        highs = df["high"].tail(50)
        return [float(x) for x in [highs.max(), highs.quantile(0.75), highs.quantile(0.9)] if not np.isnan(x)]

    # ============================================================
    # SEMANTIC CLASSIFIERS
    # ============================================================
    def _classify_volatility_regime(self, v):
        return f"{v.get('volatility', 'medium')}_volatility"

    def _classify_volume_profile(self, v):
        return v.get("volume_trend", "neutral")

    # ============================================================
    # REASONING TRACE (Explainability Layer)
    # ============================================================
    def _build_reasoning_chain(
        self, symbol, market_state, trend_direction, risk_level,
        trend_entities, momentum_entities, volume_entities,
        volatility_entities, support_levels, resistance_levels
    ) -> List[str]:
        chain = [
            f"Symbol analyzed: {symbol}",
            f"Inferred Market State: {market_state.value}",
            f"Trend Direction: {trend_direction.value}",
            f"Risk Level: {risk_level.value}"
        ]

        # Market state justification
        if market_state == MarketState.BULL_TREND:
            chain.append("Condition met: strong trend strength + bullish MACD + accumulation volume.")
        elif market_state == MarketState.BEAR_TREND:
            chain.append("Condition met: strong trend strength + bearish MACD + distribution volume.")
        elif market_state == MarketState.VOLATILE_BREAKOUT:
            chain.append("Condition met: ATR% > 5% — high volatility breakout regime.")
        else:
            chain.append("Condition met: mixed indicators → neutral/sideways market.")

        # Risk justification
        if risk_level == RiskLevel.HIGH:
            chain.append("ATR% exceeds dynamic threshold → elevated risk.")
        elif risk_level == RiskLevel.MEDIUM:
            chain.append("Moderate volatility observed → medium risk.")
        else:
            chain.append("Stable volatility → low systemic risk.")

        # Support / Resistance summary
        if support_levels:
            chain.append(f"Support levels (last 50 bars): {', '.join(f'{s:.2f}' for s in support_levels)}")
        if resistance_levels:
            chain.append(f"Resistance levels (last 50 bars): {', '.join(f'{r:.2f}' for r in resistance_levels)}")

        return chain

    # ============================================================
    # DEFAULT CONTEXT FALLBACK
    # ============================================================
    def _default_context(self):
        return MarketContext(
            market_state=MarketState.SIDEWAYS,
            trend_direction=TrendDirection.NEUTRAL,
            risk_level=RiskLevel.MEDIUM,
            volatility_regime="unknown",
            volume_profile="unknown",
            support_levels=[],
            resistance_levels=[],
            ontology_graph="",
            reasoning_chain=["Insufficient data (need ≥ 50 bars)."]
        )

    # ============================================================
    # SUMMARY INTERFACE (for Dash / API / Patent Docs)
    # ============================================================
    def generate_summary(self, symbol: str, df: pd.DataFrame):
        """Public entry for summarizing reasoning results."""
        ctx = self.infer_market_context(symbol, df)
        return {
            "market_context": {
                "state": ctx.market_state.value,
                "trend": ctx.trend_direction.value,
                "risk": ctx.risk_level.value,
                "volatility_regime": ctx.volatility_regime,
                "volume_profile": ctx.volume_profile,
                "support_levels": ctx.support_levels,
                "resistance_levels": ctx.resistance_levels,
            },
            "ontology_graph": ctx.ontology_graph,
            "reasoning_chain": ctx.reasoning_chain
        }
# ============================================================
# PART 3: DATA FETCHING, INDICATOR COMPUTATION & DASH LAYOUT
# ============================================================

# ─────────────────────────────────────────────
# Ontology Engine Initialization
# ─────────────────────────────────────────────
ontology = EnhancedStockAnalysisOntology(debug=False)

# ─────────────────────────────────────────────
# Cached YahooQuery Data Fetcher
# ─────────────────────────────────────────────
@memory.cache
def fetch_data_cached(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Fetches OHLCV data for the specified stock symbol using YahooQuery,
    with persistent caching for computational efficiency and reproducibility.
    """
    log_step(f"Fetching data for {ticker} | Period={period} | Interval={interval}")
    tq = Ticker(ticker)
    df = tq.history(period=period, interval=interval)
    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.get_level_values("date")
    df = df.dropna(subset=["close"])
    log_step(f"Retrieved {len(df)} rows for {ticker}.")
    return df


# ─────────────────────────────────────────────
# Technical Indicator Computation (Vectorized)
# ─────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all technical indicators required for ontology reasoning.
    Optimized for minimal redundancy and vectorized across Pandas.
    """
    log_step("Computing technical indicators…")
    closes, highs, lows, vols = df["close"], df["high"], df["low"], df["volume"]

    # ── Moving Averages (Trend Layer)
    for w in [8, 20, 50, 200]:
        df[f"SMA_{w}"] = closes.rolling(w).mean()
        df[f"EMA_{w}"] = closes.ewm(span=w, adjust=False).mean()

    # ── Momentum Indicators
    df["RSI"] = ta.momentum.RSIIndicator(closes).rsi()

    macd = ta.trend.MACD(closes)
    df["MACD"], df["MACD_Signal"] = macd.macd(), macd.macd_signal()

    stoch = ta.momentum.StochasticOscillator(highs, lows, closes)
    df["%K"], df["%D"] = stoch.stoch(), stoch.stoch_signal()

    # ── Volatility Indicators
    ma20, std20 = closes.rolling(20).mean(), closes.rolling(20).std()
    df["Upper_band"], df["Lower_band"] = ma20 + 2 * std20, ma20 - 2 * std20
    df["ATR"] = ta.volatility.AverageTrueRange(highs, lows, closes).average_true_range()
    df["CCI"] = ta.trend.CCIIndicator(highs, lows, closes).cci()

    # ── Volume Indicators
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(closes, vols).on_balance_volume()
    df["VWAP"] = (closes * vols).cumsum() / vols.cumsum()
    df["ADL"] = ta.volume.AccDistIndexIndicator(highs, lows, closes, vols).acc_dist_index()
    df["MFI"] = ta.volume.MFIIndicator(highs, lows, closes, vols).money_flow_index()
    df["CMF"] = ta.volume.ChaikinMoneyFlowIndicator(highs, lows, closes, vols).chaikin_money_flow()
    df["FI"]  = ta.volume.ForceIndexIndicator(closes, vols).force_index()

    # ── Trend Strength Indicators
    adx = ta.trend.ADXIndicator(highs, lows, closes)
    df["ADX"], df["DI+"], df["DI-"] = adx.adx(), adx.adx_pos(), adx.adx_neg()

    # ── Ichimoku Cloud (Support/Resistance Layer)
    df["Tenkan_sen"]   = (highs.rolling(9).max() + lows.rolling(9).min()) / 2
    df["Kijun_sen"]    = (highs.rolling(26).max() + lows.rolling(26).min()) / 2
    df["Senkou_span_a"] = ((df["Tenkan_sen"] + df["Kijun_sen"]) / 2).shift(26)
    df["Senkou_span_b"] = ((highs.rolling(52).max() + lows.rolling(52).min()) / 2).shift(26)
    df["Chikou_span"]   = closes.shift(-26)

    log_step("Indicators computed successfully.")
    return df


# ─────────────────────────────────────────────
# Dash Application Setup
# ─────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.SOLAR],
    suppress_callback_exceptions=True
)
app.title = "Ontology-Driven Stock Dashboard"
server = app.server


# ─────────────────────────────────────────────
# Dash Layout (Fully Retaining 18 Charts)
# ─────────────────────────────────────────────
app.layout = dbc.Container([
    # Navbar
    dbc.NavbarSimple(
        brand="Ontology-Driven Stock Dashboard (Patent Prototype)",
        color="dark",
        dark=True
    ),

    # Inputs
    dbc.Row([
        dbc.Col(
            dbc.Input(id="stock-input", value="AAPL", placeholder="Enter stock symbol"),
            width=4
        )
    ], justify="center", className="my-3"),

    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id="time-range",
                options=[
                    {"label": "6 Months", "value": "6mo"},
                    {"label": "1 Year", "value": "1y"},
                    {"label": "2 Years", "value": "2y"},
                    {"label": "3 Years", "value": "3y"},
                    {"label": "4 Years", "value": "4y"},
                    {"label": "5 Years", "value": "5y"},
                    {"label": "All", "value": "max"},
                ],
                value="1y",
                clearable=False
            ),
            width=4
        )
    ], justify="center", className="my-3"),

    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id="interval",
                options=[
                    {"label": "Daily", "value": "1d"},
                    {"label": "Weekly", "value": "1wk"},
                    {"label": "Monthly", "value": "1mo"},
                ],
                value="1d",
                clearable=False
            ),
            width=4
        )
    ], justify="center", className="my-3"),

    dbc.Row([
        dbc.Col(
            dcc.RadioItems(
                id="analysis-mode",
                options=[
                    {"label": "📊 Standard", "value": "standard"},
                    {"label": "🧠 Ontology Analysis", "value": "ontology"},
                ],
                value="ontology",
                inline=True
            ),
            width=8
        )
    ], justify="center", className="my-3"),

    dbc.Row([
        dbc.Col(
            dbc.Button(
                id="analyze-button",
                n_clicks=0,
                children="🧠 Analyze with Ontology",
                color="primary"
            ),
            width="auto"
        )
    ], justify="center", className="my-3"),

    # Ontology panel
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Ontology-Based Analysis", className="bg-primary text-white"),
                dbc.CardBody([
                    html.Div(id="ontology-insights"),
                    html.Div(id="trading-signals"),
                    html.Div(id="risk-assessment"),
                    html.Div(id="trading-recommendations"),
                    html.Div(id="reasoning-trace"),
                ])
            ]),
            width=12
        )
    ], className="mb-4"),

    # 1) Candlestick (full width)
    dbc.Row([
        dbc.Col(
            dbc.Card(dbc.CardBody(dcc.Graph(id="candlestick-chart"))),
            width=12
        )
    ], className="mb-4"),

    # 2) SMA / EMA (full width)
    dbc.Row([
        dbc.Col(
            dbc.Card(dbc.CardBody(dcc.Graph(id="sma-ema-chart"))),
            width=12
        )
    ], className="mb-4"),

    # 3) Support/Resistance + RSI
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="support-resistance-chart"))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="rsi-chart"))), width=6),
    ], className="mb-4"),

    # 4) Bollinger + MACD
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="bollinger-bands-chart"))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="macd-chart"))), width=6),
    ], className="mb-4"),

    # 5) Stochastic + OBV
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="stochastic-oscillator-chart"))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="obv-chart"))), width=6),
    ], className="mb-4"),

    # 6) ATR + CCI
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="atr-chart"))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="cci-chart"))), width=6),
    ], className="mb-4"),

    # 7) MFI + CMF
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="mfi-chart"))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="cmf-chart"))), width=6),
    ], className="mb-4"),

    # 8) FI + Fibonacci
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="fi-chart"))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="fibonacci-retracement-chart"))), width=6),
    ], className="mb-4"),

    # 9) Ichimoku + VWAP
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="ichimoku-cloud-chart"))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="vwap-chart"))), width=6),
    ], className="mb-4"),

    # 10) ADL + ADX/DI
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="adl-chart"))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="adx-di-chart"))), width=6),
    ], className="mb-4"),

    # Footer
    dbc.Row([
        dbc.Col(html.Footer(
            "Ontology-Driven Dashboard © 2025 by Abu Sanad",
            className="text-center text-muted"
        ))
    ], className="mt-4"),
], fluid=True)

# ============================================================
# PART 4: CALLBACKS & APPLICATION EXECUTION (FINAL PATENT VERSION)
# ============================================================

# ─────────────────────────────────────────────
# Dynamic Button Text Toggle
# ─────────────────────────────────────────────
@app.callback(Output("analyze-button", "children"),
              Input("analysis-mode", "value"))
def update_button_text(mode):
    """Updates button label dynamically according to analysis mode."""
    return "🧠 Analyze with Ontology" if mode == "ontology" else "📊 Analyze Stock"


# ─────────────────────────────────────────────
# Master Callback: Ontology + Chart Engine
# 22 Outputs → 18 Graphs + 4 Insight Panels
# ─────────────────────────────────────────────
@app.callback(
    [Output(g, "figure") for g in [
        "candlestick-chart", "sma-ema-chart", "support-resistance-chart", "rsi-chart",
        "bollinger-bands-chart", "macd-chart", "stochastic-oscillator-chart", "obv-chart",
        "atr-chart", "cci-chart", "mfi-chart", "cmf-chart", "fi-chart",
        "fibonacci-retracement-chart", "ichimoku-cloud-chart", "vwap-chart",
        "adl-chart", "adx-di-chart"
    ]]
    + [Output(x, "children") for x in [
        "ontology-insights", "trading-signals", "risk-assessment",
        "trading-recommendations", "reasoning-trace"
    ]],
    Input("analyze-button", "n_clicks"),
    State("stock-input", "value"),
    State("time-range", "value"),
    State("interval", "value"),
    State("analysis-mode", "value"),
)
def update_graphs(n_clicks, ticker, time_range, interval, analysis_mode):
    """
    Integrates the ontology reasoning pipeline with technical visualization.
    Produces synchronized analytical outputs and explainability artifacts.
    """
    # ── Initialization (Idle State)
    if not n_clicks:
        empty_fig = go.Figure().update_layout(
            title="Click 'Analyze' to Begin", template="plotly_dark"
        )
        placeholder = html.Div("Awaiting user input…")
        return (empty_fig,) * 18 + (placeholder, html.Div(), html.Div(), html.Div(), html.Div())

    # ─────────────────────────────────────────────
    # Step 1 – Data Acquisition and Indicator Computation
    # ─────────────────────────────────────────────
    try:
        log_step(f"Starting ontology-driven analysis for {ticker}…")
        df = fetch_data_cached(ticker, time_range, interval)
        df = compute_indicators(df)
    except Exception as e:
        log_step(f"❌ Data fetch error: {e}")
        err_fig = go.Figure().update_layout(title=f"Error: {e}", template="plotly_dark")
        err_msg = html.Div(f"⚠️ Error fetching data for {ticker}: {e}")
        return (err_fig,) * 18 + (err_msg, html.Div(), html.Div(), html.Div(), html.Div())

    # ─────────────────────────────────────────────
    # Step 2 – Ontology Reasoning (If Selected)
    # ─────────────────────────────────────────────
    if analysis_mode == "ontology":
        log_step("Executing ontology reasoning engine…")
        summary = ontology.generate_summary(ticker, df)
        mc = summary["market_context"]
        reasoning_chain = summary.get("reasoning_chain", [])

        # Clean numeric presentation
        sup_str = ", ".join(f"{s:.2f}" for s in mc["support_levels"]) if mc["support_levels"] else "–"
        res_str = ", ".join(f"{r:.2f}" for r in mc["resistance_levels"]) if mc["resistance_levels"] else "–"

        # Ontology Insights Panel
        insights_content = html.Div([
            html.H4("🧠 Ontological Market Summary"),
            html.P(f"Market State: {mc['state'].replace('_', ' ').title()}"),
            html.P(f"Trend Direction: {mc['trend'].replace('_', ' ').title()}"),
            html.P(f"Risk Level: {mc['risk'].replace('_', ' ').title()}"),
            html.P(f"Volatility Regime: {mc['volatility_regime'].replace('_', ' ').title()}"),
            html.P(f"Volume Profile: {mc['volume_profile'].replace('_', ' ').title()}"),
            html.P(f"Support Levels: {sup_str}"),
            html.P(f"Resistance Levels: {res_str}")
        ])

        # Trading Signals Panel
        signals_content = html.Div([
            html.H5("📈 Inferred Trading Bias"),
            html.Ul([
                html.Li("🚀 Bullish Bias") if mc["state"] == "bull_trend"
                else html.Li("🔻 Bearish Bias") if mc["state"] == "bear_trend"
                else html.Li("⚖️ Neutral / Sideways Market")
            ])
        ])

        # Risk Assessment Panel
        risk_content = html.Div([
            html.H5("🛡️ Risk & Volatility Assessment"),
            html.P(f"Risk Level: {mc['risk'].replace('_', ' ').title()}"),
            html.P(f"Volatility: {mc['volatility_regime'].replace('_', ' ').title()}")
        ])

        # Trading Recommendations Panel
        recs = []
        if mc["state"] == "bull_trend" and mc["risk"] in ["low", "very_low"]:
            recs.append("🟢 Consider buying on pullbacks to support levels.")
        elif mc["state"] == "bear_trend" and mc["risk"] in ["high", "very_high"]:
            recs.append("🔴 Consider hedging or shorting rallies.")
        elif mc["state"] == "volatile_breakout":
            recs.append("⚡ Confirm breakout before large position entries.")
        else:
            recs.append("⚖️ Maintain neutral exposure until trend confirmation.")
        recommendations_content = html.Div([
            html.H5("💡 Trading Recommendations"),
            html.Ul([html.Li(r) for r in recs])
        ])

        # Reasoning Trace Panel
        reasoning_trace = html.Div([
            html.H5("🔍 Ontology Reasoning Trace"),
            html.Ol([html.Li(step) for step in reasoning_chain])
        ])
    else:
        # Standard Mode (Indicator-Only)
        log_step("Standard mode selected – no ontology reasoning.")
        insights_content = html.Div([
            html.H4("📊 Standard Technical Analysis"),
            html.P(f"Symbol: {ticker}, Period: {time_range}, Interval: {interval}")
        ])
        signals_content = html.Div()
        risk_content = html.Div()
        recommendations_content = html.Div()
        reasoning_trace = html.Div()

    # ─────────────────────────────────────────────
    # Step 3 – Chart Rendering (18 Charts)
    # ─────────────────────────────────────────────
    log_step("Rendering technical charts…")

    # Candlestick Chart
    fig_candle = go.Figure(go.Candlestick(
        x=df.index, open=df.open, high=df.high, low=df.low, close=df.close
    )).update_layout(title=f"{ticker} Candlestick", template="plotly_dark")

    # SMA / EMA Chart
    fig_sma = go.Figure()
    fig_sma.add_trace(go.Scatter(x=df.index, y=df.close, name="Close"))
    for col in ["SMA_20","SMA_50","SMA_200","EMA_8","EMA_20","EMA_50","EMA_200"]:
        if col in df:
            fig_sma.add_trace(go.Scatter(x=df.index, y=df[col], name=col))
    fig_sma.update_layout(title=f"{ticker} SMA & EMA", template="plotly_dark")

    # Support & Resistance Chart
    pivot = (df.high + df.low + df.close) / 3
    df["S1"], df["R1"] = 2*pivot - df.high, 2*pivot - df.low
    df["S2"], df["R2"] = pivot - (df.high - df.low), pivot + (df.high - df.low)
    fig_sr = go.Figure()
    for col in ["S1","R1","S2","R2"]:
        fig_sr.add_trace(go.Scatter(x=df.index, y=df[col], name=col))
    fig_sr.update_layout(title=f"{ticker} Support & Resistance", template="plotly_dark")

    # RSI Chart
    fig_rsi = go.Figure(go.Scatter(x=df.index, y=df.RSI, name="RSI"))
    for yv,c in [(70,"red"),(30,"green")]:
        fig_rsi.add_shape(type="line",x0=df.index[0],x1=df.index[-1],
                          y0=yv,y1=yv,line=dict(color=c,dash="dash"))
    fig_rsi.update_layout(title=f"{ticker} RSI", template="plotly_dark")

    # Bollinger Bands
    fig_bb = go.Figure()
    for col in ["close","Upper_band","Lower_band"]:
        fig_bb.add_trace(go.Scatter(x=df.index, y=df[col], name=col))
    fig_bb.update_layout(title=f"{ticker} Bollinger Bands", template="plotly_dark")

    # MACD Chart
    fig_macd = go.Figure()
    for col in ["MACD","MACD_Signal"]:
        fig_macd.add_trace(go.Scatter(x=df.index, y=df[col], name=col))
    fig_macd.update_layout(title=f"{ticker} MACD", template="plotly_dark")

    # Stochastic Oscillator
    fig_sto = go.Figure()
    for col in ["%K","%D"]:
        fig_sto.add_trace(go.Scatter(x=df.index, y=df[col], name=col))
    fig_sto.update_layout(title=f"{ticker} Stochastic Oscillator", template="plotly_dark")

    # OBV Chart
    fig_obv = go.Figure(go.Scatter(x=df.index, y=df.OBV, name="OBV"))
    fig_obv.update_layout(title=f"{ticker} On-Balance Volume", template="plotly_dark")

    # ATR Chart
    fig_atr = go.Figure(go.Scatter(x=df.index, y=df.ATR, name="ATR"))
    fig_atr.update_layout(title=f"{ticker} Average True Range", template="plotly_dark")

    # CCI Chart
    fig_cci = go.Figure(go.Scatter(x=df.index, y=df.CCI, name="CCI"))
    for yv in [100, -100]:
        fig_cci.add_shape(type="line",x0=df.index[0],x1=df.index[-1],
                          y0=yv,y1=yv,line=dict(color="gray",dash="dash"))
    fig_cci.update_layout(title=f"{ticker} CCI", template="plotly_dark")

    # MFI Chart
    fig_mfi = go.Figure(go.Scatter(x=df.index, y=df.MFI, name="MFI"))
    for yv,c in [(80,"red"),(20,"green")]:
        fig_mfi.add_shape(type="line",x0=df.index[0],x1=df.index[-1],
                          y0=yv,y1=yv,line=dict(color=c,dash="dash"))
    fig_mfi.update_layout(title=f"{ticker} MFI", template="plotly_dark")

    # CMF Chart
    fig_cmf = go.Figure(go.Scatter(x=df.index, y=df.CMF, name="CMF"))
    fig_cmf.add_shape(type="line",x0=df.index[0],x1=df.index[-1],
                      y0=0,y1=0,line=dict(color="red",dash="dash"))
    fig_cmf.update_layout(title=f"{ticker} Chaikin Money Flow", template="plotly_dark")

    # Force Index
    fig_fi = go.Figure(go.Scatter(x=df.index, y=df.FI, name="Force Index"))
    fig_fi.update_layout(title=f"{ticker} Force Index", template="plotly_dark")

    # Fibonacci Retracement
    high, low = df.high.max(), df.low.min()
    diff = high - low
    fib_levels = {p: high - (v * diff) for p,v in {
        "0%":0,"23.6%":0.236,"38.2%":0.382,"50%":0.5,"61.8%":0.618,"100%":1}.items()}
    fig_fib = go.Figure(go.Scatter(x=df.index, y=df.close, name="Close"))
    for label, price in fib_levels.items():
        fig_fib.add_trace(go.Scatter(
            x=[df.index[0], df.index[-1]], y=[price, price],
            name=label, line=dict(dash="dash")
        ))
    fig_fib.update_layout(title=f"{ticker} Fibonacci Retracement", template="plotly_dark")

    # Ichimoku Cloud
    fig_ich = go.Figure()
    for col in ["close","Tenkan_sen","Kijun_sen","Senkou_span_a","Senkou_span_b","Chikou_span"]:
        fig_ich.add_trace(go.Scatter(x=df.index, y=df[col], name=col))
    fig_ich.update_layout(title=f"{ticker} Ichimoku Cloud", template="plotly_dark")

    # VWAP Chart
    fig_vwap = go.Figure()
    fig_vwap.add_trace(go.Scatter(x=df.index, y=df.close, name="Close"))
    fig_vwap.add_trace(go.Scatter(x=df.index, y=df.VWAP, name="VWAP"))
    fig_vwap.update_layout(title=f"{ticker} VWAP", template="plotly_dark")

    # ADL Chart
    fig_adl = go.Figure(go.Scatter(x=df.index, y=df.ADL, name="ADL"))
    fig_adl.update_layout(title=f"{ticker} Accumulation/Distribution Line", template="plotly_dark")

    # ADX / DI Chart
    fig_adx = go.Figure()
    for col in ["ADX","DI+","DI-"]:
        fig_adx.add_trace(go.Scatter(x=df.index, y=df[col], name=col))
    fig_adx.update_layout(title=f"{ticker} ADX & DI", template="plotly_dark")

    log_step("✅ Charts rendered successfully.")

    # Return all graphs + ontology panels
    return (
        fig_candle, fig_sma, fig_sr, fig_rsi, fig_bb, fig_macd, fig_sto, fig_obv,
        fig_atr, fig_cci, fig_mfi, fig_cmf, fig_fi, fig_fib, fig_ich, fig_vwap,
        fig_adl, fig_adx,
        insights_content, signals_content, risk_content,
        recommendations_content, reasoning_trace
    )


# ─────────────────────────────────────────────
# Application Entrypoint
# ─────────────────────────────────────────────
if __name__ == "__main__":
    log_step("🚀 Launching Ontology-Driven Stock Dashboard (Final Patent Version)…")
    app.run_server(debug=False)
