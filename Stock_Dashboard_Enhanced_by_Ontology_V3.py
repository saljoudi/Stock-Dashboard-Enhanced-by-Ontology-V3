# ============================================================
# PART 1: ENHANCED ONTOLOGY FOUNDATION
# ============================================================
#!/usr/bin/env python
# coding: utf-8

# ============================================================
# PART 1: ENTERPRISE CONFIGURATION & ONTOLOGY FOUNDATION
# ============================================================

# ─────────────────────────────────────────────
# STANDARD LIBRARY IMPORTS
# ─────────────────────────────────────────────
import os
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# ─────────────────────────────────────────────
# THIRD-PARTY IMPORTS
# ─────────────────────────────────────────────
# Data & Computation
import pandas as pd
import numpy as np
from joblib import Memory

# Technical Analysis
import ta
from yahooquery import Ticker

# Semantic Web & Ontology
from rdflib import Graph, Namespace, RDF, RDFS, OWL, Literal, URIRef, XSD
from rdflib.namespace import DefinedNamespace

# Web Dashboard
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Dash, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots



# ─────────────────────────────────────────────
# GLOBAL SETTINGS
# ─────────────────────────────────────────────
warnings.filterwarnings("ignore")
CACHE_DIR = "./cache_dir"
os.makedirs(CACHE_DIR, exist_ok=True)
memory = Memory(location=CACHE_DIR, verbose=0)

def log_step(message: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

# ─────────────────────────────────────────────
# ENHANCED ONTOLOGY VOCABULARY
# ─────────────────────────────────────────────
STOCK = Namespace("http://example.org/stock#")
TECH = Namespace("http://example.org/technical#")
MARKET = Namespace("http://example.org/market#")
TIME = Namespace("http://example.org/time#")
EVIDENCE = Namespace("http://example.org/evidence#")

# ─────────────────────────────────────────────
# OWL Ontology Schema (Patent-Grade)
# ─────────────────────────────────────────────
class StockOntologyGraph:
    """
    Production-grade OWL ontology for financial technical analysis.
    Features:
    - Temporal indexing of all statements
    - Indicator interdependencies
    - Confidence-weighted evidence
    - Contradiction detection
    - Multi-hop inference paths
    """
    
    def __init__(self):
        self.g = Graph()
        self._define_owl_schema()
        log_step("OWL ontology schema initialized with temporal semantics.")
        
    def _define_owl_schema(self):
        """Defines comprehensive OWL schema with inference rules."""
        self.g.bind("stock", STOCK)
        self.g.bind("tech", TECH)
        self.g.bind("market", MARKET)
        self.g.bind("time", TIME)
        self.g.bind("evidence", EVIDENCE)
        
        # Core Classes
        for cls in [
            STOCK.StockEntity, STOCK.Indicator, STOCK.Signal,
            MARKET.MarketState, MARKET.RiskLevel, MARKET.TrendRegime,
            TIME.Instant, TIME.Interval, EVIDENCE.EvidenceBundle
        ]:
            self.g.add((cls, RDF.type, RDFS.Class))
            self.g.add((cls, RDF.type, OWL.Class))
        
        # Indicator Subclasses
        indicator_types = {
            TECH.TrendIndicator: ["SMA", "EMA", "ADX", "Ichimoku"],
            TECH.MomentumIndicator: ["RSI", "MACD", "Stochastic", "CCI"],
            TECH.VolatilityIndicator: ["ATR", "BollingerBands"],
            TECH.VolumeIndicator: ["OBV", "VWAP", "ADL", "MFI", "CMF", "ForceIndex"]
        }
        
        for parent, children in indicator_types.items():
            self.g.add((parent, RDF.type, RDFS.Class))
            self.g.add((parent, RDFS.subClassOf, STOCK.Indicator))
            for child in children:
                child_uri = TECH[child]
                self.g.add((child_uri, RDF.type, RDFS.Class))
                self.g.add((child_uri, RDFS.subClassOf, parent))
        
        # Properties
        props = {
            # Temporal properties
            STOCK.atTime: (STOCK.Indicator, TIME.Instant),
            STOCK.observedAt: (STOCK.Signal, TIME.Instant),
            
            # Value properties
            STOCK.hasNumericValue: (STOCK.Indicator, XSD.float),
            STOCK.hasSignal: (STOCK.Indicator, STOCK.Signal),
            STOCK.hasThreshold: (STOCK.Indicator, XSD.float),
            
            # Causal properties
            STOCK.impliesState: (STOCK.Indicator, MARKET.MarketState),
            STOCK.confirms: (STOCK.Indicator, STOCK.Indicator),
            STOCK.contradicts: (STOCK.Indicator, STOCK.Indicator),
            STOCK.contributesTo: (STOCK.Indicator, MARKET.TrendRegime),
            
            # Evidence properties
            EVIDENCE.hasConfidence: (EVIDENCE.EvidenceBundle, XSD.float),
            EVIDENCE.hasWeight: (EVIDENCE.EvidenceBundle, XSD.float),
            EVIDENCE.supports: (EVIDENCE.EvidenceBundle, STOCK.Indicator),
        }
        
        for prop, (domain, range_val) in props.items():
            self.g.add((prop, RDF.type, RDF.Property))
            self.g.add((prop, RDFS.domain, domain))
            self.g.add((prop, RDFS.range, range_val))
            
        # Transitive property for inference chains
        self.g.add((STOCK.confirms, RDF.type, OWL.TransitiveProperty))
        self.g.add((STOCK.chainedSignal, RDF.type, OWL.TransitiveProperty))
        
        log_step("OWL schema with 40+ classes/properties defined.")
    
    def add_indicator(self, symbol: str, indicator_type: str, value: float, 
                     signal: str, confidence: float = 1.0, metadata: Dict = None) -> URIRef:
        """
        Adds temporally-indexed indicator with confidence weighting.
        
        Args:
            symbol: Stock ticker
            indicator_type: Indicator class (e.g., 'RSI', 'Ichimoku')
            value: Numeric value
            signal: Categorical signal
            confidence: 0.0-1.0 reliability score
            metadata: Additional temporal/parameter context
        """
        ts = metadata.get("timestamp") if metadata else datetime.now().isoformat()
        ind_uri = URIRef(f"{STOCK}{symbol}_{indicator_type}_{hash(ts)}")
        
        # Type assertion
        type_map = {
            "RSI": TECH.RSI, "MACD": TECH.MACD, "Stochastic": TECH.Stochastic,
            "CCI": TECH.CCI, "ATR": TECH.ATR, "BollingerBands": TECH.BollingerBands,
            "OBV": TECH.OBV, "VWAP": TECH.VWAP, "Ichimoku": TECH.Ichimoku,
            "SMA": TECH.SMA, "EMA": TECH.EMA, "ADX": TECH.ADX,
            "MFI": TECH.MFI, "CMF": TECH.CMF, "ForceIndex": TECH.ForceIndex
        }
        
        self.g.add((ind_uri, RDF.type, type_map.get(indicator_type, STOCK.Indicator)))
        self.g.add((ind_uri, STOCK.hasNumericValue, Literal(round(float(value), 4))))
        self.g.add((ind_uri, STOCK.hasSignal, Literal(signal)))
        self.g.add((ind_uri, STOCK.atTime, Literal(ts, datatype=XSD.dateTime)))
        
        # Evidence bundle
        if confidence < 1.0:
            ev_uri = URIRef(f"{EVIDENCE}ev_{symbol}_{indicator_type}")
            self.g.add((ev_uri, RDF.type, EVIDENCE.EvidenceBundle))
            self.g.add((ev_uri, EVIDENCE.hasConfidence, Literal(confidence)))
            self.g.add((ev_uri, EVIDENCE.supports, ind_uri))
        
        log_step(f"Indicator added: {symbol}_{indicator_type} ({signal}, conf={confidence:.2f})")
        return ind_uri
    
    def link_indicators(self, uri1: URIRef, uri2: URIRef, relationship: str):
        """Creates semantic links between indicators (confirm/contradict)."""
        prop = STOCK.confirms if relationship == "confirms" else STOCK.contradicts
        self.g.add((uri1, prop, uri2))
        log_step(f"Linked indicators: {uri1} → {relationship} → {uri2}")
    
    def link_state(self, indicator_uri: URIRef, state: str, confidence: float = 1.0):
        """Enhanced state linking with confidence."""
        state_uri = URIRef(f"{MARKET}{state}")
        self.g.add((indicator_uri, STOCK.impliesState, state_uri))
        
        if confidence < 1.0:
            ev_uri = URIRef(f"{EVIDENCE}ev_state_{hash(indicator_uri)}")
            self.g.add((ev_uri, EVIDENCE.hasConfidence, Literal(confidence)))
            self.g.add((ev_uri, EVIDENCE.supports, indicator_uri))
        
        log_step(f"State link: {indicator_uri} → {state} (conf={confidence:.2f})")
    
    def detect_contradictions(self) -> List[Tuple[URIRef, URIRef]]:
        """Finds pairs of contradictory indicator signals."""
        contradictions = []
        query = """
        SELECT ?ind1 ?ind2 WHERE {
            ?ind1 stock:contradicts ?ind2 .
            ?ind1 stock:hasSignal ?sig1 .
            ?ind2 stock:hasSignal ?sig2 .
            FILTER(?sig1 != ?sig2)
        }
        """
        for row in self.g.query(query, initNs={"stock": STOCK}):
            contradictions.append((row.ind1, row.ind2))
        return contradictions
    
    def serialize(self, format: str = "turtle") -> str:
        """Serializes with inference closure."""
        # Apply RDFS/OWL inference closure
        try:
            # Use owlrl for semantic closure if available
            import owlrl
            owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(self.g)
        except ImportError:
            log_step("owlrl not available, using basic serialization")
        except Exception as e:
            log_step(f"OWL inference error: {e}, using basic serialization")

        return self.g.serialize(format=format)

# ============================================================
# PART 2: ENHANCED INFERENCE ENGINE
# ============================================================

class SignalType(Enum):
    """Standardized signal vocabulary."""
    BULLISH_STRONG = "bullish_strong"
    BULLISH_MODERATE = "bullish_moderate"
    BEARISH_STRONG = "bearish_strong"
    BEARISH_MODERATE = "bearish_moderate"
    NEUTRAL = "neutral"
    OVERSOLD = "oversold"
    OVERBOUGHT = "overbought"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


class MarketState(Enum):
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS_CONSOLIDATION = "sideways_consolidation"
    VOLATILE_BREAKOUT = "volatile_breakout"
    RANGE_BOUND = "range_bound"


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
    """Enhanced context with confidence scores."""
    market_state: MarketState
    trend_direction: TrendDirection
    risk_level: RiskLevel
    confidence_score: float
    volatility_regime: str
    volume_profile: str
    support_levels: List[float]
    resistance_levels: List[float]
    ontology_graph: str
    reasoning_chain: List[str]
    contradictions: List[Dict[str, str]]


class EnhancedStockAnalysisOntology:
    """
    Patent-grade OWL reasoning engine with:
    - 15+ indicator extractors
    - Weighted evidence aggregation
    - Contradiction detection
    - Temporal semantics
    - Dynamic reasoning trace generation
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.version = "6.0-owl-enhanced"
        self._context_cache: Dict[str, MarketContext] = {}
        
        # Weight configurations (tune these for sensitivity)
        self.indicator_weights = {
            "trend": 0.30, "momentum": 0.25, "volume": 0.20, 
            "volatility": 0.15, "support_resistance": 0.10
        }

    def _safe_get_value(self, df, column, default=0.0, index_offset=0):
        """Safely get value from DataFrame column with fallback and historical lookback."""
        if (column in df.columns and len(df) > abs(index_offset) and 
            not pd.isna(df[column].iloc[index_offset])):
            return df[column].iloc[index_offset]
        return default
    def infer_market_context(self, symbol: str, df: pd.DataFrame) -> MarketContext:
        """Main pipeline with full indicator coverage."""
        cache_key = f"{symbol}_{len(df)}_{df.index[-1].strftime('%Y%m%d')}"
        if cache_key in self._context_cache:
            return self._context_cache[cache_key]
        
        if len(df) < 50:
            return self._default_context()
        
        graph = StockOntologyGraph()
        timestamp = df.index[-1].isoformat()
        
        # Extract ALL indicator categories
        extracts = {
            "trend": self._extract_trend(symbol, df, graph, timestamp),
            "momentum": self._extract_momentum(symbol, df, graph, timestamp),
            "volume": self._extract_volume(symbol, df, graph, timestamp),
            "volatility": self._extract_volatility(symbol, df, graph, timestamp),
            "ichimoku": self._extract_ichimoku(symbol, df, graph, timestamp),
            "fibonacci": self._extract_fibonacci(symbol, df, graph, timestamp)
        }
        
        # Detect contradictions early
        contradictions = graph.detect_contradictions()
        
        # Weighted inference
        market_state, state_conf = self._infer_market_state_weighted(extracts)
        trend_direction, trend_conf = self._infer_trend_direction_weighted(extracts)
        risk_level, risk_conf = self._infer_risk_level_weighted(extracts)
        
        # Aggregate confidence
        overall_confidence = (state_conf * 0.4 + trend_conf * 0.3 + risk_conf * 0.3)
        
        # S/R levels
        sr_levels = self._calculate_sr_levels(df)
        
        # Build dynamic reasoning trace
        reasoning_chain = self._build_dynamic_reasoning(
            symbol, extracts, market_state, trend_direction, risk_level,
            contradictions, overall_confidence
        )
        
        # Link all to market state
        for category in extracts.values():
            for uri in category.get("uris", []):
                graph.link_state(uri, market_state.value, confidence=category.get("avg_confidence", 1.0))
        
        context = MarketContext(
            market_state=market_state,
            trend_direction=trend_direction,
            risk_level=risk_level,
            confidence_score=round(overall_confidence, 3),
            volatility_regime=extracts["volatility"]["regime"],
            volume_profile=extracts["volume"]["profile"],
            support_levels=sr_levels["support"],
            resistance_levels=sr_levels["resistance"],
            ontology_graph=graph.serialize(),
            reasoning_chain=reasoning_chain,
            contradictions=[{"indicator1": str(c[0]), "indicator2": str(c[1])} for c in contradictions]
        )
        
        self._context_cache[cache_key] = context
        return context
    
    # ============================================================
    # COMPREHENSIVE INDICATOR EXTRACTORS
    # ============================================================
    
    def _extract_trend(self, symbol, df, graph, timestamp):
        """Enhanced trend extraction with EMAs and DI+/DI-."""
        closes = df["close"]
        entities = {"uris": [], "signals": [], "confidences": []}
        
        # Moving averages with confidence scoring
        for period, weight in [(20, 0.3), (50, 0.4), (200, 0.3)]:
            sma = closes.rolling(period).mean().iloc[-1]
            ema = closes.ewm(span=period).mean().iloc[-1]
            current = closes.iloc[-1]
            
            # SMA signal
            sma_signal, sma_conf = self._classify_ma_signal(current, sma)
            sma_uri = graph.add_indicator(symbol, f"SMA_{period}", sma, sma_signal, sma_conf, 
                                        {"timestamp": timestamp})
            entities["uris"].append(sma_uri)
            entities["signals"].append(sma_signal)
            entities["confidences"].append(sma_conf * weight)
            
            # EMA signal (more responsive, higher weight)
            ema_signal, ema_conf = self._classify_ma_signal(current, ema, threshold=0.015)
            ema_uri = graph.add_indicator(symbol, f"EMA_{period}", ema, ema_signal, ema_conf,
                                        {"timestamp": timestamp})
            entities["uris"].append(ema_uri)
            entities["signals"].append(ema_signal)
            entities["confidences"].append(ema_conf * weight * 1.2)
        
        # ADX with directional components
        adx_ind = ta.trend.ADXIndicator(df["high"], df["low"], df["close"])
        adx_value = adx_ind.adx().iloc[-1]
        di_plus = adx_ind.adx_pos().iloc[-1]
        di_minus = adx_ind.adx_neg().iloc[-1]
        
        adx_strength, adx_conf = self._classify_adx(adx_value)
        trend_strength = "strong" if adx_value > 25 else "weak"
        
        adx_uri = graph.add_indicator(symbol, "ADX", adx_value, adx_strength, adx_conf,
                                    {"timestamp": timestamp})
        entities["uris"].append(adx_uri)
        entities["signals"].append(adx_strength)
        entities["confidences"].append(adx_conf * 0.5)
        
        # DI+/- signals
        di_signal, di_conf = "bullish" if di_plus > di_minus else "bearish", abs(di_plus - di_minus) / 100
        di_uri = graph.add_indicator(symbol, "DI_Cross", di_plus - di_minus, di_signal, di_conf,
                                   {"timestamp": timestamp})
        entities["uris"].append(di_uri)
        entities["confidences"].append(di_conf * 0.3)
        
        entities["avg_confidence"] = sum(entities["confidences"]) / len(entities["confidences"]) if entities["confidences"] else 0.5
        entities["trend_strength"] = trend_strength
        entities["di_signal"] = di_signal
        
        return entities
    
    def _extract_momentum(self, symbol, df, graph, timestamp):
        """Complete momentum suite: RSI, MACD, Stochastic, CCI."""
        closes = df["close"]
        entities = {"uris": [], "signals": [], "confidences": []}
        
        # RSI (overbought/oversold)
        rsi_val = ta.momentum.RSIIndicator(closes).rsi().iloc[-1]
        rsi_signal, rsi_conf = self._classify_rsi(rsi_val)
        rsi_uri = graph.add_indicator(symbol, "RSI", rsi_val, rsi_signal, rsi_conf,
                                    {"timestamp": timestamp})
        entities["uris"].append(rsi_uri)
        entities["signals"].append(rsi_signal)
        entities["confidences"].append(rsi_conf * 0.25)
        
        # MACD with histogram
        macd_ind = ta.trend.MACD(closes)
        macd_val = macd_ind.macd().iloc[-1]
        macd_signal = macd_ind.macd_signal().iloc[-1]
        macd_hist = macd_ind.macd_diff().iloc[-1]
        
        macd_signal_type, macd_conf = self._classify_macd(macd_val, macd_signal, macd_hist)
        macd_uri = graph.add_indicator(symbol, "MACD", macd_val, macd_signal_type, macd_conf,
                                     {"timestamp": timestamp})
        entities["uris"].append(macd_uri)
        entities["signals"].append(macd_signal_type)
        entities["confidences"].append(macd_conf * 0.25)
        
        # Stochastic Oscillator
        stoch_ind = ta.momentum.StochasticOscillator(df["high"], df["low"], closes)
        stoch_k = stoch_ind.stoch().iloc[-1]
        stoch_d = stoch_ind.stoch_signal().iloc[-1]
        
        stoch_signal, stoch_conf = self._classify_stochastic(stoch_k, stoch_d)
        stoch_uri = graph.add_indicator(symbol, "Stochastic", stoch_k, stoch_signal, stoch_conf,
                                      {"timestamp": timestamp})
        entities["uris"].append(stoch_uri)
        entities["confidences"].append(stoch_conf * 0.25)
        
        # CCI (Commodity Channel Index)
        cci_val = ta.trend.CCIIndicator(df["high"], df["low"], closes).cci().iloc[-1]
        cci_signal, cci_conf = self._classify_cci(cci_val)
        cci_uri = graph.add_indicator(symbol, "CCI", cci_val, cci_signal, cci_conf,
                                    {"timestamp": timestamp})
        entities["uris"].append(cci_uri)
        entities["confidences"].append(cci_conf * 0.25)
        
        entities["avg_confidence"] = sum(entities["confidences"]) / len(entities["confidences"])
        return entities
    
    def _extract_volume(self, symbol, df, graph, timestamp):
        """Comprehensive volume profile: OBV, VWAP, MFI, CMF, ADL, ForceIndex."""
        closes, vols = df["close"], df["volume"]
        entities = {"uris": [], "profile": "neutral", "confidence": 0.5}

        # OBV
        obv_val = ta.volume.OnBalanceVolumeIndicator(closes, vols).on_balance_volume().iloc[-1]
        obv_prev = df["OBV"].iloc[-5] if len(df) > 5 else obv_val
        obv_signal = "accumulation" if obv_val > obv_prev else "distribution"
        obv_conf = min(abs(obv_val - obv_prev) / abs(obv_prev), 1.0) if obv_prev != 0 else 0.5

        obv_uri = graph.add_indicator(symbol, "OBV", obv_val, obv_signal, obv_conf,
                                    {"timestamp": timestamp})
        entities["uris"].append(obv_uri)

        # VWAP deviation
        vwap_val = self._safe_get_value(df, "VWAP", df["close"].iloc[-1])
        vwap_dev = (closes.iloc[-1] - vwap_val) / vwap_val
        vwap_signal = "above_vwap" if vwap_dev > 0 else "below_vwap"
        vwap_conf = min(abs(vwap_dev) * 10, 1.0)

        vwap_uri = graph.add_indicator(symbol, "VWAP", vwap_val, vwap_signal, vwap_conf,
                                     {"timestamp": timestamp})
        entities["uris"].append(vwap_uri)

        # MFI (Money Flow Index)
        mfi_val = ta.volume.MFIIndicator(df["high"], df["low"], closes, vols).money_flow_index().iloc[-1]
        mfi_signal, mfi_conf = self._classify_mfi(mfi_val)
        mfi_uri = graph.add_indicator(symbol, "MFI", mfi_val, mfi_signal, mfi_conf,
                                    {"timestamp": timestamp})
        entities["uris"].append(mfi_uri)

        # CMF (Chaikin Money Flow)
        cmf_val = ta.volume.ChaikinMoneyFlowIndicator(df["high"], df["low"], closes, vols).chaikin_money_flow().iloc[-1]
        cmf_signal = "accumulation" if cmf_val > 0 else "distribution"
        cmf_conf = min(abs(cmf_val) * 5, 1.0)

        cmf_uri = graph.add_indicator(symbol, "CMF", cmf_val, cmf_signal, cmf_conf,
                                    {"timestamp": timestamp})
        entities["uris"].append(cmf_uri)

        # ADL (Accumulation/Distribution Line)
        adl_val = self._safe_get_value(df, "ADL", 0.0)
        adl_prev = self._safe_get_value(df, "ADL", adl_val, -5)
        adl_signal = "accumulation" if adl_val > adl_prev else "distribution"
        adl_conf = min(abs(adl_val - adl_prev) / abs(adl_prev), 1.0) if adl_prev != 0 else 0.5

        adl_uri = graph.add_indicator(symbol, "ADL", adl_val, adl_signal, adl_conf,
                                    {"timestamp": timestamp})
        entities["uris"].append(adl_uri)

        # Force Index - FIXED
        fi_val = ta.volume.ForceIndexIndicator(closes, vols).force_index().iloc[-1]
        fi_signal = "positive_force" if fi_val > 0 else "negative_force"

        # Calculate FI confidence using a simpler approach
        # Use a fixed threshold or relative to recent average
        fi_series = ta.volume.ForceIndexIndicator(closes, vols).force_index().tail(20)
        if len(fi_series) > 0:
            fi_avg = abs(fi_series).mean()
            fi_conf = min(abs(fi_val) / max(fi_avg, 1.0), 1.0)  # Avoid division by zero
        else:
            fi_conf = 0.5

        fi_uri = graph.add_indicator(symbol, "ForceIndex", fi_val, fi_signal, fi_conf,
                                   {"timestamp": timestamp})
        entities["uris"].append(fi_uri)

        # Aggregate volume profile
        acc_signals = [obv_signal, cmf_signal, adl_signal]
        acc_count = sum(1 for s in acc_signals if "accumulation" in s)
        if acc_count >= 2:
            entities["profile"] = "strong_accumulation"
            entities["confidence"] = 0.8
        elif "distribution" in acc_signals:
            entities["profile"] = "distribution"
            entities["confidence"] = 0.6

        return entities

    def _extract_volatility(self, symbol, df, graph, timestamp):
        """Volatility regime: ATR, Bollinger Bands width, CCI volatility."""
        closes = df["close"]
        entities = {"uris": [], "regime": "medium", "confidence": 0.5}
        
        # ATR%
        atr_val = ta.volatility.AverageTrueRange(df["high"], df["low"], closes).average_true_range().iloc[-1]
        atr_pct = (atr_val / closes.iloc[-1]) * 100
        
        if atr_pct > 5:
            vol_signal, vol_conf, regime = "high_volatility", 0.9, "high"
        elif atr_pct < 2:
            vol_signal, vol_conf, regime = "low_volatility", 0.9, "low"
        else:
            vol_signal, vol_conf, regime = "medium_volatility", 0.7, "medium"
        
        atr_uri = graph.add_indicator(symbol, "ATR%", atr_pct, vol_signal, vol_conf,
                                    {"timestamp": timestamp})
        entities["uris"].append(atr_uri)
        entities["regime"] = regime
        entities["confidence"] = vol_conf
        
        # Bollinger Bands squeeze/expansion
        bb_width = (df["Upper_band"].iloc[-1] - df["Lower_band"].iloc[-1]) / df["SMA_20"].iloc[-1] if all(col in df.columns for col in ["Upper_band", "Lower_band", "SMA_20"]) else 0.1
        if bb_width < 0.05:
            bb_signal, bb_conf = "squeeze", 0.85
        elif bb_width > 0.15:
            bb_signal, bb_conf = "expansion", 0.7
        else:
            bb_signal, bb_conf = "normal", 0.5
        
        bb_uri = graph.add_indicator(symbol, "BollingerWidth", bb_width * 100, bb_signal, bb_conf,
                                   {"timestamp": timestamp})
        entities["uris"].append(bb_uri)
        
        return entities
        
    def _extract_ichimoku(self, symbol, df, graph, timestamp):
        """Ichimoku Cloud signals."""
        entities = {"uris": [], "signals": []}

        # Check if Ichimoku columns exist, if not return empty
        required_cols = ["Tenkan_sen", "Kijun_sen", "Senkou_span_a", "Senkou_span_b", "Chikou_span"]
        if not all(col in df.columns for col in required_cols):
            return entities

        tenkan, kijun = df["Tenkan_sen"].iloc[-1], df["Kijun_sen"].iloc[-1]
        senkou_a, senkou_b = df["Senkou_span_a"].iloc[-1], df["Senkou_span_b"].iloc[-1]
        chikou = df["Chikou_span"].iloc[-26] if len(df) > 26 else df["close"].iloc[-1]
        current = df["close"].iloc[-1]

        # TK cross
        tk_signal = "bullish" if tenkan > kijun else "bearish"
        tk_uri = graph.add_indicator(symbol, "Ichimoku_TK", tenkan - kijun, tk_signal, 0.7,
                                   {"timestamp": timestamp})
        entities["uris"].append(tk_uri)
        
        # Price vs Cloud
        cloud_top, cloud_bottom = max(senkou_a, senkou_b), min(senkou_a, senkou_b)
        if current > cloud_top:
            price_signal, price_conf = "above_cloud", 0.85
        elif current < cloud_bottom:
            price_signal, price_conf = "below_cloud", 0.85
        else:
            price_signal, price_conf = "in_cloud", 0.5
        
        price_uri = graph.add_indicator(symbol, "Ichimoku_PriceVsCloud", current, price_signal, price_conf,
                                      {"timestamp": timestamp})
        entities["uris"].append(price_uri)
        
        # Lagging span
        lag_signal = "bullish" if chikou > current else "bearish"
        lag_uri = graph.add_indicator(symbol, "Ichimoku_Chikou", chikou, lag_signal, 0.6,
                                    {"timestamp": timestamp})
        entities["uris"].append(lag_uri)
        
        # Link confirmations
        if tk_signal == price_signal == lag_signal:
            for i in range(len(entities["uris"]) - 1):
                graph.link_indicators(entities["uris"][i], entities["uris"][i+1], "confirms")
        
        return entities
    
    def _extract_fibonacci(self, symbol, df, graph, timestamp):
        """Fibonacci retracement levels as dynamic support/resistance."""
        entities = {"uris": []}
        high, low = df["high"].max(), df["low"].min()
        diff = high - low
        
        levels = {
            "0%": high, "23.6%": high - 0.236 * diff, "38.2%": high - 0.382 * diff,
            "50%": high - 0.5 * diff, "61.8%": high - 0.618 * diff, "100%": low
        }
        
        current = df["close"].iloc[-1]
        for name, level in levels.items():
            proximity = abs(current - level) / current
            conf = max(1 - proximity * 2, 0.2)
            
            lvl_uri = graph.add_indicator(symbol, f"Fib_{name}", level, 
                                        "support_resistance", conf, {"timestamp": timestamp})
            entities["uris"].append(lvl_uri)
        
        return entities
    
    def _calculate_sr_levels(self, df):
        """Dynamic S/R using quantiles and recent pivots."""
        recent = df.tail(30)
        return {
            "support": sorted([float(recent["low"].min()), 
                             float(recent["low"].quantile(0.25)),  # REMOVED EXTRA PARENTHESIS
                             float(recent["low"].quantile(0.1))]),
            "resistance": sorted([float(recent["high"].max()),
                                float(recent["high"].quantile(0.75)),
                                float(recent["high"].quantile(0.9))], reverse=True)
        }
    # ============================================================
    # WEIGHTED INFERENCE RULES
    # ============================================================
    
    def _infer_market_state_weighted(self, extracts) -> Tuple[MarketState, float]:
        """Weighted scoring across all evidence."""
        scores = {state: 0.0 for state in MarketState}
        confidences = {state: [] for state in MarketState}
        
        # Trend evidence
        t = extracts["trend"]
        if t.get("trend_strength") in ["strong", "very_strong"]:
            if "bullish" in t.get("di_signal", ""):
                scores[MarketState.BULL_TREND] += self.indicator_weights["trend"]
                confidences[MarketState.BULL_TREND].append(t["avg_confidence"])
            else:
                scores[MarketState.BEAR_TREND] += self.indicator_weights["trend"]
                confidences[MarketState.BEAR_TREND].append(t["avg_confidence"])
        
        # Momentum evidence
        m = extracts["momentum"]
        bullish_mom = sum(1 for s in m["signals"] if "bullish" in s)
        bearish_mom = sum(1 for s in m["signals"] if "bearish" in s)
        
        if bullish_mom >= 2:
            scores[MarketState.BULL_TREND] += self.indicator_weights["momentum"]
            confidences[MarketState.BULL_TREND].append(m["avg_confidence"])
        elif bearish_mom >= 2:
            scores[MarketState.BEAR_TREND] += self.indicator_weights["momentum"]
            confidences[MarketState.BEAR_TREND].append(m["avg_confidence"])
        
        # Volume evidence
        v = extracts["volume"]
        if "strong_accumulation" in v["profile"]:
            scores[MarketState.BULL_TREND] += self.indicator_weights["volume"] * 1.5
            confidences[MarketState.BULL_TREND].append(v["confidence"])
        elif "distribution" in v["profile"]:
            scores[MarketState.BEAR_TREND] += self.indicator_weights["volume"]
            confidences[MarketState.BEAR_TREND].append(v["confidence"])
        
        # Volatility regime
        vol = extracts["volatility"]
        if vol["regime"] == "high":
            scores[MarketState.VOLATILE_BREAKOUT] += self.indicator_weights["volatility"]
            confidences[MarketState.VOLATILE_BREAKOUT].append(vol["confidence"])
        elif vol["regime"] == "low" and max(scores.values()) < 0.3:
            scores[MarketState.RANGE_BOUND] += self.indicator_weights["volatility"]
            confidences[MarketState.RANGE_BOUND].append(vol["confidence"])
        
        # Select winner
        winning_state = max(scores.items(), key=lambda x: x[1])[0] if scores else MarketState.SIDEWAYS_CONSOLIDATION
        avg_conf = sum(confidences.get(winning_state, [0.5])) / max(len(confidences.get(winning_state, [])), 1)
        
        return winning_state, avg_conf
    
    def _infer_trend_direction_weighted(self, extracts) -> Tuple[TrendDirection, float]:
        """Multi-factor trend direction scoring."""
        bullish_score = 0.0
        total_conf = 0.0
        
        # Trend indicators
        t = extracts["trend"]
        if t.get("trend_strength") == "very_strong":
            bullish_score += 2.0
            total_conf += t["avg_confidence"]
        elif t.get("trend_strength") == "strong":
            bullish_score += 1.5
            total_conf += t["avg_confidence"]
        
        # Momentum
        m = extracts["momentum"]
        bullish_mom = sum(1 for s in m["signals"] if "bullish" in s)
        bullish_score += bullish_mom * 0.8
        total_conf += m["avg_confidence"]
        
        # Ichimoku
        ich = extracts.get("ichimoku", {})
        if ich.get("signals", []).count("bullish") >= 2:
            bullish_score += 1.2
        
        # Volume confirmation
        v = extracts["volume"]
        if "accumulation" in v["profile"]:
            bullish_score += 0.5
        
        # Determine direction
        if bullish_score >= 3.0:
            direction = TrendDirection.STRONG_UP
        elif bullish_score >= 1.5:
            direction = TrendDirection.MODERATE_UP
        elif bullish_score <= -3.0:
            direction = TrendDirection.STRONG_DOWN
        elif bullish_score <= -1.5:
            direction = TrendDirection.MODERATE_DOWN
        else:
            direction = TrendDirection.NEUTRAL
        
        avg_conf = total_conf / 3 if total_conf > 0 else 0.5
        return direction, avg_conf
    
    def _infer_risk_level_weighted(self, extracts) -> Tuple[RiskLevel, float]:
        """Multi-dimensional risk assessment."""
        risk_score = 0.0
        confidences = []
        
        # Volatility risk (primary factor)
        vol = extracts["volatility"]
        if vol["regime"] == "high":
            risk_score += 4.0
            confidences.append(vol["confidence"])
        elif vol["regime"] == "medium":
            risk_score += 2.0
            confidences.append(vol["confidence"])
        
        # Trend risk (counter-trend increases risk)
        t = extracts["trend"]
        if t.get("trend_strength") == "weak":
            risk_score += 1.0
            confidences.append(0.6)
        
        # Momentum exhaustion risk
        m = extracts["momentum"]
        if "overbought" in m["signals"] or "oversold" in m["signals"]:
            risk_score += 1.5
            confidences.append(0.7)
        
        # Map to RiskLevel
        if risk_score >= 4.5:
            level = RiskLevel.VERY_HIGH
        elif risk_score >= 3.5:
            level = RiskLevel.HIGH
        elif risk_score >= 2.5:
            level = RiskLevel.MEDIUM
        elif risk_score >= 1.5:
            level = RiskLevel.LOW
        else:
            level = RiskLevel.VERY_LOW
        
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.5
        return level, avg_conf
    
    # ============================================================
    # SIGNAL CLASSIFICATION HELPERS
    # ============================================================
    
    def _classify_ma_signal(self, current, ma_value, threshold: float = 0.02):
        """Classifies price vs MA relationship."""
        deviation = (current - ma_value) / ma_value
        if deviation > threshold:
            return "strong_above", min(deviation * 2, 1.0)
        elif deviation > 0:
            return "above", deviation * 1.5
        elif deviation < -threshold:
            return "strong_below", min(abs(deviation) * 2, 1.0)
        else:
            return "below", abs(deviation) * 1.5
    
    def _classify_adx(self, value: float) -> Tuple[str, float]:
        if value > 40:
            return "very_strong", 0.95
        elif value > 25:
            return "strong", 0.8
        elif value > 20:
            return "moderate", 0.6
        return "weak", 0.4
    
    def _classify_rsi(self, value: float) -> Tuple[str, float]:
        if value > 80:
            return "extremely_overbought", 0.95
        elif value > 70:
            return "overbought", 0.8
        elif value < 20:
            return "extremely_oversold", 0.95
        elif value < 30:
            return "oversold", 0.8
        return "neutral", 0.5
    

# Add this import at the top:

    def _classify_macd(self, macd: float, signal: float, hist: float) -> Tuple[str, float]:
        # Use a simple threshold instead of quantile on a single value
        strength = "strong_" if abs(hist) > 0.02 else ""  # Example threshold
        if macd > signal and macd > 0:
            return f"{strength}bullish", 0.85 if "strong" in strength else 0.6
        elif macd < signal and macd < 0:
            return f"{strength}bearish", 0.85 if "strong" in strength else 0.6
        return "neutral", 0.4

    def _classify_stochastic(self, k: float, d: float) -> Tuple[str, float]:
        if k > 80 and d > 80:
            return "overbought", 0.8
        elif k < 20 and d < 20:
            return "oversold", 0.8
        elif k > d and k < 50:
            return "bullish_cross", 0.7
        elif k < d and k > 50:
            return "bearish_cross", 0.7
        return "neutral", 0.5
    
    def _classify_mfi(self, value: float) -> Tuple[str, float]:
        return self._classify_rsi(value)  # Similar logic
    
    def _classify_cci(self, value: float) -> Tuple[str, float]:
        if value > 100:
            return "overbought", 0.7
        elif value < -100:
            return "oversold", 0.7
        return "neutral", 0.5
    
    # ============================================================
    # DYNAMIC REASONING TRACE
    # ============================================================
    
    def _build_dynamic_reasoning(self, symbol, extracts, market_state, trend_direction,
                               risk_level, contradictions, confidence) -> List[str]:
        """Generates trace from actual evidence weights."""
        chain = [
            f"Analysis for {symbol} at {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Overall Confidence: {confidence:.1%}",
            f"Market State: {market_state.value.replace('_', ' ').title()} (Score: {self._format_confidence(confidence)})"
        ]
        
        # Add top evidence
        top_evidence = []
        for category, data in extracts.items():
            if "avg_confidence" in data and data["avg_confidence"] > 0.7:
                top_evidence.append(f"- {category.title()}: High confidence signals")
        
        if top_evidence:
            chain.append("Key Evidence:")
            chain.extend(top_evidence)
        
        # Contradictions
        if contradictions:
            chain.append(f"⚠️ Detected {len(contradictions)} indicator contradictions")
        
        # Risk justification
        chain.append(f"Risk Assessment: {risk_level.value.replace('_', ' ').title()}")
        
        return chain
    
    def _format_confidence(self, conf: float) -> str:
        if conf > 0.8:
            return "Very High"
        elif conf > 0.6:
            return "High"
        elif conf > 0.4:
            return "Moderate"
        return "Low"
    
    def _default_context(self):
        return MarketContext(
            market_state=MarketState.SIDEWAYS_CONSOLIDATION,
            trend_direction=TrendDirection.NEUTRAL,
            risk_level=RiskLevel.MEDIUM,
            confidence_score=0.0,
            volatility_regime="unknown",
            volume_profile="unknown",
            support_levels=[],
            resistance_levels=[],
            ontology_graph="",
            reasoning_chain=["Insufficient data (need ≥ 50 bars)."],
            contradictions=[]
        )
    
    def generate_summary(self, symbol: str, df: pd.DataFrame):
        """Unchanged interface for backward compatibility."""
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
                "confidence": ctx.confidence_score
            },
            "ontology_graph": ctx.ontology_graph,
            "reasoning_chain": ctx.reasoning_chain,
            "contradictions": ctx.contradictions
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
    
    # Make a copy to avoid modifying original
    result_df = df.copy()

    try:
        # ── Moving Averages (Trend Layer)
        for w in [8, 20, 50, 200]:
            result_df[f"SMA_{w}"] = closes.rolling(w).mean()
            result_df[f"EMA_{w}"] = closes.ewm(span=w, adjust=False).mean()

        # ── Momentum Indicators
        result_df["RSI"] = ta.momentum.RSIIndicator(closes).rsi()

        macd = ta.trend.MACD(closes)
        result_df["MACD"], result_df["MACD_Signal"] = macd.macd(), macd.macd_signal()

        stoch = ta.momentum.StochasticOscillator(highs, lows, closes)
        result_df["%K"], result_df["%D"] = stoch.stoch(), stoch.stoch_signal()

        # ── Volatility Indicators
        ma20, std20 = closes.rolling(20).mean(), closes.rolling(20).std()
        result_df["Upper_band"], result_df["Lower_band"] = ma20 + 2 * std20, ma20 - 2 * std20
        result_df["ATR"] = ta.volatility.AverageTrueRange(highs, lows, closes).average_true_range()
        result_df["CCI"] = ta.trend.CCIIndicator(highs, lows, closes).cci()

        # ── Volume Indicators
        result_df["OBV"] = ta.volume.OnBalanceVolumeIndicator(closes, vols).on_balance_volume()
        result_df["VWAP"] = (closes * vols).cumsum() / vols.cumsum()
        result_df["ADL"] = ta.volume.AccDistIndexIndicator(highs, lows, closes, vols).acc_dist_index()
        result_df["MFI"] = ta.volume.MFIIndicator(highs, lows, closes, vols).money_flow_index()
        result_df["CMF"] = ta.volume.ChaikinMoneyFlowIndicator(highs, lows, closes, vols).chaikin_money_flow()
        result_df["FI"]  = ta.volume.ForceIndexIndicator(closes, vols).force_index()

        # ── Trend Strength Indicators
        adx = ta.trend.ADXIndicator(highs, lows, closes)
        result_df["ADX"], result_df["DI+"], result_df["DI-"] = adx.adx(), adx.adx_pos(), adx.adx_neg()

        # ── Ichimoku Cloud (Support/Resistance Layer)
        result_df["Tenkan_sen"]   = (highs.rolling(9).max() + lows.rolling(9).min()) / 2
        result_df["Kijun_sen"]    = (highs.rolling(26).max() + lows.rolling(26).min()) / 2
        result_df["Senkou_span_a"] = ((result_df["Tenkan_sen"] + result_df["Kijun_sen"]) / 2).shift(26)
        result_df["Senkou_span_b"] = ((highs.rolling(52).max() + lows.rolling(52).min()) / 2).shift(26)
        result_df["Chikou_span"]   = closes.shift(-26)

        log_step("Indicators computed successfully.")
        
    except Exception as e:
        log_step(f"Warning: Some indicators failed to compute: {e}")
        # Return at least the basic computed indicators
        pass

    return result_df

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
    app.run_server(debug=False, port=8050)
