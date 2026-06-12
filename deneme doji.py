import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone
import xgboost as xgb
import concurrent.futures

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="AI Doji Terminal", layout="wide", initial_sidebar_state="auto")

# --- CACHE ACCELERATOR: CACHE FUNCTION ---
@st.cache_data(ttl=300) 
def veri_indir(symbol, periyot, interval):
    return yf.download(symbol, period=periyot, interval=interval, progress=False)

# --- GLOBAL MARKET DEFINITIONS ---
MARKETS = [
    {"name": "Gold (XAU/USD)", "symbol": "GC=F", "tv": "OANDA:XAUUSD", "category": "Commodity", "color": "#F59E0B"},
    {"name": "Silver (XAG/USD)", "symbol": "SI=F", "tv": "OANDA:XAGUSD", "category": "Commodity", "color": "#94A3B8"},
    {"name": "EUR/USD", "symbol": "EURUSD=X", "tv": "OANDA:EURUSD", "category": "Forex", "color": "#3B82F6"},
    {"name": "NASDAQ Endeksi", "symbol": "^IXIC", "tv": "NASDAQ:IXIC", "category": "NASDAQ", "color": "#A855F7"},
    {"name": "Apple", "symbol": "AAPL", "tv": "NASDAQ:AAPL", "category": "NASDAQ", "color": "#60A5FA"},
    {"name": "Tesla", "symbol": "TSLA", "tv": "NASDAQ:TSLA", "category": "NASDAQ", "color": "#EF4444"},
    {"name": "NVIDIA", "symbol": "NVDA", "tv": "NASDAQ:NVDA", "category": "NASDAQ", "color": "#10B981"},
    {"name": "Microsoft", "symbol": "MSFT", "tv": "NASDAQ:MSFT", "category": "NASDAQ", "color": "#3B82F6"},
    {"name": "Bitcoin", "symbol": "BTC-USD", "tv": "BINANCE:BTCUSDT", "category": "Kripto", "color": "#F59E0B"},
    {"name": "Ethereum", "symbol": "ETH-USD", "tv": "BINANCE:ETHUSDT", "category": "Kripto", "color": "#6366F1"}
]

# --- LIVE DATA FUNCTIONS ---
def get_crypto_fng():
    try:
        response = requests.get("https://api.alternative.me/fng/", timeout=5).json()
        value = int(response['data'][0]['value'])
        classification = response['data'][0]['value_classification']
        tr_map = {
            "Extreme Fear": ("Extreme Fear 📊", "#EF4444"),
            "Fear": ("Fear 📊", "#F97316"),
            "Neutral": ("Neutral 📊", "#94A3B8"),
            "Greed": ("Greed 📊", "#10B981"),
            "Extreme Greed": ("Extreme Greed 📊", "#34D399")
        }
        status, color = tr_map.get(classification, (classification, "#94A3B8"))
        return value, status, color
    except:
        return 50, "Neutral 📊", "#94A3B8"

def get_nasdaq_fng():
    try:
        vix_df = yf.download("^VIX", period="5d", interval="1d", progress=False)
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = vix_df.columns.get_level_values(0)
            
        son_vix = float(vix_df['Close'].iloc[-1])
        fng_score = 100 - ((son_vix - 10) / (40 - 10) * 100)
        fng_score = max(0, min(100, int(fng_score))) 
        
        if fng_score < 25: return fng_score, "Extreme Fear 📊", "#EF4444"
        elif fng_score < 45: return fng_score, "Fear 📊", "#F97316"
        elif fng_score < 55: return fng_score, "Neutral 📊", "#94A3B8"
        elif fng_score < 75: return fng_score, "Greed 📊", "#10B981"
        else: return fng_score, "Extreme Greed 📊", "#34D399"
    except:
        return 50, "Neutral 📊", "#94A3B8"

def get_real_market_dynamics(symbols):
    try:
        vol_results = []
        vol_ratios = []
        vol_counts = 0
        for sym in symbols:
            df = veri_indir(sym, "30d", "1d")
            if df.empty or len(df) < 20: continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            high_low = df['High'] - df['Low']
            high_close = abs(df['High'] - df['Close'].shift())
            low_close = abs(df['Low'] - df['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr20 = true_range.rolling(20).mean().iloc[-1]
            son_fiyat = df['Close'].iloc[-1]
            vol_ratio = (atr20 / son_fiyat) * 100
            vol_ratios.append(vol_ratio)
            
            hacim_sutunu = df['Volume'] if 'Volume' in df.columns else None
            if hacim_sutunu is not None:
                aktif_hacim_idx = -1 if df['Volume'].iloc[-1] > 0 else -2
                son_hacim = df['Volume'].rolling(3).mean().iloc[aktif_hacim_idx]
                ort_hacim = df['Volume'].rolling(20).mean().iloc[aktif_hacim_idx]
                hacim_soku = son_hacim / (ort_hacim + 1e-10)
                vol_results.append(hacim_soku)
                vol_counts += 1

        final_vol, final_vol_clr, final_hac = "Low 📊", "#64748B", "No Data 📊"
        if vol_ratios:
            avg_vol_ratio = sum(vol_ratios) / len(vol_ratios)
            if avg_vol_ratio > 2.0: final_vol, final_vol_clr = "High 📊", "#34D399"
            elif avg_vol_ratio > 1.0: final_vol, final_vol_clr = "Normal 📊", "#F59E0B"
            
        if vol_counts > 0 and sum(vol_results) > 0:
            avg_hacim_soku = sum(vol_results) / vol_counts
            if avg_hacim_soku > 1.15: final_hac = "Strong 📊"
            elif avg_hacim_soku > 0.85: final_hac = "Normal 📊"
            else: final_hac = "Weak 📊"
            
        return final_vol, final_vol_clr, final_hac
    except:
        return "Normal 📊", "#F59E0B", "Normal 📊"

def dinamik_piyasa_durumu():
    gun = datetime.now(timezone.utc).weekday()
    if gun >= 5:
        return "Weekend Closed 📊"
    else:
        return "Open / Trading 📊"

def piyasa_rejimi_hesapla(symbol):
    try:
        df = veri_indir(symbol, "60d", "1d")
        if df.empty: return "No Data", "#334155"
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs)).iloc[-1]

        ema20 = df['EMA20'].iloc[-1]
        ema50 = df['EMA50'].iloc[-1]

        if ema20 > ema50 and rsi > 60: return "Strong Bull 📊", "#10B981"
        elif ema20 > ema50: return "Bull 📊", "#059669"
        elif ema20 < ema50 and rsi < 40: return "Strong Bear 📊", "#EF4444"
        elif ema20 < ema50: return "Bear 📊", "#B91C1C"
        else: return "Sideways •", "#64748B"
    except:
        return "Bilinmiyor", "#334155"

def buyuk_trend_kontrol(symbol):
    try:
        df_big = veri_indir(symbol, "60d", "4h")
        if df_big.empty: return "Neutral"
        if isinstance(df_big.columns, pd.MultiIndex):
            df_big.columns = df_big.columns.get_level_values(0)
        ema200 = df_big['Close'].ewm(span=200, adjust=False).mean().iloc[-1]
        son_fiyat = df_big['Close'].iloc[-1]
        return "Bull (Up)" if son_fiyat > ema200 else "Bear (Down)"
    except:
        return "Neutral"

def analiz_et_safe(market, min_hours, interval, doji_modu):
    try:
        if interval == "1h": periyot = "6mo" 
        elif interval == "4h": periyot = "2y"
        else: periyot = "5y"
        
        df = veri_indir(market["symbol"], periyot, interval)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        govde = abs(df['Open'] - df['Close'])
        toplam_boy = df['High'] - df['Low']
        
        # --- MARKET-SENSITIVE DOJI FILTER (Forex Noise Reduction) ---
        if "Dinamik" in doji_modu:
            ortalama_boy = toplam_boy.rolling(window=20).mean()
            volatilite_carpani = toplam_boy / (ortalama_boy + 1e-10)
            
            # Forex market is very narrow; halve sensitivity to eliminate false signals
            if market["category"] == "Forex":
                dinamik_sinir = (0.12 * volatilite_carpani).clip(lower=0.05, upper=0.20)
            else:
                dinamik_sinir = (0.3 * volatilite_carpani).clip(lower=0.15, upper=0.45)
                
            df['Doji'] = govde <= (toplam_boy * dinamik_sinir)
        else:
            # Even in fixed mode, apply a stricter threshold (10%) for Forex
            sinir = 0.10 if market["category"] == "Forex" else 0.30
            df['Doji'] = govde <= (toplam_boy * sinir)
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['Price_to_EMA20'] = df['Close'] / df['EMA20']
        
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        df['Upper_Shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / (df['High'] - df['Low'] + 1e-10)
        df['Lower_Shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / (df['High'] - df['Low'] + 1e-10)
        df['Volume_Shock'] = df['Volume'].rolling(5).mean() / (df['Volume'].rolling(20).mean() + 1e-10)
        
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = macd_line - macd_signal
        
        sma20 = df['Close'].rolling(window=20).mean()
        std20 = df['Close'].rolling(window=20).std()
        df['Upper_BB'] = sma20 + (2 * std20)
        df['Lower_BB'] = sma20 - (2 * std20)
        df['BB_Width'] = (df['Upper_BB'] - df['Lower_BB']) / (sma20 + 1e-10)
        df['Price_to_BB'] = (df['Close'] - df['Lower_BB']) / (df['Upper_BB'] - df['Lower_BB'] + 1e-10)
        df['Trend_Slope'] = df['EMA20'].diff(3) / (df['EMA20'] + 1e-10) * 100

        df = df.dropna()
        suanki_fiyat = df['Close']
        ilerideki_kapanis = df['Close'].shift(-int(min_hours))
        ilerideki_min = df['Low'].shift(-int(min_hours)).rolling(window=int(min_hours)).min()
        
        # BUY target: 1 if future close >1% above AND stop not breached
        buy_target = np.where(
            (ilerideki_kapanis > suanki_fiyat * 1.01) &
            (ilerideki_min >= suanki_fiyat * 0.99),
            1, 0
        )
        # SELL target: 1 if future close >1% below
        sell_target = np.where(
            ilerideki_kapanis < suanki_fiyat * 0.99,
            1, 0
        )
        # Final target: BUY=1, SELL=0, neither=-1 (ambiguous candles filtered)
        df['Hedef'] = np.where(buy_target == 1, 1, np.where(sell_target == 1, 0, -1))
        df = df[df['Hedef'] != -1].copy()  # ambiguous candles pollute the model, remove them
        
        features = [
            'RSI', 'Price_to_EMA20', 'ATR', 'Upper_Shadow', 'Lower_Shadow', 
            'Volume_Shock', 'MACD_Hist', 'BB_Width', 'Price_to_BB', 'Trend_Slope'
        ]
        
        feature_names_tr = {
            'RSI': 'RSI (Extremity)', 'Price_to_EMA20': 'Trend Distance', 'ATR': 'Volatility',
            'Upper_Shadow': 'Upper Shadow', 'Lower_Shadow': 'Alt G•lge', 'Volume_Shock': 'Volume Shock',
            'MACD_Hist': 'MACD Momentum', 'BB_Width': 'Bollinger Squeeze', 'Price_to_BB': 'Bant Konumu',
            'Trend_Slope': 'Trend Slope'
        }
        
        son_20_mum = df.tail(20)
        doji_olanlar = son_20_mum[son_20_mum['Doji'] == True]
        
        if interval == "1h":
            min_mum, max_mum = 4, 10  
        elif interval == "4h":
            min_mum, max_mum = 1, 3   
        else: 
            min_mum, max_mum = 1, 3   
            
        olgun_dojiler = []
        if not doji_olanlar.empty:
            for idx in doji_olanlar.index:
                mum_yasi = len(df) - 1 - df.index.get_loc(idx)
                if min_mum <= mum_yasi <= max_mum: 
                    olgun_dojiler.append(mum_yasi)
                    
        is_forced = "force_past" in st.session_state and st.session_state.force_past
        
        if not olgun_dojiler and not is_forced: 
            return None 
            
        if olgun_dojiler:
            gecen_mum = min(olgun_dojiler) 
        else:
            gecen_mum = 0 
            
        X = df[features].iloc[:-int(min_hours)]
        y = df['Hedef'].iloc[:-int(min_hours)]
        win_rate, toplam_sinyal = 50, 0
        en_etkili_faktorler = {}
        
        min_required_len = 15 if is_forced else 30
        
        if len(X) > min_required_len:
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=2,             
                learning_rate=0.05,
                subsample=0.6,           
                colsample_bytree=0.6,    
                reg_lambda=5.0,          
                reg_alpha=2.0,           
                random_state=42,
                eval_metric='logloss',
                n_jobs=-1
            )
            
            split_idx = int(len(X) * 0.8)
            X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
            
            model.fit(X_train, y_train)
            
            doji_test_df = df.iloc[split_idx:][df.iloc[split_idx:]['Doji'] == True].iloc[:-int(min_hours)]
            
            if not doji_test_df.empty:
                X_test_doji = doji_test_df[features]
                y_test_doji = doji_test_df['Hedef']
                preds = model.predict(X_test_doji)
                toplam_sinyal = len(y_test_doji)
                basarili_tahmin = np.sum(preds == y_test_doji.values)
                if toplam_sinyal > 0: 
                    win_rate = int((basarili_tahmin / toplam_sinyal) * 100)
            
            son_veri = df[features].iloc[[-1]]
            tahmin_yon = model.predict(son_veri)[0]
            guven_orani = int(max(model.predict_proba(son_veri)[0]) * 100)
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                sirali = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
                for feat, imp in sirali[:3]:
                    if imp > 0.01:
                        en_etkili_faktorler[feature_names_tr.get(feat, feat)] = float(imp * 100)
        else:
            tahmin_yon = 1 if df['RSI'].iloc[-1] < 50 else 0
            guven_orani = 55
            
        signal = "BUY" if tahmin_yon == 1 else "SELL"
        rsi_val = int(df['RSI'].iloc[-1])
        # Doji type is determined by shadow lengths (not RSI)
        upper_shadow_val = float(df['Upper_Shadow'].iloc[-1])
        lower_shadow_val = float(df['Lower_Shadow'].iloc[-1])
        if upper_shadow_val > 0.6 and lower_shadow_val < 0.2:
            doji_type = "Gravestone"  # Long upper shadow, short lower shadow
        elif lower_shadow_val > 0.6 and upper_shadow_val < 0.2:
            doji_type = "Dragonfly"   # Long lower shadow, short upper shadow
        else:
            doji_type = "Standard"    # Balanced shadows
        big_trend = buyuk_trend_kontrol(market["symbol"])
        
        # Change lookback: 1h->daily(24 candles), 4h->weekly(30 candles), 1d->weekly(5 candles)
        _lookback = {"1h": 24, "4h": 30, "1d": 5}.get(interval, 12)
        _lookback = min(_lookback, len(df) - 1)
        
        # --- BEARISH REBOUND TRAP SCORE ---
        # For SELL signals: how much did price bounce UP after the bearish doji•
        # The higher the bounce, the stronger the short opportunity (liquidity grab).
        rebound_pct = 0.0
        if signal == "SELL" and gecen_mum > 0:
            doji_idx = len(df) - 1 - gecen_mum          # index of the doji candle
            doji_close = float(df['Close'].iloc[doji_idx])
            peak_after = float(df['High'].iloc[doji_idx + 1:].max())  # highest wick after doji
            if doji_close > 0:
                rebound_pct = round(((peak_after - doji_close) / doji_close) * 100, 2)
        
        return {
            "hoursAgo": gecen_mum, "signal": signal, "rsi": rsi_val, "confidence": guven_orani,
            "winRate": win_rate, "totalSignals": toplam_sinyal, "bigTrend": big_trend,
            "price": float(df['Close'].iloc[-1]),
            "change": float(((df['Close'].iloc[-1] - df['Open'].iloc[-_lookback]) / (df['Open'].iloc[-_lookback] + 1e-10)) * 100),
            "dojiType": doji_type, "topFeatures": en_etkili_faktorler,
            "reboundPct": rebound_pct
        }
    except Exception as e:
        return None

# --- STATE TANIMLAMALARI ---
if "results" not in st.session_state: st.session_state.results = {}
if "chart_open" not in st.session_state: st.session_state.chart_open = None
if "force_past" not in st.session_state: st.session_state.force_past = False

# Style Definitions
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2•family=Inter:wght@400;600;700;800&display=swap');
    html, body, [data-testid="stAppViewContainer"] { background-color: #020817 !important; color: #F1F5F9 !important; font-family: 'Inter', sans-serif !important; }
    .stButton>button { background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important; color: white !important; border: none !important; border-radius: 8px !important; font-weight: 700 !important; }
    [data-testid="stSidebar"] { background-color: #0F172A !important; border-right: 1px solid #1E293B !important; }
</style>
""", unsafe_allow_html=True)

# --- 📊 LEFT MENU NAVIGATION (SIDEBAR) ---
st.sidebar.markdown("""
<div style='text-align: center; padding: 10px; border-bottom: 1px solid #1E293B; margin-bottom: 20px;'>
    <h3 style='color: #FFF; margin: 0; font-size: 16px;'>📊 AI TERMINAL v5 (Pro)</h3>
</div>
""", unsafe_allow_html=True)

secilen_sayfa = st.sidebar.radio(
    "📊 TRADING ROOMS",
    ["📊 General Dashboard", "📊 Forex Terminal", "📊 Crypto Terminal", "📊📊 NASDAQ Terminal", "📊 Commodity Terminali"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊• Global Filters")
global_interval = st.sidebar.selectbox("• Time Frame (Period)", ["1h", "4h", "1d"], index=0)
global_min_hours = st.sidebar.slider("📊 AI Forecast Window (Forward Candles)", 1, 12, 4)
global_doji_modu = st.sidebar.radio("📊 Doji Sensitivity Mode", ["Dynamic (Auto-Adaptive)", "Fixed (Classic 0.3)"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("📊• System Test Mode")
st.session_state.force_past = st.sidebar.checkbox(
    "📊 Remove Time Filter", 
    value=st.session_state.force_past,
    help="Use to force-display the most recent candle when no fresh Doji exists."
)

st.markdown(f"""
<div style="background: linear-gradient(180deg, #0F172A 0%, #020817 100%); border-bottom: 1px solid #1E293B; padding: 15px; margin-bottom: 15px; border-radius: 8px;">
    <h1 style="margin: 0; font-size: 22px; font-weight: 800; color: #FFF;">📊 Joe Barbarov Doji Signal</h1>
    <p style="margin: 0; font-size: 12px; color: #64748B;">Current Room: <b>{secilen_sayfa}</b> • XGBoost + Feature Engineering Active</p>
</div>
""", unsafe_allow_html=True)

# --- PANEL CONTENT AND VISUAL CARDS ---
aktif_list = [] # SAFETY NET: System won't crash even if no room matches!

if secilen_sayfa == "📊 General Dashboard":
    with st.spinner("Querying all market dynamics..."):
        c_val, c_status, c_color = get_crypto_fng()
        n_vol, n_vol_clr, n_hac = get_real_market_dynamics(["AAPL", "TSLA", "NVDA", "MSFT"])
        e_vol, e_vol_clr, e_hac = get_real_market_dynamics(["GC=F", "SI=F"])
        c_vol = "High 📊" if c_val > 65 else ("Low 📊" if c_val < 35 else "Normal 📊")
        c_vol_clr = "#34D399" if c_val > 65 else ("#64748B" if c_val < 35 else "#F59E0B")
        c_hac = "Strong 📊" if c_val > 55 else "Weak 📊"
        n_bar_color = "#EF4444" if "Closed" in n_hac else ("#10B981" if "Strong" in n_hac else "#94A3B8")
        e_bar_color = "#EF4444" if "Closed" in e_hac else ("#10B981" if "Strong" in e_hac else "#94A3B8")
        
        from datetime import datetime, timezone
        gun = datetime.now(timezone.utc).weekday()
        p_durum = "Weekend Closed 📊" if gun >= 5 else "Open / Trading 📊"

    fng_cols = st.columns(3)
    with fng_cols[0]:
        html_c = """<div style="background:#0F172A; border:1px solid #1E293B; padding:12px; border-radius:8px; min-height:110px;">
            <div style="font-size:11px; font-weight:700; color:#64748B; margin-bottom:6px;">📊 KR•PTO P•YASASI (BTC/ETH)</div>
            <div style="background:#1E293B; height:6px; border-radius:3px; overflow:hidden; margin-bottom:8px;"><div style="background:{clr}; width:{val}%; height:6px;"></div></div>
            <div style="color:{clr}; font-weight:800; font-size:13px; text-align:right; margin-bottom:6px;">{stat} ({val}/100)</div>
            <div style="display:flex; justify-content:space-between; font-size:10px; color:#64748B; border-top:1px solid rgba(51,65,85,0.3); padding-top:4px;"><span>• Vol: <b style="color:{v_clr};">{vol}</b></span><span>📊 Hacim: <b style="color:#FFF;">{hac}</b></span></div>
        </div>""".format(clr=c_color, val=c_val, stat=c_status, v_clr=c_vol_clr, vol=c_vol, hac=c_hac)
        st.markdown(html_c, unsafe_allow_html=True)
        
    with fng_cols[1]:
        n_fng_val, n_fng_stat, n_fng_clr = get_nasdaq_fng()
        p_durum = "Open 📊" if datetime.now(timezone.utc).weekday() < 5 else "Closed 📊"
        
        html_n = """<div style="background:#0F172A; border:1px solid #1E293B; padding:12px; border-radius:8px; min-height:110px;">
            <div style="font-size:11px; font-weight:700; color:#64748B; margin-bottom:6px;">📊📊 ABD BORSALARI (NASDAQ)</div>
            <div style="background:#1E293B; height:6px; border-radius:3px; overflow:hidden; margin-bottom:8px;"><div style="background:{clr}; width:{val}%; height:6px;"></div></div>
            <div style="color:{clr}; font-weight:800; font-size:13px; text-align:right; margin-bottom:6px;">{stat} ({val}/100)</div>
            <div style="display:flex; justify-content:space-between; font-size:10px; color:#64748B; border-top:1px solid rgba(51,65,85,0.3); padding-top:4px;">
                <span>• Vol: <b style="color:{v_clr};">{vol}</b></span>
                <span>📊 Hacim: <b style="color:#FFF;">{hac}</b></span>
                <span>📊 <b style="color:#FFF;">{durum}</b></span>
            </div>
        </div>""".format(clr=n_fng_clr, val=n_fng_val, stat=n_fng_stat, v_clr=n_vol_clr, vol=n_vol, hac=n_hac, durum=p_durum)
        st.markdown(html_n, unsafe_allow_html=True)
        
    with fng_cols[2]:
        html_e = """<div style="background:#0F172A; border:1px solid #1E293B; padding:12px; border-radius:8px; min-height:110px;">
            <div style="font-size:11px; font-weight:700; color:#64748B; margin-bottom:6px;">📊 EMT•A P•YASASI (ALTIN/G•M📊)</div>
            <div style="background:#1E293B; height:6px; border-radius:3px; overflow:hidden; margin-bottom:8px;"><div style="background:{b_clr}; width:100%; height:6px;"></div></div>
            <div style="color:{b_clr}; font-weight:800; font-size:13px; text-align:right; margin-bottom:6px;">Piyasa: {durum}</div>
            <div style="display:flex; justify-content:space-between; font-size:10px; color:#64748B; border-top:1px solid rgba(51,65,85,0.3); padding-top:4px;"><span>• Vol: <b style="color:{v_clr};">{vol}</b></span><span>📊 Hacim: <b style="color:#FFF;">{hac}</b></span></div>
        </div>""".format(b_clr=e_bar_color, hac=e_hac, v_clr=e_vol_clr, vol=e_vol, durum=p_durum)
        st.markdown(html_e, unsafe_allow_html=True)
        
    aktif_list = MARKETS
    st.markdown("<h3 style='color: #F1F5F9; font-size: 16px; margin-top: 15px; margin-bottom: 10px;'>📊• Live• Market Regime (Heatmap)</h3>", unsafe_allow_html=True)
    with st.spinner("Processing heatmap data..."):
        heatmap_html = "<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-bottom: 25px;'>"
        for m in MARKETS:
            rejim, renk = piyasa_rejimi_hesapla(m["symbol"])
            heatmap_html += f"<div style='background: {renk}; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 4px 6px rgba(0,0,0,0.1);'><div style='color: rgba(255,255,255,0.9); font-size: 11px; font-weight: 700; margin-bottom: 4px;'>{m['name']}</div><div style='color: #FFF; font-size: 13px; font-weight: 800;'>{rejim}</div></div>"
        heatmap_html += "</div>"
        st.markdown(heatmap_html, unsafe_allow_html=True)

elif secilen_sayfa == "📊 Forex Terminal":
    with st.spinner("Analyzing Forex data..."):
        f_vol, f_vol_clr, f_hac = get_real_market_dynamics(["EURUSD=X"])
        
        # --- FOREX VOLUME FIX ---
        # E•er hacim verisi Forex do•as• gere•i bo• gelirse ekrandaki yaz•y• ve rengi g•zelle•tir
        if "No Data" in f_hac:
            f_hac = "Decentralized Volume 📊"
            f_bar_color = "#3B82F6" # 📊k bir Forex mavisi
        else:
            f_bar_color = "#EF4444" if "Closed" in f_hac else ("#10B981" if "Strong" in f_hac else "#94A3B8")
            
        p_durum = dinamik_piyasa_durumu()
        
    html_single_f = """<div style="background:#0F172A; border:1px solid #1E293B; padding:15px; border-radius:8px; margin-bottom:20px;">
        <div style="font-size:12px; font-weight:700; color:#64748B; margin-bottom:6px;">📊 K•RESEL D•V•Z P•YASASI (FOREX)</div>
        <div style="background:#1E293B; height:8px; border-radius:4px; overflow:hidden; margin-bottom:10px;"><div style="background:{b_clr}; width:100%; height:8px;"></div></div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="font-size:12px; color:#94A3B8;">• Volatility (ATR): <b style="color:{v_clr};">{vol}</b> • 📊 Durum: <b style="color:#FFF;">{durum}</b></div>
            <div style="color:{b_clr}; font-weight:800; font-size:14px;">{hac}</div>
        </div>
    </div>""".format(b_clr=f_bar_color, v_clr=f_vol_clr, vol=f_vol, hac=f_hac, durum=p_durum)
    st.markdown(html_single_f, unsafe_allow_html=True)
    
    aktif_list = [m for m in MARKETS if m["category"] == "Forex"]
        
elif secilen_sayfa == "📊 Crypto Terminal":
    with st.spinner("Querying crypto sentiment..."):
        c_val, c_status, c_color = get_crypto_fng()
        c_vol = "High 📊" if c_val > 65 else ("Low 📊" if c_val < 35 else "Normal 📊")
        c_vol_clr = "#34D399" if c_val > 65 else ("#64748B" if c_val < 35 else "#F59E0B")
        c_hac = "Strong 📊" if c_val > 55 else "Weak 📊"
        
    html_single_c = """<div style="background:#0F172A; border:1px solid #1E293B; padding:15px; border-radius:8px; margin-bottom:20px;">
        <div style="font-size:12px; font-weight:700; color:#64748B; margin-bottom:6px;">📊 CANLI KR•PTO DUYARLILI•I VE ANAL•Z• (BTC/ETH)</div>
        <div style="background:#1E293B; height:8px; border-radius:4px; overflow:hidden; margin-bottom:10px;"><div style="background:{clr}; width:{val}%; height:8px;"></div></div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="font-size:12px; color:#94A3B8;">• Volatility: <b style="color:{v_clr};">{vol}</b> • 📊 Ger•ek Hacim: <b style="color:#FFF;">{hac}</b></div>
            <div style="color:{clr}; font-weight:800; font-size:15px;">{stat} ({val}/100)</div>
        </div>
    </div>""".format(clr=c_color, val=c_val, v_clr=c_vol_clr, vol=c_vol, hac=c_hac, stat=c_status)
    st.markdown(html_single_c, unsafe_allow_html=True)
    
    aktif_list = [m for m in MARKETS if m["category"] == "Kripto"]

elif secilen_sayfa == "📊📊 NASDAQ Terminal":
    with st.spinner("Calculating NASDAQ dynamics..."):
        n_vol, n_vol_clr, n_hac = get_real_market_dynamics(["AAPL", "TSLA", "NVDA", "MSFT"])
        n_bar_color = "#EF4444" if "Closed" in n_hac else ("#10B981" if "Strong" in n_hac else "#94A3B8")
        p_durum = dinamik_piyasa_durumu()
        
    html_single_n = """<div style="background:#0F172A; border:1px solid #1E293B; padding:15px; border-radius:8px; margin-bottom:20px;">
        <div style="font-size:12px; font-weight:700; color:#64748B; margin-bottom:6px;">📊📊 ABD TEKNOLOJ• BORSASI D•NAM•KLER• (NASDAQ)</div>
        <div style="background:#1E293B; height:8px; border-radius:4px; overflow:hidden; margin-bottom:10px;"><div style="background:{b_clr}; width:100%; height:8px;"></div></div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="font-size:12px; color:#94A3B8;">• Volatility (ATR Ratio•): <b style="color:{v_clr};">{vol}</b> • 📊 Durum: <b style="color:#FFF;">{durum}</b></div>
            <div style="color:{b_clr}; font-weight:800; font-size:15px;">{hac}</div>
        </div>
    </div>""".format(b_clr=n_bar_color, v_clr=n_vol_clr, vol=n_vol, hac=n_hac, durum=p_durum)
    st.markdown(html_single_n, unsafe_allow_html=True)
    
    aktif_list = [m for m in MARKETS if m["category"] == "NASDAQ"]

elif secilen_sayfa == "📊 Commodity Terminali":
    with st.spinner("Commodity verileri analiz ediliyor..."):
        e_vol, e_vol_clr, e_hac = get_real_market_dynamics(["GC=F", "SI=F"])
        e_bar_color = "#EF4444" if "Closed" in e_hac else ("#10B981" if "Strong" in e_hac else "#94A3B8")
        p_durum = dinamik_piyasa_durumu()
        
    html_single_e = """<div style="background:#0F172A; border:1px solid #1E293B; padding:15px; border-radius:8px; margin-bottom:20px;">
        <div style="font-size:12px; font-weight:700; color:#64748B; margin-bottom:6px;">📊 DE•ERL• METAL P•YASA PS•KOLOJ•S• (ALTIN/G•M📊)</div>
        <div style="background:#1E293B; height:8px; border-radius:4px; overflow:hidden; margin-bottom:10px;"><div style="background:{b_clr}; width:100%; height:8px;"></div></div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="font-size:12px; color:#94A3B8;">• Volatility (ATR): <b style="color:{v_clr};">{vol}</b> • 📊 Durum: <b style="color:#FFF;">{durum}</b></div>
            <div style="color:{b_clr}; font-weight:800; font-size:15px;">{hac}</div>
        </div>
    </div>""".format(b_clr=e_bar_color, v_clr=e_vol_clr, vol=e_vol, hac=e_hac, durum=p_durum)
    st.markdown(html_single_e, unsafe_allow_html=True)
    
    aktif_list = [m for m in MARKETS if m["category"] == "Commodity"]

# --- TRADINGVIEW MODAL MOTORU ---
if st.session_state.chart_open:
    c_market = st.session_state.chart_open
    st.markdown("### 📊 Live Chart: {} ({})".format(c_market['name'], c_market['tv']))
    tv_interval_map = {"1h": "60", "4h": "240", "1d": "D"}
    secilen_tv_interval = tv_interval_map.get(global_interval, "60")
    html_code = f"""<div id="tv-chart-container" style="height:450px;"></div><script src="https://s3.tradingview.com/tv.js"></script><script>new TradingView.widget({{"autosize":true,"symbol":"{c_market['tv']}","interval":"{secilen_tv_interval}","timezone":"Europe/Istanbul","theme":"dark","style":"1","locale":"tr","container_id":"tv-chart-container","studies":["RSI@tv-basicstudies","MAExp@tv-basicstudies","MACD@tv-basicstudies","BB@tv-basicstudies"],"disabled_features":["header_saveload"]}});</script>"""
    st.components.v1.html(html_code, height=460)
    if st.button("• Grafi•i Kapat"):
        st.session_state.chart_open = None
        st.rerun()
    st.markdown("---")

# --- GEL📊M📊 TARAMA BUTONU VE •LERLEME •UBU•U (ASYNC ENGINE) ---
if st.button("📊 {} 📊in Live• AI Taramas• Ba•lat".format(secilen_sayfa.split()[1])):
    st.info("• Async (Parallel) scan started, markets processing simultaneously...")
    ilerleme_cubugu = st.progress(0)
    durum_metni = st.empty()
    
    yeni_sonuclar = {}
    toplam_varlik = len(aktif_list)
    tamamlanan = 0
    
    # •oklu i•lem i•in yard•mc• fonksiyon
    def piyasa_isle(m):
        return m, analiz_et_safe(m, global_min_hours, global_interval, global_doji_modu)

    # Ayn• anda maksimum 10 i•lem (i• par•ac📊•) •al📊t•r
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # T•m g•revleri havuza at ve ayn• anda ba•lat
        gelecek_gorevler = [executor.submit(piyasa_isle, m) for m in aktif_list]
        
        # G•revler tamamland•k•a (asenkron olarak) sonu•lar• topla
        for future in concurrent.futures.as_completed(gelecek_gorevler):
            m_data, analiz_sonucu = future.result()
            
            if analiz_sonucu: 
                yeni_sonuclar[m_data["symbol"]] = {"market": m_data, "result": analiz_sonucu}
                
            tamamlanan += 1
            durum_metni.markdown(f"**• Processing in Parallel:** {tamamlanan}/{toplam_varlik} markets completed.")
            ilerleme_cubugu.progress(tamamlanan / toplam_varlik)
            
    durum_metni.success("• Async scan completed at lightning speed!")
    st.session_state.results = yeni_sonuclar
    st.rerun()

# --- RENDERING SIGNAL CARDS ---
valid_signals = dict(
    sorted(
        ((k, v) for k, v in st.session_state.results.items() if v["market"] in aktif_list),
        key=lambda x: x[1]["result"].get("winRate", 0),
        reverse=True
    )
)

if not valid_signals:
    st.info("Bu odada şu an uygun bir Doji sinyali bulunmuyor.")
else:
    for sym, data in valid_signals.items():
        m, r = data["market"], data["result"]
        is_buy = r["signal"] == "BUY"
        
        # Trend Uyum Mantığı
        is_confluence = (is_buy and r["bigTrend"] == "Bull (Up)") or (not is_buy and r["bigTrend"] == "Bear (Down)")
        confluence_text = "🔥 Trend Aligned" if is_confluence else "⚠️ Counter-Trend Risk"
        
        with st.container(border=True):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                # Sinyal Başlığı
                st.subheader(f"{m['name']}  :{'green' if is_buy else 'red'}[{r['signal']}]")
                st.write(f"⏳ **Time since Doji:** {r['hoursAgo']} Candle(s) ({r['dojiType']} Doji)")
                st.caption(f"**Trend (4h):** {r['bigTrend']} | {confluence_text}")
                
                # --- BEARISH REBOUND TRAP (Native Streamlit ile) ---
                if not is_buy:
                    rb = r.get("reboundPct", 0.0)
                    if rb >= 2.0:
                        st.error(f"🚨 Extreme Rebound Trap (+{rb:.2f}%) - Güçlü Satış Fırsatı")
                    elif rb >= 1.0:
                        st.warning(f"🔥 Strong Rebound Trap (+{rb:.2f}%)")
                    elif rb >= 0.4:
                        st.info(f"⚠️ Mild Rebound (+{rb:.2f}%)")
                    elif rb > 0:
                        st.success(f"💤 Low Rebound (+{rb:.2f}%)")
                
                # Karar Etkenleri (Feature Importances)
                if "topFeatures" in r and r["topFeatures"]:
                    feat_str = " • ".join([f"{k}: %{v:.1f}" for k, v in r["topFeatures"].items()])
                    st.caption(f"🧠 **Decision Factors:** {feat_str}")

            with col2:
                st.metric("Forecast Confidence", f"%{int(r['confidence'])}")
                st.metric("Historical Win-Rate", f"%{int(r['winRate'])}")
                
            with col3:
                st.metric("Current Price", f"${r['price']:,.2f}", f"{r['change']:.2f}%")
                
            if st.button(f"📊 {m['name']} Chart View", key=f"chart_btn_{m['symbol']}_{m['category']}"):
                st.session_state.chart_open = m
                st.rerun()
