# -*- coding: utf-8 -*-
"""
AI Doji Terminali - XGBoost Edition
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone
import xgboost as xgb

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="AI Doji Terminali", layout="wide", initial_sidebar_state="auto")

# --- GLOBAL PİYASA TANIMLARI ---
MARKETS = [
    {"name": "Altın (XAU/USD)", "symbol": "GC=F", "tv": "OANDA:XAUUSD", "category": "Emtia", "color": "#F59E0B"},
    {"name": "Gümüş (XAG/USD)", "symbol": "SI=F", "tv": "OANDA:XAGUSD", "category": "Emtia", "color": "#94A3B8"},
    {"name": "Apple", "symbol": "AAPL", "tv": "NASDAQ:AAPL", "category": "NASDAQ", "color": "#60A5FA"},
    {"name": "Tesla", "symbol": "TSLA", "tv": "NASDAQ:TSLA", "category": "NASDAQ", "color": "#EF4444"},
    {"name": "NVIDIA", "symbol": "NVDA", "tv": "NASDAQ:NVDA", "category": "NASDAQ", "color": "#10B981"},
    {"name": "Microsoft", "symbol": "MSFT", "tv": "NASDAQ:MSFT", "category": "NASDAQ", "color": "#3B82F6"},
    {"name": "Bitcoin", "symbol": "BTC-USD", "tv": "BINANCE:BTCUSDT", "category": "Kripto", "color": "#F59E0B"},
    {"name": "Ethereum", "symbol": "ETH-USD", "tv": "BINANCE:ETHUSDT", "category": "Kripto", "color": "#6366F1"}
]

# --- CANLI VERİ FONKSİYONLARI ---
def get_crypto_fng():
    try:
        response = requests.get("https://api.alternative.me/fng/", timeout=5).json()
        value = int(response['data'][0]['value'])
        classification = response['data'][0]['value_classification']
        tr_map = {
            "Extreme Fear": ("Aşırı Korku 😱", "#EF4444"),
            "Fear": ("Korku 😨", "#F97316"),
            "Neutral": ("Nötr 😐", "#94A3B8"),
            "Greed": ("Açgözlülük 🤑", "#10B981"),
            "Extreme Greed": ("Aşırı Açgözlülük 🚀", "#34D399")
        }
        status, color = tr_map.get(classification, (classification, "#94A3B8"))
        return value, status, color
    except:
        return 50, "Nötr 😐", "#94A3B8"

def get_real_market_dynamics(symbols):
    try:
        vol_results = []
        vol_ratios = []
        vol_counts = 0
        for sym in symbols:
            df = yf.download(sym, period="30d", interval="1d", progress=False)
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
            if 'Volume' in df.columns and df['Volume'].iloc[-1] > 0:
                son_hacim = df['Volume'].rolling(3).mean().iloc[-1]
                ort_hacim = df['Volume'].rolling(20).mean().iloc[-1]
                hacim_soku = son_hacim / (ort_hacim + 1e-10)
                vol_results.append(hacim_soku)
                vol_counts += 1
        final_vol, final_vol_clr, final_hac = "Düşük 💤", "#64748B", "Piyasa Kapalı 🔒"
        if vol_ratios:
            avg_vol_ratio = sum(vol_ratios) / len(vol_ratios)
            if avg_vol_ratio > 2.0: final_vol, final_vol_clr = "Yüksek 🔥", "#34D399"
            elif avg_vol_ratio > 1.0: final_vol, final_vol_clr = "Normal 📊", "#F59E0B"
        if vol_counts > 0 and sum(vol_results) > 0:
            avg_hacim_soku = sum(vol_results) / vol_counts
            if avg_hacim_soku > 1.1: final_hac = "Güçlü 💰"
            elif avg_hacim_soku > 0.4: final_hac = "Normal 📈"
            else: final_hac = "Zayıf 📉"
        return final_vol, final_vol_clr, final_hac
    except:
        return "Normal 📊", "#F59E0B", "Normal 📈"

# --- PİYASA REJİMİ (HEATMAP) HESAPLAYICI ---
def piyasa_rejimi_hesapla(symbol):
    try:
        df = yf.download(symbol, period="60d", interval="1d", progress=False)
        if df.empty: return "Veri Yok", "#334155"
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # EMA ve RSI Hesaplama
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs)).iloc[-1]

        ema20 = df['EMA20'].iloc[-1]
        ema50 = df['EMA50'].iloc[-1]

        # Rejim Sınıflandırması
        if ema20 > ema50 and rsi > 60: return "Güçlü Boğa 🚀", "#10B981"
        elif ema20 > ema50: return "Boğa 🟢", "#059669"
        elif ema20 < ema50 and rsi < 40: return "Güçlü Ayı 🩸", "#EF4444"
        elif ema20 < ema50: return "Ayı 🔴", "#B91C1C"
        else: return "Yatay ⚪", "#64748B"
    except:
        return "Bilinmiyor", "#334155"

def buyuk_trend_kontrol(symbol):
    try:
        df_big = yf.download(symbol, period="60d", interval="4h", progress=False)
        if df_big.empty: return "Yansız"
        if isinstance(df_big.columns, pd.MultiIndex):
            df_big.columns = df_big.columns.get_level_values(0)
        ema200 = df_big['Close'].ewm(span=200, adjust=False).mean().iloc[-1]
        son_fiyat = df_big['Close'].iloc[-1]
        return "Boğa (Yukarı)" if son_fiyat > ema200 else "Ayı (Aşağı)"
    except:
        return "Yansız"

def analiz_et_safe(market, min_hours, interval):
    try:
        if interval == "1h": periyot = "1y"
        elif interval == "4h": periyot = "2y"
        else: periyot = "5y"
        df = yf.download(market["symbol"], period=periyot, interval=interval, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        govde = abs(df['Open'] - df['Close'])
        toplam_boy = df['High'] - df['Low']
        df['Doji'] = govde <= (toplam_boy * 0.1)
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
        df = df.dropna()
        df['Hedef'] = np.where(df['Close'].shift(-int(min_hours)) > df['Close'], 1, 0)
        özellikler = ['RSI', 'Price_to_EMA20', 'ATR', 'Upper_Shadow', 'Lower_Shadow', 'Volume_Shock']
        
        doji_satirlari = df[df['Doji'] == True]
        if doji_satirlari.empty: return None
        
        son_doji_zaman = doji_satirlari.index[-1].to_pydatetime().replace(tzinfo=timezone.utc)
        su_an = datetime.now(timezone.utc)
        gecen_saat = round((su_an - son_doji_zaman).total_seconds() / 3600, 1)
        saat_katsayisi = 4 if interval == "4h" else (24 if interval == "1d" else 1)
        gecen_mum = round(gecen_saat / saat_katsayisi, 1)
        
        is_forced = "force_past" in st.session_state and st.session_state.force_past
        
        if not is_forced:
            if gecen_mum < min_hours or gecen_mum > (24 / saat_katsayisi): 
                return None
            
        X = df[özellikler].iloc[:-int(min_hours)]
        y = df['Hedef'].iloc[:-int(min_hours)]
        win_rate, toplam_sinyal = 50, 0
        
        min_required_len = 15 if is_forced else 50
        
        if len(X) > min_required_len:
            # --- Hızlandırılmış XGBoost Model Tanımlaması ---
            model = xgb.XGBClassifier(
                n_estimators=80,         # 150 yerine 80 ağaç yeterlidir
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                n_jobs=-1                # SİHİRLİ DOKUNUŞ: Sunucunun tüm işlemci çekirdeklerini aynı anda kullanır
            )
            
            model.fit(X, y)
            doji_df = df[df['Doji'] == True].iloc[:-int(min_hours)]
            if not doji_df.empty:
                X_doji = doji_df[özellikler]
                y_doji = doji_df['Hedef']
                preds = model.predict(X_doji)
                toplam_sinyal = len(y_doji)
                basarili_tahmin = np.sum(preds == y_doji.values)
                if toplam_sinyal > 0: win_rate = int((basarili_tahmin / toplam_sinyal) * 100)
            son_veri = df[özellikler].iloc[[-1]]
            tahmin_yon = model.predict(son_veri)[0]
            guven_orani = int(max(model.predict_proba(son_veri)[0]) * 100)
        else:
            tahmin_yon = 1 if df['RSI'].iloc[-1] < 50 else 0
            guven_orani = 55
            
        signal = "BUY" if tahmin_yon == 1 else "SELL"
        rsi_val = int(df['RSI'].iloc[-1])
        if rsi_val < 40: doji_type = "Dragonfly"
        elif rsi_val > 60: doji_type = "Gravestone"
        else: doji_type = "Standard"
        big_trend = buyuk_trend_kontrol(market["symbol"])
        
        return {
            "hoursAgo": gecen_mum, "signal": signal, "rsi": rsi_val, "confidence": guven_orani,
            "winRate": win_rate, "totalSignals": toplam_sinyal, "bigTrend": big_trend,
            "price": float(df['Close'].iloc[-1]),
            "change": float(((df['Close'].iloc[-1] - df['Open'].iloc[-12]) / df['Open'].iloc[-12]) * 100),
            "dojiType": doji_type
        }
    except:
        return None

# --- STATE TANIMLAMALARI ---
if "results" not in st.session_state: st.session_state.results = {}
if "chart_open" not in st.session_state: st.session_state.chart_open = None
if "force_past" not in st.session_state: st.session_state.force_past = False

# Stil Tanımlamaları
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    html, body, [data-testid="stAppViewContainer"] { background-color: #020817 !important; color: #F1F5F9 !important; font-family: 'Inter', sans-serif !important; }
    .stButton>button { background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important; color: white !important; border: none !important; border-radius: 8px !important; font-weight: 700 !important; }
    [data-testid="stSidebar"] { background-color: #0F172A !important; border-right: 1px solid #1E293B !important; }
</style>
""", unsafe_allow_html=True)

# --- 🌐 SOL MENÜ NAVİGASYONU (SIDEBAR) ---
st.sidebar.markdown("""
<div style='text-align: center; padding: 10px; border-bottom: 1px solid #1E293B; margin-bottom: 20px;'>
    <h3 style='color: #FFF; margin: 0; font-size: 16px;'>🌐 AI TERMINAL v4 (XGBoost)</h3>
</div>
""", unsafe_allow_html=True)

secilen_sayfa = st.sidebar.radio(
    "📊 İŞLEM ODALARI",
    ["🏠 Genel Dashboard", "🪙 Kripto Terminali", "🇺🇸 NASDAQ Terminali", "👑 Emtia Terminali"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Küresel Filtreler")
global_interval = st.sidebar.selectbox("⏳ Zaman Dilimi (Periyot)", ["1h", "4h", "1d"], index=0)
global_min_hours = st.sidebar.slider("🎯 AI Tahmin Süresi (Mum)", 1, 12, 4)

st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Sistem Test Modu")
st.session_state.force_past = st.sidebar.checkbox(
    "🔓 Zaman Filtresini Kaldır", 
    value=st.session_state.force_past,
    help="Piyasada taze Doji olmadığında geçmişteki en son mumu zorla listelemek için kullan."
)

# --- PANEL BAŞLIĞI ---
st.markdown(f"""
<div style="background: linear-gradient(180deg, #0F172A 0%, #020817 100%); border-bottom: 1px solid #1E293B; padding: 15px; margin-bottom: 15px; border-radius: 8px;">
    <h1 style="margin: 0; font-size: 22px; font-weight: 800; color: #FFF;">🤖 AI Doji Sinyal Paneli</h1>
    <p style="margin: 0; font-size: 12px; color: #64748B;">Mevcut Oda: <b>{secilen_sayfa}</b> • XGBoost Motoru Aktif</p>
</div>
""", unsafe_allow_html=True)


# --- 🎛️ SEKME BAZLI DİNAMİK BAR ENJEKSİYONU 🎛️ ---

if secilen_sayfa == "🏠 Genel Dashboard":
    with st.spinner("Tüm piyasa dinamikleri sorgulanıyor..."):
        c_val, c_status, c_color = get_crypto_fng()
        n_vol, n_vol_clr, n_hac = get_real_market_dynamics(["AAPL", "TSLA", "NVDA", "MSFT"])
        e_vol, e_vol_clr, e_hac = get_real_market_dynamics(["GC=F", "SI=F"])
        c_vol = "Yüksek 🔥" if c_val > 65 else ("Düşük 💤" if c_val < 35 else "Normal 📊")
        c_vol_clr = "#34D399" if c_val > 65 else ("#64748B" if c_val < 35 else "#F59E0B")
        c_hac = "Güçlü 💰" if c_val > 55 else "Zayıf 📉"
        n_bar_color = "#EF4444" if "Kapalı" in n_hac else ("#10B981" if "Güçlü" in n_hac else "#94A3B8")
        e_bar_color = "#EF4444" if "Kapalı" in e_hac else ("#10B981" if "Güçlü" in e_hac else "#94A3B8")

    fng_cols = st.columns(3)
    with fng_cols[0]:
        html_c = """<div style="background:#0F172A; border:1px solid #1E293B; padding:12px; border-radius:8px; min-height:110px;">
            <div style="font-size:11px; font-weight:700; color:#64748B; margin-bottom:6px;">🪙 KRİPTO PİYASASI (BTC/ETH)</div>
            <div style="background:#1E293B; height:6px; border-radius:3px; overflow:hidden; margin-bottom:8px;"><div style="background:{clr}; width:{val}%; height:6px;"></div></div>
            <div style="color:{clr}; font-weight:800; font-size:13px; text-align:right; margin-bottom:6px;">{stat} ({val}/100)</div>
            <div style="display:flex; justify-content:space-between; font-size:10px; color:#64748B; border-top:1px solid rgba(51,65,85,0.3); padding-top:4px;"><span>⚡ Vol: <b style="color:{v_clr};">{vol}</b></span><span>💵 Hacim: <b style="color:#FFF;">{hac}</b></span></div>
        </div>""".format(clr=c_color, val=c_val, stat=c_status, v_clr=c_vol_clr, vol=c_vol, hac=c_hac)
        st.markdown(html_c, unsafe_allow_html=True)
        
    with fng_cols[1]:
        html_n = """<div style="background:#0F172A; border:1px solid #1E293B; padding:12px; border-radius:8px; min-height:110px;">
            <div style="font-size:11px; font-weight:700; color:#64748B; margin-bottom:6px;">🇺🇸 ABD BORSALARI (NASDAQ)</div>
            <div style="background:#1E293B; height:6px; border-radius:3px; overflow:hidden; margin-bottom:8px;"><div style="background:{b_clr}; width:100%; height:6px;"></div></div>
            <div style="color:{b_clr}; font-weight:800; font-size:13px; text-align:right; margin-bottom:6px;">{hac}</div>
            <div style="display:flex; justify-content:space-between; font-size:10px; color:#64748B; border-top:1px solid rgba(51,65,85,0.3); padding-top:4px;"><span>⚡ Vol: <b style="color:{v_clr};">{vol}</b></span><span>💵 Durum: <b style="color:#FFF;">Hafta Sonu</b></span></div>
        </div>""".format(b_clr=n_bar_color, hac=n_hac, v_clr=n_vol_clr, vol=n_vol)
        st.markdown(html_n, unsafe_allow_html=True)
        
    with fng_cols[2]:
        html_e = """<div style="background:#0F172A; border:1px solid #1E293B; padding:12px; border-radius:8px; min-height:110px;">
            <div style="font-size:11px; font-weight:700; color:#64748B; margin-bottom:6px;">👑 EMTİA PİYASASI (ALTIN/GÜMÜŞ)</div>
            <div style="background:#1E293B; height:6px; border-radius:3px; overflow:hidden; margin-bottom:8px;"><div style="background:{b_clr}; width:100%; height:6px;"></div></div>
            <div style="color:{b_clr}; font-weight:800; font-size:13px; text-align:right; margin-bottom:6px;">{hac}</div>
            <div style="display:flex; justify-content:space-between; font-size:10px; color:#64748B; border-top:1px solid rgba(51,65,85,0.3); padding-top:4px;"><span>⚡ Vol: <b style="color:{v_clr};">{vol}</b></span><span>💵 Durum: <b style="color:#FFF;">Hafta Sonu</b></span></div>
        </div>""".format(b_clr=e_bar_color, hac=e_hac, v_clr=e_vol_clr, vol=e_vol)
        st.markdown(html_e, unsafe_allow_html=True)
        
    aktif_list = MARKETS
    st.markdown("""<div style="background:rgba(30,41,59,0.5); border:1px dashed #334155; padding:12px; border-radius:8px; margin-bottom:20px;"><div style="display:flex; gap:20px; flex-wrap:wrap; font-size:11px; color:#94A3B8; line-height:1.4;"><div>🔴 <b style="color:#EF4444;">Aşırı Korku (0-30):</b> BUY yönlü dönüş şansı yüksek.</div><div>⚪ <b style="color:#94A3B8;">Nötr (45-55):</b> Doji daha stabil çalışır.</div><div>🟢 <b style="color:#34D399;">Aşırı Açgözlülük (75-100):</b> BUY sinyallerine temkinli yaklaşılmalıdır.</div></div></div>""", unsafe_allow_html=True)
   
    # --- 🗺️ CANLI PİYASA REJİMİ ISI HARİTASI ---
    st.markdown("<h3 style='color: #F1F5F9; font-size: 16px; margin-top: 15px; margin-bottom: 10px;'>🗺️ Canlı Piyasa Rejimi (Heatmap)</h3>", unsafe_allow_html=True)

    with st.spinner("Isı haritası verileri işleniyor..."):
        heatmap_html = "<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-bottom: 25px;'>"
        for m in MARKETS:
            rejim, renk = piyasa_rejimi_hesapla(m["symbol"])
            heatmap_html += f"<div style='background: {renk}; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 4px 6px rgba(0,0,0,0.1);'><div style='color: rgba(255,255,255,0.9); font-size: 11px; font-weight: 700; margin-bottom: 4px;'>{m['name']}</div><div style='color: #FFF; font-size: 13px; font-weight: 800;'>{rejim}</div></div>"
        
        heatmap_html += "</div>"
        st.markdown(heatmap_html, unsafe_allow_html=True)
        
    st.subheader("🚀 Küresel Takip Listesi (Tüm Piyasalar)")
    
elif secilen_sayfa == "🪙 Kripto Terminali":
    with st.spinner("Kripto psikolojisi sorgulanıyor..."):
        c_val, c_status, c_color = get_crypto_fng()
        c_vol = "Yüksek 🔥" if c_val > 65 else ("Düşük 💤" if c_val < 35 else "Normal 📊")
        c_vol_clr = "#34D399" if c_val > 65 else ("#64748B" if c_val < 35 else "#F59E0B")
        c_hac = "Güçlü 💰" if c_val > 55 else "Zayıf 📉"
        
    html_single_c = """<div style="background:#0F172A; border:1px solid #1E293B; padding:15px; border-radius:8px; margin-bottom:20px;">
        <div style="font-size:12px; font-weight:700; color:#64748B; margin-bottom:6px;">🪙 CANLI KRİPTO DUYARLILIĞI VE ANALİZİ (BTC/ETH)</div>
        <div style="background:#1E293B; height:8px; border-radius:4px; overflow:hidden; margin-bottom:10px;"><div style="background:{clr}; width:{val}%; height:8px;"></div></div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="font-size:12px; color:#94A3B8;">⚡ Volatilite: <b style="color:{v_clr};">{vol}</b> • 💵 Gerçek Hacim: <b style="color:#FFF;">{hac}</b></div>
            <div style="color:{clr}; font-weight:800; font-size:15px;">{stat} ({val}/100)</div>
        </div>
    </div>""".format(clr=c_color, val=c_val, v_clr=c_vol_clr, vol=c_vol, hac=c_hac, stat=c_status)
    st.markdown(html_single_c, unsafe_allow_html=True)
    
    aktif_list = [m for m in MARKETS if m["category"] == "Kripto"]
    st.subheader("🪙 Kripto Para Odası")

elif secilen_sayfa == "🇺🇸 NASDAQ Terminali":
    with st.spinner("NASDAQ dinamikleri hesaplanıyor..."):
        n_vol, n_vol_clr, n_hac = get_real_market_dynamics(["AAPL", "TSLA", "NVDA", "MSFT"])
        n_bar_color = "#EF4444" if "Kapalı" in n_hac else ("#10B981" if "Güçlü" in n_hac else "#94A3B8")
        
    html_single_n = """<div style="background:#0F172A; border:1px solid #1E293B; padding:15px; border-radius:8px; margin-bottom:20px;">
        <div style="font-size:12px; font-weight:700; color:#64748B; margin-bottom:6px;">🇺🇸 ABD TEKNOLOJİ BORSASI DİNAMİKLERİ (NASDAQ)</div>
        <div style="background:#1E293B; height:8px; border-radius:4px; overflow:hidden; margin-bottom:10px;"><div style="background:{b_clr}; width:100%; height:8px;"></div></div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="font-size:12px; color:#94A3B8;">⚡ Volatilite (ATR Oranı): <b style="color:{v_clr};">{vol}</b> • 💵 Durum: <b style="color:#FFF;">Hafta Sonu Kapanışı</b></div>
            <div style="color:{b_clr}; font-weight:800; font-size:15px;">{hac}</div>
        </div>
    </div>""".format(b_clr=n_bar_color, v_clr=n_vol_clr, vol=n_vol, hac=n_hac)
    st.markdown(html_single_n, unsafe_allow_html=True)
    
    aktif_list = [m for m in MARKETS if m["category"] == "NASDAQ"]
    st.subheader("🇺🇸 NASDAQ Hisse Senedi Odası")

elif secilen_sayfa == "👑 Emtia Terminali":
    with st.spinner("Emtia verileri analiz ediliyor..."):
        e_vol, e_vol_clr, e_hac = get_real_market_dynamics(["GC=F", "SI=F"])
        e_bar_color = "#EF4444" if "Kapalı" in e_hac else ("#10B981" if "Güçlü" in e_hac else "#94A3B8")
        
    html_single_e = """<div style="background:#0F172A; border:1px solid #1E293B; padding:15px; border-radius:8px; margin-bottom:20px;">
        <div style="font-size:12px; font-weight:700; color:#64748B; margin-bottom:6px;">👑 DEĞERLİ METAL PİYASA PSİKOLOJİSİ (ALTIN/GÜMÜŞ)</div>
        <div style="background:#1E293B; height:8px; border-radius:4px; overflow:hidden; margin-bottom:10px;"><div style="background:{b_clr}; width:100%; height:8px;"></div></div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="font-size:12px; color:#94A3B8;">⚡ Volatilite (ATR): <b style="color:{v_clr};">{vol}</b> • 💵 Durum: <b style="color:#FFF;">Hafta Sonu Kapanışı</b></div>
            <div style="color:{b_clr}; font-weight:800; font-size:15px;">{hac}</div>
        </div>
    </div>""".format(b_clr=e_bar_color, v_clr=e_vol_clr, vol=e_vol, hac=e_hac)
    st.markdown(html_single_e, unsafe_allow_html=True)
    
    aktif_list = [m for m in MARKETS if m["category"] == "Emtia"]
    st.subheader("👑 Emtia Vadeli İşlem Odası")


# --- TRADINGVIEW MODAL MOTORU ---
if st.session_state.chart_open:
    c_market = st.session_state.chart_open
    st.markdown("### 📊 Canlı Grafik: {} ({})".format(c_market['name'], c_market['tv']))
    tv_interval_map = {"1h": "60", "4h": "240", "1d": "D"}
    secilen_tv_interval = tv_interval_map.get(global_interval, "60")
    html_code = f"""<div id="tv-chart-container" style="height:450px;"></div><script src="https://s3.tradingview.com/tv.js"></script><script>new TradingView.widget({{"autosize":true,"symbol":"{c_market['tv']}","interval":"{secilen_tv_interval}","timezone":"Europe/Istanbul","theme":"dark","style":"1","locale":"tr","container_id":"tv-chart-container","studies":["RSI@tv-basicstudies","MAExp@tv-basicstudies"],"disabled_features":["header_saveload"]}});</script>"""
    st.components.v1.html(html_code, height=460)
    if st.button("❌ Grafiği Kapat"):
        st.session_state.chart_open = None
        st.rerun()
    st.markdown("---")

# CANLI TARAMA BUTONU VE İLERLEME ÇUBUĞU
if st.button("🚀 {} Odası İçin Canlı AI Taraması Başlat".format(secilen_sayfa)):
    st.info("📡 Sistem başlatıldı, piyasa verileri çekiliyor...")
    ilerleme_cubugu = st.progress(0)
    durum_metni = st.empty()
    
    yeni_sonuclar = {}
    toplam_varlik = len(aktif_list)
    
    for i, m in enumerate(aktif_list):
        # Ekrana hangi varlığın tarandığını yaz
        durum_metni.markdown(f"**🔍 Analiz Ediliyor:** {m['name']} ({i+1}/{toplam_varlik})")
        
        # Analizi çalıştır
        analiz = analiz_et_safe(m, global_min_hours, global_interval)
        
        if analiz: 
            yeni_sonuclar[m["symbol"]] = {"market": m, "result": analiz}
            
        # İlerleme çubuğunu güncelle
        ilerleme_cubugu.progress((i + 1) / toplam_varlik)
        
    durum_metni.success("✅ Tüm piyasa taraması ve XGBoost modellemesi tamamlandı!")
    st.session_state.results = yeni_sonuclar
    st.rerun() # Sayfayı yenileyip sonuçları hemen ekrana basmak için

# --- SİNYAL KARTLARININ EKRANA BASILMASI ---
valid_signals = {k: v for k, v in st.session_state.results.items() if v["market"] in aktif_list}

if not valid_signals:
    st.markdown("""<div style="background:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:35px; text-align:center; margin-top:10px;"><p style="color:#64748B; font-weight:600; margin:0;">Bu odada kriterlerine uyan aktif bir Doji sinyali bulunamadı.</p><p style="color:#475569; font-size:11px; margin:4px 0 0 0;">Sol menüden 'Zaman Filtresini Kaldır' seçeneğini aktif ederek geçmişteki son mumu ekrana zorlayabilirsiniz.</p></div>""", unsafe_allow_html=True)
else:
    for sym, data in valid_signals.items():
        m, r = data["market"], data["result"]
        is_buy = r["signal"] == "BUY"
        border_color = "#34D399" if is_buy else "#F87171"
        badge_bg = "rgba(52, 211, 153, 0.1)" if is_buy else "rgba(248, 113, 113, 0.1)"
        text_color = '#4ADE80' if r['change'] >= 0 else '#F87171'
        plus_sign = '+' if r['change'] >= 0 else ''
        is_confluence = (is_buy and r["bigTrend"] == "Boğa (Yukarı)") or (not is_buy and r["bigTrend"] == "Ayı (Aşağı)")
        
        if is_confluence:
            confluence_badge = '<span style="background:rgba(52,211,153,0.2); border:1px solid #34D399; color:#34D399; padding:3px 10px; border-radius:6px; font-weight:800;">🔥 Trend Uyumlu</span>'
        else:
            confluence_badge = '<span style="background:rgba(239,68,68,0.1); border:1px solid #EF4444; color:#EF4444; padding:3px 10px; border-radius:6px; font-weight:800;">⚠️ Trend Tersi Riskli</span>'
        
        html_card = """
        <div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); border: 1px solid #1E293B; border-left: 5px solid {b_color}; border-radius: 10px; padding: 15px; margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong style="color: #F1F5F9; font-size: 16px;">{m_name}</strong>
                    <span style="background: #020817; border: 1px solid #334155; color: #64748B; font-size: 11px; padding: 2px 6px; border-radius: 4px; margin-left: 8px;">{m_cat}</span>
                    <div style="color: #94A3B8; font-size: 13px; margin-top: 4px;">
                        ⏳ <b>Doji üzerinden geçen süre: {h_ago} Mum</b> ({d_type} Doji) • <span style="color: #CBD5E1;">Büyük Trend (4h): <b>{b_trend}</b></span>
                    </div>
                    <div style="margin-top: 8px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center;">
                        <span style="background: {b_bg}; border: 1px solid {b_color}; color: {b_color}; padding: 3px 10px; border-radius: 6px; font-weight: 700;">{sig}</span>
                        {conf_badge}
                        <span style="background: #020817; border: 1px solid #1E293B; color: #94A3B8; padding: 3px 10px; border-radius: 6px;">Tahmin Güveni: %{conf}</span>
                        <span style="background: #1E293B; border: 1px solid #F59E0B; color: #F59E0B; padding: 3px 10px; border-radius: 6px; font-weight: 600;">🎯 AI Tarihsel Win-Rate: %{w_rate} (Son {t_sig} geçmiş Doji modellemesinde)</span>
                        <span style="background: #020817; border: 1px solid #1E293B; color: #94A3B8; padding: 3px 10px; border-radius: 6px;">RSI: {rsi_v}</span>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="color: #F1F5F9; font-weight: 700; font-size: 18px; font-family: monospace;">${price:,.2f}</div>
                    <div style="color: {t_color}; font-size: 12px; font-family: monospace;">{p_sign}{change:.2f}%</div>
                </div>
            </div>
        </div>
        """.format(
            b_color=border_color, m_name=m['name'], m_cat=m['category'], h_ago=r['hoursAgo'],
            d_type=r['dojiType'], b_trend=r['bigTrend'], b_bg=badge_bg, sig=r['signal'],
            conf_badge=confluence_badge, conf=r['confidence'], w_rate=r['winRate'],
            t_sig=r['totalSignals'], rsi_v=r['rsi'], price=r['price'], t_color=text_color,
            p_sign=plus_sign, change=r['change']
        )
        st.markdown(html_card, unsafe_allow_html=True)
        
        if st.button("📊 {} Canlı Grafiğini İncele".format(m['name']), key="chart_btn_{}_{}".format(m['symbol'], m['category'])):
            st.session_state.chart_open = m
            st.rerun()
