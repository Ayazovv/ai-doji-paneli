# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 23:04:21 2026

@author: ayazk
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier

# Sayfa Ayarları
st.set_page_config(page_title="AI Doji Sinyal Paneli", layout="wide", initial_sidebar_state="collapsed")

# --- ARKA PLAN PİYASA TANIMLARI ---
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

# --- SAF PYTHON MATEMATİĞİ İLE İNDİKATÖR HESAPLAMA (KÜTÜPHANESİZ) ---
def analiz_et_safe(market, min_hours):
    try:
        df = yf.download(market["symbol"], period="1mo", interval="1h", progress=False)
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
        df = df.dropna()
        
        doji_satirlari = df[df['Doji'] == True]
        if doji_satirlari.empty: return None
        
        son_doji_zaman = doji_satirlari.index[-1].to_pydatetime().replace(tzinfo=timezone.utc)
        su_an = datetime.now(timezone.utc)
        gecen_saat = round((su_an - son_doji_zaman).total_seconds() / 3600, 1)
        
        if gecen_saat < min_hours or gecen_saat > 24: return None
        
        df['Hedef'] = np.where(df['Close'].shift(-4) > df['Close'], 1, 0)
        X = df[['RSI', 'EMA20', 'Close']].iloc[:-4]
        y = df['Hedef'].iloc[:-4]
        
        if len(X) > 20:
            model = RandomForestClassifier(n_estimators=30, random_state=42)
            model.fit(X, y)
            son_veri = df[['RSI', 'EMA20', 'Close']].iloc[[-1]]
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
        
        return {
            "hoursAgo": gecen_saat,
            "signal": signal,
            "rsi": rsi_val,
            "confidence": guven_orani,
            "price": float(df['Close'].iloc[-1]),
            "change": float(((df['Close'].iloc[-1] - df['Open'].iloc[-12]) / df['Open'].iloc[-12]) * 100),
            "dojiType": doji_type
        }
    except Exception:
        return None

# --- STATE TANIMLAMALARI ---
if "selected_markets" not in st.session_state:
    st.session_state.selected_markets = [m["symbol"] for m in MARKETS]
if "min_hours" not in st.session_state:
    st.session_state.min_hours = 4
if "results" not in st.session_state:
    st.session_state.results = {}
if "chart_open" not in st.session_state:
    st.session_state.chart_open = None

# Stil Tanımlamaları
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #020817 !important;
        color: #F1F5F9 !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stButton>button {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
        color: white !important; border: none !important; border-radius: 8px !important;
        font-weight: 700 !important; padding: 10px 20px !important;
    }
</style>
""", unsafe_allow_html=True)

# PANEL BAŞLIĞI
st.markdown("""
<div style="background: linear-gradient(180deg, #0F172A 0%, #020817 100%); border-bottom: 1px solid #1E293B; padding: 15px; margin-bottom: 20px; border-radius: 8px;">
    <h1 style="margin: 0; font-size: 22px; font-weight: 800; color: #FFF;">🤖 AI Doji Sinyal Paneli</h1>
    <p style="margin: 0; font-size: 12px; color: #64748B;">Gerçek zamanlı borsa verileri • Python Uyumlu Kararlılık Modu</p>
</div>
""", unsafe_allow_html=True)

# --- TRADINGVIEW MODAL ALANI (SAYFA BAŞINDA DURMASI ÇAKIŞMALARI ENGELLER) ---
if st.session_state.chart_open:
    c_market = st.session_state.chart_open
    
    doji_turu = "Standart"
    gecen_sure = "Bilinmeyen"
    rsi_degeri = "-"
    sinyal_yonu = "-"
    
    # Python'da bulduğumuz Doji saatini çekelim
    if c_market["symbol"] in st.session_state.results:
        res_data = st.session_state.results[c_market["symbol"]]["result"]
        doji_turu = res_data["dojiType"]
        gecen_sure = f"{res_data['hoursAgo']} Saat Önce"
        rsi_degeri = res_data["rsi"]
        sinyal_yonu = res_data["signal"]

    st.markdown(f"### 📊 Canlı Grafik: {c_market['name']} ({c_market['tv']})")
    
    # Kullanıcıya nerede arayacağını net gösteren yeni bilgi paneli
    border_color = "#34D399" if sinyal_yonu == "BUY" else "#F87171"
    st.markdown(f"""
    <div style="background: #0F172A; border: 1px solid #1E293B; border-left: 5px solid {border_color}; padding: 15px; margin-bottom: 15px; border-radius: 8px;">
        <h4 style="margin: 0 0 8px 0; color: #FFF; font-size: 15px;">🔍 Doji Formasyon Lokasyonu</h4>
        <p style="margin: 0; font-size: 13px; color: #94A3B8; line-height: 1.5;">
            • <b>Tespit Edilen Tür:</b> <span style="color: #F59E0B;">{doji_turu} Doji</span><br>
            • <b>Zaman:</b> Grafikteki <b>en sağdaki (son) mumlardan yaklaşık {gecen_sure} önceki</b> muma bakmalısınız.<br>
            • <b>RSI Durumu:</b> {rsi_degeri} (Yapay zeka bu doğrultuda <b style="color: {border_color};">{sinyal_yonu}</b> yönlü tahmin üretti).
        </p>
        <p style="margin: 8px 0 0 0; font-size: 12px; color: #64748B; font-style: italic;">
            Not: TradingView widget kısıtlamaları nedeniyle grafik üzerine harici "D" harfi basılamamaktadır. Lütfen son mumdan geriye doğru saat hesabı yapınız.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Temizlenmiş ve kararlı TradingView kodu
    html_code = """
    <div id="tv-chart-container" style="height:500px;"></div>
    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
    <script type="text/javascript">
    new TradingView.widget({
      "autosize": true,
      "symbol": \"""" + str(c_market['tv']) + """\",
      "interval": "60",
      "timezone": "Europe/Istanbul",
      "theme": "dark",
      "style": "1",
      "locale": "tr",
      "container_id": "tv-chart-container",
      "studies": [
        "RSI@tv-basicstudies", 
        "MAExp@tv-basicstudies"
      ],
      "disabled_features": ["header_saveload"],
      "enabled_features": ["move_playlist_to_left"]
    });
    </script>
    """
    st.components.v1.html(html_code, height=510)
    if st.button("❌ Grafiği Kapat", key="close_chart_btn"):
        st.session_state.chart_open = None
        st.rerun()
    st.markdown("---")

# KONTROL BUTONLARI
st.write("📊 TAKİP LİSTESİ:")
cols = st.columns(len(MARKETS))
for i, m in enumerate(MARKETS):
    with cols[i]:
        is_selected = m["symbol"] in st.session_state.selected_markets
        label = f"✅ {m['name']}" if is_selected else f"❌ {m['name']}"
        if st.button(label, key=f"btn_{m['symbol']}", help=m["category"]):
            if is_selected: st.session_state.selected_markets.remove(m["symbol"])
            else: st.session_state.selected_markets.append(m["symbol"])
            st.rerun()

st.session_state.min_hours = st.slider("🎯 Doji Sonrası Minimum Geçmesi Gereken Süre (Saat)", 1, 12, st.session_state.min_hours)

if st.button("🚀 Piyasaları Canlı Tara ve Analiz Et", key="scan_markets_main"):
    with st.spinner("Canlı fiyat verileri çekiliyor ve yapay zeka eğitiliyor..."):
        yeni_sonuclar = {}
        for m in MARKETS:
            if m["symbol"] in st.session_state.selected_markets:
                analiz = analiz_et_safe(m, st.session_state.min_hours)
                if analiz:
                    yeni_sonuclar[m["symbol"]] = {"market": m, "result": analiz}
        st.session_state.results = yeni_sonuclar

# SİNYAL KARTLARI
if not st.session_state.results:
    st.markdown("""
    <div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 12px; padding: 40px; text-align: center; margin-top: 20px;">
        <p style="color: #64748B; font-weight: 600; margin: 0;">Şu anda belirlediğin saat kriterine uyan aktif bir Doji sinyali yok.</p>
        <p style="color: #475569; font-size: 12px; margin: 5px 0 0 0;">"Piyasaları Canlı Tara" butonuna basarak taramayı başlatabilirsin.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    for sym, data in st.session_state.results.items():
        m = data["market"]
        r = data["result"]
        is_buy = r["signal"] == "BUY"
        border_color = "#34D399" if is_buy else "#F87171"
        badge_bg = "rgba(52, 211, 153, 0.1)" if is_buy else "rgba(248, 113, 113, 0.1)"
        text_color = '#4ADE80' if r['change'] >= 0 else '#F87171'
        plus_sign = '+' if r['change'] >= 0 else ''
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); border: 1px solid #1E293B; border-left: 5px solid {border_color}; border-radius: 10px; padding: 15px; margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong style="color: #F1F5F9; font-size: 16px;">{m['name']}</strong>
                    <span style="background: #020817; border: 1px solid #334155; color: #64748B; font-size: 11px; padding: 2px 6px; border-radius: 4px; margin-left: 8px;">{m['category']}</span>
                    <div style="color: #94A3B8; font-size: 13px; margin-top: 4px;">
                        ⏳ <b>Doji üzerinden geçen süre: {r['hoursAgo']} Saat</b> ({r['dojiType']} Doji)
                    </div>
                    <div style="margin-top: 8px; display: flex; gap: 8px;">
                        <span style="background: {badge_bg}; border: 1px solid {border_color}; color: {border_color}; padding: 3px 10px; border-radius: 6px; font-weight: 700;">{r['signal']}</span>
                        <span style="background: #020817; border: 1px solid #1E293B; color: #94A3B8; padding: 3px 10px; border-radius: 6px;">Yapay Zeka Doğruluğu: %{r['confidence']}</span>
                        <span style="background: #020817; border: 1px solid #1E293B; color: #94A3B8; padding: 3px 10px; border-radius: 6px;">RSI: {r['rsi']}</span>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="color: #F1F5F9; font-weight: 700; font-size: 18px; font-family: monospace;">${r['price']:,}</div>
                    <div style="color: {text_color}; font-size: 12px; font-family: monospace;">{plus_sign}{round(r['change'], 2)}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Grafik Butonu
        if st.button(f"📊 {m['name']} Canlı Grafiğini İncele", key=f"chart_btn_{m['symbol']}"):
            st.session_state.chart_open = m
            st.rerun()
