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
def analiz_et_safe(market, min_hours, interval):
    try:
        if interval == "15m": periyot = "1mo"
        elif interval == "1h": periyot = "1y"
        elif interval == "4h": periyot = "2y"
        else: periyot = "5y"
        
        df = yf.download(market["symbol"], period=periyot, interval=interval, progress=False)
        if df.empty: return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Doji Tespiti
        govde = abs(df['Open'] - df['Close'])
        toplam_boy = df['High'] - df['Low']
        df['Doji'] = govde <= (toplam_boy * 0.1)
        
        # RSI Hesaplama
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # EMA20 Hesaplama
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # Gelişmiş Özellikler
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
        
        # Hedef Tanımlama (4 mum sonrası)
        df['Hedef'] = np.where(df['Close'].shift(-4) > df['Close'], 1, 0)
        özellikler = ['RSI', 'Price_to_EMA20', 'ATR', 'Upper_Shadow', 'Lower_Shadow', 'Volume_Shock']
        
        # Doji kontrolü (Canlı tarama için)
        doji_satirlari = df[df['Doji'] == True]
        if doji_satirlari.empty: return None
        
        son_doji_zaman = doji_satirlari.index[-1].to_pydatetime().replace(tzinfo=timezone.utc)
        su_an = datetime.now(timezone.utc)
        gecen_saat = round((su_an - son_doji_zaman).total_seconds() / 3600, 1)
        
        saat_katsayisi = 0.25 if interval == "15m" else (4 if interval == "4h" else (24 if interval == "1d" else 1))
        gecen_mum = round(gecen_saat / saat_katsayisi, 1)
        
        if gecen_mum < min_hours or gecen_mum > (24 / saat_katsayisi): return None
        
        X = df[özellikler].iloc[:-4]
        y = df['Hedef'].iloc[:-4]
        
        win_rate = 50
        toplam_sinyal = 0
        
        if len(X) > 50:
            # Tek bir model eğitiyoruz (Hafif ve ultra hızlı)
            model = RandomForestClassifier(n_estimators=60, max_depth=8, random_state=42)
            model.fit(X, y)
            
            # --- JET HIZINDA BACKTEST MOTORU ---
            # Modelin geçmişteki tüm Doji mumlarındaki performansını tek seferde ölçüyoruz
            doji_df = df[df['Doji'] == True].iloc[:-4] # Son mumlar hariç geçmiş dojiler
            if not doji_df.empty:
                X_doji = doji_df[özellikler]
                y_doji = doji_df['Hedef']
                
                preds = model.predict(X_doji)
                toplam_sinyal = len(y_doji)
                basarili_tahmin = np.sum(preds == y_doji.values)
                
                if toplam_sinyal > 0:
                    win_rate = int((basarili_tahmin / toplam_sinyal) * 100)
            # -----------------------------------
            
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
        
        return {
            "hoursAgo": gecen_mum,
            "signal": signal,
            "rsi": rsi_val,
            "confidence": guven_orani,
            "winRate": win_rate,
            "totalSignals": toplam_sinyal,
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
if "interval" not in st.session_state:
    st.session_state.interval = "1h"

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
<div style="background: linear-gradient(180deg, #0F172A 0%, #020817 100%); border-bottom: 1
