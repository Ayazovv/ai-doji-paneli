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
        
        son_doji_zaman = doji_satirlari.index[-1].
