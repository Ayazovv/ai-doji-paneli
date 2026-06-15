# -*- coding: utf-8 -*-
"""
AI Doji Terminali - v6.9.4 (Production Release - Kusursuz ML & Sinyal Motoru)
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone
import xgboost as xgb
import concurrent.futures
import traceback
from sklearn.model_selection import TimeSeriesSplit

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="AI Doji Terminali v6.9.4", layout="wide", initial_sidebar_state="auto")

# --- HIZLANDIRICI: CACHE (ÖNBELLEK) FONKSİYONU ---
@st.cache_data(ttl=300) 
def veri_indir(symbol, periyot, interval):
    return yf.download(symbol, period=periyot, interval=interval, progress=False)

# --- CANLI SPOT İÇİN CACHE'SİZ FONKSİYON ---
def canli_spot_cek(symbol):
    try:
        spot_df = yf.download(symbol, period="1d", interval="1m", progress=False)
        if spot_df is not None and not spot_df.empty:
            if isinstance(spot_df.columns, pd.MultiIndex):
                spot_df.columns = spot_df.columns.get_level_values(0)
            return float(spot_df['Close'].iloc[-1])
    except:
        pass
    return None

# --- GLOBAL PİYASA TANIMLARI ---
MARKETS = [
    {"name": "Altın (XAU/USD)", "symbol": "GC=F", "tv": "OANDA:XAUUSD", "category": "Emtia", "color": "#F59E0B"},
    {"name": "Gümüş (XAG/USD)", "symbol": "SI=F", "tv": "OANDA:XAGUSD", "category": "Emtia", "color": "#94A3B8"},
    {"name": "EUR/USD", "symbol": "EURUSD=X", "tv": "OANDA:EURUSD", "category": "Forex", "color": "#3B82F6"},
    {"name": "GBP/USD", "symbol": "GBPUSD=X", "tv": "OANDA:GBPUSD", "category": "Forex", "color": "#8B5CF6"},
    {"name": "USD/JPY", "symbol": "JPY=X", "tv": "OANDA:USDJPY", "category": "Forex", "color": "#10B981"},
    {"name": "NASDAQ Endeksi", "symbol": "^IXIC", "tv": "NASDAQ:IXIC", "category": "NASDAQ", "color": "#A855F7"},
    {"name": "Apple", "symbol": "AAPL", "tv": "NASDAQ:AAPL", "category": "NASDAQ", "color": "#60A5FA"},
    {"name": "Tesla", "symbol": "TSLA", "tv": "NASDAQ:TSLA", "category": "NASDAQ", "color": "#EF4444"},
    {"name": "NVIDIA", "symbol": "NVDA", "tv": "NASDAQ:NVDA", "category": "NASDAQ", "color": "#10B981"},
    {"name": "Microsoft", "symbol": "MSFT", "tv": "NASDAQ:MSFT", "category": "NASDAQ", "color": "#3B82F6"},
    {"name": "Bitcoin", "symbol": "BTC-USD", "tv": "BINANCE:BTCUSDT", "category": "Kripto", "color": "#F59E0B"},
    {"name": "Ethereum", "symbol": "ETH-USD", "tv": "BINANCE:ETHUSDT", "category": "Kripto", "color": "#6366F1"},
    {"name": "Solana", "symbol": "SOL-USD", "tv": "BINANCE:SOLUSDT", "category": "Kripto", "color": "#14B8A6"},
    {"name": "BNB", "symbol": "BNB-USD", "tv": "BINANCE:BNBUSDT", "category": "Kripto", "color": "#EAB308"},
    {"name": "BIST 100", "symbol": "XU100.IS", "tv": "BIST:XU100", "category": "BIST", "color": "#06B6D4"},
    {"name": "Garanti BBVA", "symbol": "GARAN.IS", "tv": "BIST:GARAN", "category": "BIST", "color": "#10B981"}
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

def get_nasdaq_fng():
    try:
        vix_df = yf.download("^VIX", period="5d", interval="1d", progress=False)
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = vix_df.columns.get_level_values(0)
        son_vix = float(vix_df['Close'].iloc[-1])
        fng_score = 100 - ((son_vix - 10) / (40 - 10) * 100)
        fng_score = max(0, min(100, int(fng_score))) 
        if fng_score < 25: return fng_score, "Aşırı Korku 😱", "#EF4444"
        elif fng_score < 45: return fng_score, "Korku 😨", "#F97316"
        elif fng_score < 55: return fng_score, "Nötr 😐", "#94A3B8"
        elif fng_score < 75: return fng_score, "Açgözlülük 🤑", "#10B981"
        else: return fng_score, "Aşırı Açgözlülük 🚀", "#34D399"
    except:
        return 50, "Nötr 😐", "#94A3B8"

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
            
            if 'Volume' in df.columns and not df['Volume'].isna().all() and not df['Volume'].eq(0).all():
                aktif_hacim_idx = -1 if df['Volume'].iloc[-1] > 0 else -2
                son_hacim = df['Volume'].rolling(3).mean().iloc[aktif_hacim_idx]
                ort_hacim = df['Volume'].rolling(20).mean().iloc[aktif_hacim_idx]
                hacim_soku = son_hacim / (ort_hacim + 1e-10)
                vol_results.append(hacim_soku)
                vol_counts += 1

        final_vol, final_vol_clr, final_hac = "Düşük 💤", "#64748B", "Veri Yok 🚫"
        if vol_ratios:
            avg_vol_ratio = sum(vol_ratios) / len(vol_ratios)
            if avg_vol_ratio > 2.0: final_vol, final_vol_clr = "Yüksek 🔥", "#34D399"
            elif avg_vol_ratio > 1.0: final_vol, final_vol_clr = "Normal 📊", "#F59E0B"
            
        if vol_counts > 0 and len(vol_results) > 0:
            avg_hacim_soku = sum(vol_results) / vol_counts
            if avg_hacim_soku > 1.15: final_hac = "Güçlü 💰"
            elif avg_hacim_soku > 0.85: final_hac = "Normal 📈"
            else: final_hac = "Zayıf 📉"
            
        return final_vol, final_vol_clr, final_hac
    except:
        return "Normal 📊", "#F59E0B", "Normal 📈"

def dinamik_piyasa_durumu(kategori="Genel"):
    if kategori == "Kripto": return "Açık 🟢"
    
    now_utc = datetime.now(timezone.utc)
    gun = now_utc.weekday() 
    saat = now_utc.hour 
    tarih_str = now_utc.strftime("%Y-%m-%d")
    
    NYSE_TATILLER_2026 = ["2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25"]
    
    if tarih_str in NYSE_TATILLER_2026 and kategori in ["NASDAQ", "Genel"]: 
        return "Tatil 💤"
    
    if kategori == "BIST":
        if gun >= 5: return "Kapalı 💤"
        if 7 <= saat < 15 or (saat == 15 and now_utc.minute <= 10): return "Açık 🟢"
        return "Kapalı 💤"
        
    if kategori in ["Forex", "Emtia"]:
        if gun == 5: return "Kapalı 💤"
        if gun == 6 and saat < 22: return "Kapalı 💤"
        if gun == 4 and saat >= 21: return "Kapalı 💤"
        return "Açık 🟢"
        
    if gun >= 5: return "Kapalı 💤"
    if gun == 4 and saat >= 21: return "Kapalı 💤"
    return "Açık 🟢"

def piyasa_rejimi_hesapla(symbol):
    try:
        df = veri_indir(symbol, "60d", "1d")
        if df.empty: return "Veri Yok", "#334155"
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

        if ema20 > ema50 and rsi > 60: return "Güçlü Boğa 🚀", "#10B981"
        elif ema20 > ema50: return "Boğa 🟢", "#059669"
        elif ema20 < ema50 and rsi < 40: return "Güçlü Ayı 🩸", "#EF4444"
        elif ema20 < ema50: return "Ayı 🔴", "#B91C1C"
        else: return "Yatay ⚪", "#64748B"
    except:
        return "Bilinmiyor", "#334155"

def buyuk_trend_kontrol(symbol):
    try:
        df_big = veri_indir(symbol, "1y", "4h")
        if df_big.empty: return "Yansız"
        if isinstance(df_big.columns, pd.MultiIndex):
            df_big.columns = df_big.columns.get_level_values(0)
            
        ema200 = df_big['Close'].ewm(span=200, min_periods=200, adjust=False).mean()
        son_ema = ema200.dropna()
        if son_ema.empty: return "Yansız"
        
        ema200_son = son_ema.iloc[-1]
        son_fiyat = df_big['Close'].iloc[-1]
        return "Boğa (Yukarı)" if son_fiyat > ema200_son else "Ayı (Aşağı)"
    except:
        return "Yansız"

def analiz_et_safe(market, min_hours, interval, doji_modu, is_forced):
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
        
        if "Dinamik" in doji_modu:
            ortalama_boy = toplam_boy.rolling(window=20).mean()
            volatilite_carpani = toplam_boy / (ortalama_boy + 1e-10)
            if market["category"] == "Forex":
                dinamik_sinir = (0.12 * volatilite_carpani).clip(lower=0.05, upper=0.20)
            else:
                dinamik_sinir = (0.3 * volatilite_carpani).clip(lower=0.15, upper=0.45)
            df['Doji'] = govde <= (toplam_boy * dinamik_sinir)
        else:
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
        
        if 'Volume' in df.columns and not df['Volume'].isna().all() and not df['Volume'].eq(0).all():
            df['Volume_Shock'] = df['Volume'].rolling(5).mean() / (df['Volume'].rolling(20).mean() + 1e-10)
        else:
            df['Volume_Shock'] = 1.0 
            
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
        
        gercek_canli_fiyat = float(df['Close'].iloc[-1])
        tam_veri_uzunlugu = len(df)
        df['Gercek_Sira'] = range(tam_veri_uzunlugu)
        
        suanki_fiyat = df['Close']
        ilerideki_kapanis = df['Close'].shift(-int(min_hours))
        
        # ML Eğitimi İçin Kusursuz SL/TP Penceresi Hizalaması (i+1 to i+N)
        ilerideki_min = df['Low'].rolling(window=int(min_hours)).min().shift(-int(min_hours))
        ilerideki_max = df['High'].rolling(window=int(min_hours)).max().shift(-int(min_hours))
        
        if market["category"] == "Forex": marj = 0.002    
        elif market["category"] == "Kripto": marj = 0.012 
        else: marj = 0.006                                
        
        buy_target = np.where((ilerideki_kapanis > suanki_fiyat * (1 + marj)) & (ilerideki_min >= suanki_fiyat * (1 - marj)), 1, 0)
        sell_target = np.where((ilerideki_kapanis < suanki_fiyat * (1 - marj)) & (ilerideki_max <= suanki_fiyat * (1 + marj)), 1, 0)
        
        df['Hedef'] = np.where(buy_target == 1, 1, np.where(sell_target == 1, 0, -1))
        
        train_df = df[df['Hedef'] != -1].copy() 
        
        features = [
            'RSI', 'Price_to_EMA20', 'ATR', 'Upper_Shadow', 'Lower_Shadow', 
            'MACD_Hist', 'BB_Width', 'Price_to_BB', 'Trend_Slope'
        ]
        if market["category"] not in ["Forex", "Emtia", "Özel"]: features.append('Volume_Shock')
        
        feature_names_tr = {
            'RSI': 'RSI', 'Price_to_EMA20': 'Trend Uzaklığı', 'ATR': 'Volatilite',
            'Upper_Shadow': 'Üst Gölge', 'Lower_Shadow': 'Alt Gölge', 'Volume_Shock': 'Hacim Şoku',
            'MACD_Hist': 'MACD', 'BB_Width': 'Bollinger Sıkışması', 'Price_to_BB': 'Bant Konumu',
            'Trend_Slope': 'Trend Eğimi'
        }
        
        son_50_mum = df.tail(50)
        doji_olanlar = son_50_mum[son_50_mum['Doji'] == True]
        
        if interval == "1h": min_mum, max_mum = 4, 10  
        elif interval == "4h": min_mum, max_mum = 1, 3   
        else: min_mum, max_mum = 1, 5   
            
        olgun_dojiler = []
        if not doji_olanlar.empty:
            for idx in doji_olanlar.index:
                orijinal_sira = df.loc[idx, 'Gercek_Sira']
                mum_yasi = tam_veri_uzunlugu - 1 - orijinal_sira
                if min_mum <= mum_yasi <= max_mum: 
                    olgun_dojiler.append(mum_yasi)
                elif is_forced:
                    olgun_dojiler.append(mum_yasi)
                    
        if not olgun_dojiler: 
            return None 
            
        gecen_mum = min(olgun_dojiler)
            
        doji_train_df = train_df[train_df['Doji'] == True]
        
        if len(doji_train_df) >= 20 and len(doji_train_df['Hedef'].unique()) > 1: 
            X = doji_train_df[features]
            y = doji_train_df['Hedef']
        elif len(train_df['Hedef'].unique()) > 1:
            X = train_df[features]
            y = train_df['Hedef']
        else:
            return None 
            
        win_rate, toplam_sinyal = 50, 0
        en_etkili_faktorler = {}
        
        model = xgb.XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.03, subsample=0.3,
            colsample_bytree=0.4, reg_lambda=20.0, reg_alpha=10.0,
            random_state=42, eval_metric='logloss', n_jobs=-1
        )
        
        if len(X) > 15:
            tscv = TimeSeriesSplit(n_splits=5)
            fold_scores = []
            
            for train_idx, test_idx in tscv.split(X):
                try:
                    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
                    model.fit(X_tr, y_tr)
                    preds = model.predict(X_te)
                    fold_scores.append(np.mean(preds == y_te.values))
                except: continue
                
            if fold_scores: win_rate = int(np.mean(fold_scores) * 100)
            
            model.fit(X, y)
            
            son_veri = df[features].iloc[[-1]]
            tahmin_yon = model.predict(son_veri)[0]
            
            try:
                guven_orani = int(max(model.predict_proba(son_veri)[0]) * 100)
            except:
                guven_orani = 55
                
            toplam_sinyal = len(y)
            
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
        
        doji_iloc = tam_veri_uzunlugu - 1 - gecen_mum

        upper_shadow_val = float(df['Upper_Shadow'].iloc[doji_iloc])
        lower_shadow_val = float(df['Lower_Shadow'].iloc[doji_iloc])
        
        if upper_shadow_val > 0.6 and lower_shadow_val < 0.2: doji_type = "Gravestone"  
        elif lower_shadow_val > 0.6 and upper_shadow_val < 0.2: doji_type = "Dragonfly"   
        else: doji_type = "Standard"    
        
        _lookback = {"1h": 24, "4h": 30, "1d": 5}.get(interval, 12)
        _lookback = max(1, min(_lookback, len(df) - 1))
        
        rebound_pct = 0.0
        drawdown_pct = 0.0
        yapisal_short_guclu = False 
        yapisal_long_guclu = False  
        
        if gecen_mum > 0:
            gelecek_mumlar_high = df['High'].iloc[doji_iloc + 1:]
            gelecek_mumlar_low = df['Low'].iloc[doji_iloc + 1:]
            gelecek_mumlar_close = df['Close'].iloc[doji_iloc + 1:]
            
            if not gelecek_mumlar_high.empty:
                doji_close = float(df['Close'].iloc[doji_iloc])
                doji_low = float(df['Low'].iloc[doji_iloc])
                doji_high = float(df['High'].iloc[doji_iloc])
                
                if doji_close > 0:
                    if signal == "SELL":
                        peak_after = float(gelecek_mumlar_high.max())
                        rebound_pct = round(((peak_after - doji_close) / doji_close) * 100, 2)
                        
                        en_dusuk_alinmadi = float(gelecek_mumlar_low.min()) >= doji_low
                        altinda_kapanis_yok = float(gelecek_mumlar_close.min()) >= doji_close
                        
                        if (en_dusuk_alinmadi or altinda_kapanis_yok) and (gercek_canli_fiyat > doji_close):
                            yapisal_short_guclu = True
                            
                    elif signal == "BUY":
                        dip_after = float(gelecek_mumlar_low.min())
                        drawdown_pct = round(((doji_close - dip_after) / doji_close) * 100, 2)
                        
                        en_yuksek_alinmadi = float(gelecek_mumlar_high.max()) <= doji_high
                        ustunde_kapanis_yok = float(gelecek_mumlar_close.max()) <= doji_close
                        
                        if (en_yuksek_alinmadi or ustunde_kapanis_yok) and (gercek_canli_fiyat < doji_close):
                            yapisal_long_guclu = True

        big_trend = buyuk_trend_kontrol(market["symbol"])
        is_confluence = (signal == "BUY" and big_trend == "Boğa (Yukarı)") or (signal == "SELL" and big_trend == "Ayı (Aşağı)")
        
        price_to_bb = float(df['Price_to_BB'].iloc[-1])
        macd_hist = float(df['MACD_Hist'].iloc[-1])
        rsi_val = float(df['RSI'].iloc[-1])
        
        skor = 0
        if guven_orani >= 65: skor += 2
        elif guven_orani >= 55: skor += 1
        if is_confluence: skor += 2
        if (signal == "BUY" and rsi_val < 35) or (signal == "SELL" and rsi_val > 65): skor += 2
        if (signal == "BUY" and price_to_bb < 0.2) or (signal == "SELL" and price_to_bb > 0.8): skor += 1
        
        # MACD'nin doji dönüşlerinde fiyata olan ters (aşırı satım/alım) konumu değerlendirildi
        if (signal == "BUY" and macd_hist < 0) or (signal == "SELL" and macd_hist > 0): skor += 1
        if signal == "SELL" and yapisal_short_guclu: skor += 2
        elif signal == "BUY" and yapisal_long_guclu: skor += 2
        
        skor = min(skor, 10)
            
        gosterim_fiyati = gercek_canli_fiyat
        
        if market["category"] == "Emtia":
            spot_sym = "XAUUSD=X" if "Altın" in market["name"] else ("XAGUSD=X" if "Gümüş" in market["name"] else market["symbol"])
            canli_fiyat = canli_spot_cek(spot_sym)
            if canli_fiyat:
                gosterim_fiyati = canli_fiyat
                
        degisim_yuzdesi = float(((gosterim_fiyati - df['Open'].iloc[-_lookback]) / (df['Open'].iloc[-_lookback] + 1e-10)) * 100)

        return {
            "hoursAgo": gecen_mum, "signal": signal, "rsi": rsi_val, "confidence": guven_orani,
            "winRate": win_rate, "totalSignals": toplam_sinyal, "bigTrend": big_trend,
            "price": gosterim_fiyati, "skor": skor,
            "change": degisim_yuzdesi,
            "dojiType": doji_type, "topFeatures": en_etkili_faktorler,
            "reboundPct": rebound_pct, "drawdownPct": drawdown_pct,
            "yapisalShortGuclu": yapisal_short_guclu,
            "yapisalLongGuclu": yapisal_long_guclu
        }
    except Exception as e:
        hata_mesaji = f"{market['symbol']} Hatası: {str(e)}\n{traceback.format_exc()}"
        st.session_state.setdefault("hatalar", []).append(hata_mesaji)
        return None

# --- STATE TANIMLAMALARI ---
if "results" not in st.session_state: st.session_state.results = {}
if "chart_open" not in st.session_state: st.session_state.chart_open = None
if "force_past" not in st.session_state: st.session_state.force_past = False
if "strict_mode" not in st.session_state: st.session_state.strict_mode = False
if "hatalar" not in st.session_state: st.session_state.hatalar = []
if "ozel_semboller" not in st.session_state: st.session_state.ozel_semboller = []

# Stil Tanımlamaları
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    html, body, [data-testid="stAppViewContainer"] { background-color: #020817 !important; color: #F1F5F9 !important; font-family: 'Inter', sans-serif !important; }
    .stButton>button { background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important; color: white !important; border: none !important; border-radius: 8px !important; font-weight: 700 !important; }
    [data-testid="stSidebar"] { background-color: #0F172A !important; border-right: 1px solid #1E293B !important; }
</style>
""", unsafe_allow_html=True)

# --- SOL MENÜ NAVİGASYONU (SIDEBAR) ---
st.sidebar.markdown("""
<div style='text-align: center; padding: 10px; border-bottom: 1px solid #1E293B; margin-bottom: 20px;'>
    <h3 style='color: #FFF; margin: 0; font-size: 16px;'>🌐 AI TERMINAL v6.9.4</h3>
</div>
""", unsafe_allow_html=True)

secilen_sayfa = st.sidebar.radio(
    "📊 İŞLEM ODALARI",
    ["🏠 Genel Dashboard", "💱 Forex Terminali", "🪙 Kripto Terminali", "🇺🇸 NASDAQ Terminali", "👑 Emtia Terminali", "🇹🇷 BIST Terminali", "✨ Özel İzleme Listesi"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Küresel Filtreler")
global_interval = st.sidebar.selectbox("⏳ Zaman Dilimi (Periyot)", ["1h", "4h", "1d"], index=0)
global_min_hours = st.sidebar.slider("🎯 AI Gelecek Vadesi (İleri Mum)", 1, 12, 4)
global_doji_modu = st.sidebar.radio("🎯 Doji Hassasiyet Modu", ["Dinamik (Otomatik Esner)", "Sabit (Klasik 0.3)"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("➕ Özel Sembol Ekle")
ozel_sembol = st.sidebar.text_input("Yahoo Finance Sembolü", placeholder="Örn: THYAO.IS")
if st.sidebar.button("Listeye Ekle") and ozel_sembol:
    st.session_state.ozel_semboller.append({
        "name": ozel_sembol.upper(), "symbol": ozel_sembol.upper(),
        "tv": ozel_sembol.upper(), "category": "Özel", "color": "#8B5CF6"
    })
    st.sidebar.success(f"{ozel_sembol} eklendi!")
    st.rerun()

TUM_MARKETLER = MARKETS + st.session_state.ozel_semboller

st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Sistem Test & Görünüm")
st.session_state.force_past = st.sidebar.checkbox("🔓 Zaman Filtresini Kaldır (Eski Dojileri Bul)", value=st.session_state.force_past)
st.session_state.strict_mode = st.sidebar.checkbox("🔒 Skor Filtresi (Sadece 5+ Rozetleri Göster)", value=st.session_state.strict_mode)

if st.session_state.hatalar:
    st.sidebar.markdown("---")
    with st.sidebar.expander("🐛 Hata Logu", expanded=False):
        for hata in st.session_state.hatalar[-10:]:
            st.code(hata, language="text")
        if st.sidebar.button("Logları Temizle"):
            st.session_state.hatalar = []
            st.rerun()

st.markdown(f"""
<div style="background: linear-gradient(180deg, #0F172A 0%, #020817 100%); border-bottom: 1px solid #1E293B; padding: 15px; margin-bottom: 15px; border-radius: 8px;">
    <h1 style="margin: 0; font-size: 22px; font-weight: 800; color: #FFF;">🤖 Joe Barbarov AI Terminal v6.9.4</h1>
    <p style="margin: 0; font-size: 12px; color: #64748B;">Oda: <b>{secilen_sayfa}</b> • Gerçek Zamanlı Veri İşleme & PA Fırsat Sıralaması</p>
</div>
""", unsafe_allow_html=True)

# --- PANEL İÇERİĞİ VE GÖRSEL KARTLAR ---
aktif_list = []

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
        p_durum = dinamik_piyasa_durumu()

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
        n_fng_val, n_fng_stat, n_fng_clr = get_nasdaq_fng()
        html_n = """<div style="background:#0F172A; border:1px solid #1E293B; padding:12px; border-radius:8px; min-height:110px;">
            <div style="font-size:11px; font-weight:700; color:#64748B; margin-bottom:6px;">🇺🇸 ABD BORSALARI (NASDAQ)</div>
            <div style="background:#1E293B; height:6px; border-radius:3px; overflow:hidden; margin-bottom:8px;"><div style="background:{clr}; width:{val}%; height:6px;"></div></div>
            <div style="color:{clr}; font-weight:800; font-size:13px; text-align:right; margin-bottom:6px;">{stat} ({val}/100)</div>
            <div style="display:flex; justify-content:space-between; font-size:10px; color:#64748B; border-top:1px solid rgba(51,65,85,0.3); padding-top:4px;">
                <span>⚡ Vol: <b style="color:{v_clr};">{vol}</b></span>
                <span>💵 Hacim: <b style="color:#FFF;">{hac}</b></span>
                <span>📍 <b style="color:#FFF;">{durum}</b></span>
            </div>
        </div>""".format(clr=n_fng_clr, val=n_fng_val, stat=n_fng_stat, v_clr=n_vol_clr, vol=n_vol, hac=n_hac, durum=p_durum)
        st.markdown(html_n, unsafe_allow_html=True)
        
    with fng_cols[2]:
        html_e = """<div style="background:#0F172A; border:1px solid #1E293B; padding:12px; border-radius:8px; min-height:110px;">
            <div style="font-size:11px; font-weight:700; color:#64748B; margin-bottom:6px;">👑 EMTİA PİYASASI (ALTIN/GÜMÜŞ)</div>
            <div style="background:#1E293B; height:6px; border-radius:3px; overflow:hidden; margin-bottom:8px;"><div style="background:{b_clr}; width:100%; height:6px;"></div></div>
            <div style="color:{b_clr}; font-weight:800; font-size:13px; text-align:right; margin-bottom:6px;">Piyasa: {durum}</div>
            <div style="display:flex; justify-content:space-between; font-size:10px; color:#64748B; border-top:1px solid rgba(51,65,85,0.3); padding-top:4px;"><span>⚡ Vol: <b style="color:{v_clr};">{vol}</b></span><span>💵 Hacim: <b style="color:#FFF;">{hac}</b></span></div>
        </div>""".format(b_clr=e_bar_color, hac=e_hac, v_clr=e_vol_clr, vol=e_vol, durum=p_durum)
        st.markdown(html_e, unsafe_allow_html=True)
        
    aktif_list = TUM_MARKETLER
    st.markdown("<h3 style='color: #F1F5F9; font-size: 16px; margin-top: 15px; margin-bottom: 10px;'>🗺️ Canlı Piyasa Rejimi (Isı Haritası)</h3>", unsafe_allow_html=True)
    with st.spinner("Isı haritası verileri işleniyor..."):
        heatmap_html = "<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-bottom: 25px;'>"
        for m in aktif_list[:12]:
            rejim, renk = piyasa_rejimi_hesapla(m["symbol"])
            heatmap_html += f"<div style='background: {renk}; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 4px 6px rgba(0,0,0,0.1);'><div style='color: rgba(255,255,255,0.9); font-size: 11px; font-weight: 700; margin-bottom: 4px;'>{m['name']}</div><div style='color: #FFF; font-size: 13px; font-weight: 800;'>{rejim}</div></div>"
        heatmap_html += "</div>"
        st.markdown(heatmap_html, unsafe_allow_html=True)

elif secilen_sayfa == "💱 Forex Terminali":
    with st.spinner("Forex (Döviz) verileri analiz ediliyor..."):
        f_vol, f_vol_clr, f_hac = get_real_market_dynamics(["EURUSD=X"])
        if "Veri Yok" in f_hac:
            f_hac, f_bar_color = "Merkeziyetsiz Hacim 🌐", "#3B82F6" 
        else:
            f_bar_color = "#EF4444" if "Kapalı" in f_hac else ("#10B981" if "Güçlü" in f_hac else "#94A3B8")
        p_durum = dinamik_piyasa_durumu("Forex")
        
    html_single_f = """<div style="background:#0F172A; border:1px solid #1E293B; padding:15px; border-radius:8px; margin-bottom:20px;">
        <div style="font-size:12px; font-weight:700; color:#64748B; margin-bottom:6px;">💱 KÜRESEL DÖVİZ PİYASASI (FOREX)</div>
        <div style="background:#1E293B; height:8px; border-radius:4px; overflow:hidden; margin-bottom:10px;"><div style="background:{b_clr}; width:100%; height:8px;"></div></div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="font-size:12px; color:#94A3B8;">⚡ Volatilite (ATR): <b style="color:{v_clr};">{vol}</b> • 💵 Durum: <b style="color:#FFF;">{durum}</b></div>
            <div style="color:{b_clr}; font-weight:800; font-size:14px;">{hac}</div>
        </div>
    </div>""".format(b_clr=f_bar_color, v_clr=f_vol_clr, vol=f_vol, hac=f_hac, durum=p_durum)
    st.markdown(html_single_f, unsafe_allow_html=True)
    aktif_list = [m for m in TUM_MARKETLER if m["category"] == "Forex"]
        
elif secilen_sayfa == "🪙 Kripto Terminali":
    with st.spinner("Kripto duyarlılığı sorgulanıyor..."):
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
    aktif_list = [m for m in TUM_MARKETLER if m["category"] == "Kripto"]

elif secilen_sayfa == "🇺🇸 NASDAQ Terminali":
    with st.spinner("NASDAQ dinamikleri hesaplanıyor..."):
        n_vol, n_vol_clr, n_hac = get_real_market_dynamics(["AAPL", "TSLA", "NVDA", "MSFT"])
        n_bar_color = "#EF4444" if "Kapalı" in n_hac else ("#10B981" if "Güçlü" in n_hac else "#94A3B8")
        p_durum = dinamik_piyasa_durumu("NASDAQ")
        
    html_single_n = """<div style="background:#0F172A; border:1px solid #1E293B; padding:15px; border-radius:8px; margin-bottom:20px;">
        <div style="font-size:12px; font-weight:700; color:#64748B; margin-bottom:6px;">🇺🇸 ABD TEKNOLOJİ BORSASI DİNAMİKLERİ (NASDAQ)</div>
        <div style="background:#1E293B; height:8px; border-radius:4px; overflow:hidden; margin-bottom:10px;"><div style="background:{b_clr}; width:100%; height:8px;"></div></div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="font-size:12px; color:#94A3B8;">⚡ Volatilite (ATR): <b style="color:{v_clr};">{vol}</b> • 💵 Durum: <b style="color:#FFF;">{durum}</b></div>
            <div style="color:{b_clr}; font-weight:800; font-size:15px;">{hac}</div>
        </div>
    </div>""".format(b_clr=n_bar_color, v_clr=n_vol_clr, vol=n_vol, hac=n_hac, durum=p_durum)
    st.markdown(html_single_n, unsafe_allow_html=True)
    aktif_list = [m for m in TUM_MARKETLER if m["category"] == "NASDAQ"]

elif secilen_sayfa == "👑 Emtia Terminali":
    with st.spinner("Emtia verileri analiz ediliyor..."):
        e_vol, e_vol_clr, e_hac = get_real_market_dynamics(["GC=F", "SI=F"])
        e_bar_color = "#EF4444" if "Kapalı" in e_hac else ("#10B981" if "Güçlü" in e_hac else "#94A3B8")
        p_durum = dinamik_piyasa_durumu("Emtia")
        
    html_single_e = """<div style="background:#0F172A; border:1px solid #1E293B; padding:15px; border-radius:8px; margin-bottom:20px;">
        <div style="font-size:12px; font-weight:700; color:#64748B; margin-bottom:6px;">👑 DEĞERLİ METAL PİYASA PSİKOLOJİSİ (ALTIN/GÜMÜŞ)</div>
        <div style="background:#1E293B; height:8px; border-radius:4px; overflow:hidden; margin-bottom:10px;"><div style="background:{b_clr}; width:100%; height:8px;"></div></div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="font-size:12px; color:#94A3B8;">⚡ Volatilite (ATR): <b style="color:{v_clr};">{vol}</b> • 💵 Durum: <b style="color:#FFF;">{durum}</b></div>
            <div style="color:{b_clr}; font-weight:800; font-size:15px;">{hac}</div>
        </div>
    </div>""".format(b_clr=e_bar_color, v_clr=e_vol_clr, vol=e_vol, hac=e_hac, durum=p_durum)
    st.markdown(html_single_e, unsafe_allow_html=True)
    aktif_list = [m for m in TUM_MARKETLER if m["category"] == "Emtia"]

elif secilen_sayfa == "🇹🇷 BIST Terminali":
    aktif_list = [m for m in TUM_MARKETLER if m["category"] == "BIST"]

elif secilen_sayfa == "✨ Özel İzleme Listesi":
    aktif_list = st.session_state.ozel_semboller

else:
    aktif_list = []

# --- TRADINGVIEW MODAL MOTORU ---
if st.session_state.chart_open:
    c_market = st.session_state.chart_open
    st.markdown("### 📊 Canlı Grafik: {} ({})".format(c_market['name'], c_market['tv']))
    tv_interval_map = {"1h": "60", "4h": "240", "1d": "D"}
    secilen_tv_interval = tv_interval_map.get(global_interval, "60")
    html_code = f"""<div id="tv-chart-container" style="height:450px;"></div><script src="https://s3.tradingview.com/tv.js"></script><script>new TradingView.widget({{"autosize":true,"symbol":"{c_market['tv']}","interval":"{secilen_tv_interval}","timezone":"Europe/Istanbul","theme":"dark","style":"1","locale":"tr","container_id":"tv-chart-container","studies":["RSI@tv-basicstudies","MAExp@tv-basicstudies","MACD@tv-basicstudies","BB@tv-basicstudies"],"disabled_features":["header_saveload"]}});</script>"""
    st.components.v1.html(html_code, height=460)
    if st.button("❌ Grafiği Kapat"):
        st.session_state.chart_open = None
        st.rerun()
    st.markdown("---")

sayfa_adi_temiz = secilen_sayfa.split(' ', 1)[-1]

if st.button(f"🚀 {sayfa_adi_temiz} İçin Sinyal Taraması Başlat"):
    if not aktif_list:
        st.warning("Bu listede taranacak sembol bulunamadı.")
    else:
        st.info("⚡ Asenkron (Paralel) Tarama başlatıldı...")
        ilerleme_cubugu = st.progress(0)
        durum_metni = st.empty()
        
        yeni_sonuclar = {}
        toplam_varlik = len(aktif_list)
        tamamlanan = 0
        
        aktif_forced_state = "force_past" in st.session_state and st.session_state.force_past
        
        def piyasa_isle(m):
            return m, analiz_et_safe(m, global_min_hours, global_interval, global_doji_modu, aktif_forced_state)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            gelecek_gorevler = [executor.submit(piyasa_isle, m) for m in aktif_list]
            
            for future in concurrent.futures.as_completed(gelecek_gorevler):
                m_data, analiz_sonucu = future.result()
                if analiz_sonucu: 
                    yeni_sonuclar[m_data["symbol"]] = {"market": m_data, "result": analiz_sonucu}
                    
                tamamlanan += 1
                durum_metni.markdown(f"**⚡ Paralel İşleniyor:** {tamamlanan}/{toplam_varlik} piyasa tamamlandı.")
                ilerleme_cubugu.progress(tamamlanan / toplam_varlik)
                
        durum_metni.success("✅ Tarama tamamlandı! Rozetli sinyaller aşağıda listeleniyor.")
        st.session_state.results = yeni_sonuclar
        st.rerun()

# --- SİNYAL KARTLARI ---
ham_sinyaller = {k: v for k, v in st.session_state.results.items() if v["market"] in aktif_list}
if st.session_state.strict_mode:
    ham_sinyaller = {k: v for k, v in ham_sinyaller.items() if v["result"].get("skor", 0) >= 5}

valid_signals = dict(
    sorted(
        ham_sinyaller.items(), 
        key=lambda x: max(x[1]["result"].get("reboundPct", 0.0), x[1]["result"].get("drawdownPct", 0.0)), 
        reverse=True
    )
)

if not valid_signals:
    st.info("Piyasada 'olgun ve taze' bir Doji bulunamadı. Sol menüden 'Zaman Filtresini Kaldır' seçeneğini açarak daha eski sinyallere bakabilirsin.")
else:
    for sym, data in valid_signals.items():
        m, r = data["market"], data["result"]
        is_buy = r["signal"] == "BUY"
        is_confluence = (is_buy and r["bigTrend"] == "Boğa (Yukarı)") or (not is_buy and r["bigTrend"] == "Ayı (Aşağı)")
        
        skor_seviyesi = r.get("skor", 0)
        if skor_seviyesi >= 7: rozet_metni, rozet_ikon = "Elmas Rozet (Kusursuz Fırsat)", "💎"
        elif skor_seviyesi >= 5: rozet_metni, rozet_ikon = "Altın Rozet (Güçlü Sinyal)", "🥇"
        elif skor_seviyesi >= 3: rozet_metni, rozet_ikon = "Gümüş Rozet (Orta Potansiyel)", "🥈"
        else: rozet_metni, rozet_ikon = "Bronz Rozet (İzleme Amaçlı)", "🥉"
        
        with st.container(border=True):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.subheader(f"{m['name']}  :{'green' if is_buy else 'red'}[{r['signal']}]")
                st.write(f"{rozet_ikon} **Skor:** {skor_seviyesi}/10 — {rozet_metni}")
                st.write(f"⏳ **Doji Yaşı:** {r['hoursAgo']} Mum Önce ({r['dojiType']} Doji)")
                st.caption(f"**Büyük Trend (4h):** {r['bigTrend']} | {'🔥 Uyumlu' if is_confluence else '⚠️ Trend Tersi Riskli'}")
                
                if not is_buy:
                    rb = r.get("reboundPct", 0.0)
                    if rb >= 2.0: st.error(f"🚨 Aşırı Tepe Fırsatı (+%{rb:.2f}) - Güçlü Satış Fırsatı")
                    elif rb >= 1.0: st.error(f"🔥 Güçlü Tepe Fırsatı (+%{rb:.2f})")
                    elif rb >= 0.4: st.warning(f"⚠️ Orta Tepe Fırsatı (+%{rb:.2f})")
                    elif rb > 0: st.info(f"💤 Zayıf Tepe Fırsatı (+%{rb:.2f})")
                    
                    if r.get("yapisalShortGuclu", False):
                        st.markdown("""
                        <div style='background-color: rgba(239, 68, 68, 0.15); padding: 10px; border-radius: 6px; border-left: 4px solid #EF4444; margin-top: 8px; margin-bottom: 5px;'>
                            <span style='color: #F87171; font-weight: 700; font-size: 13px;'>🎯 YAPISAL BLOK AKTİF (SHORT GÜÇLÜ)</span><br>
                            <span style='color: #CBD5E1; font-size: 11px;'>Sinyal mumunun en düşüğü kırılmadı veya altında gövde kapanışı gelmedi. Fiyat yukarı esnedikçe SHORT yönlü baskı ve R/R oranı maksimize oluyor! (+2 Ek Skor uygulandı)</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    dd = r.get("drawdownPct", 0.0)
                    if dd >= 2.0: st.success(f"🚨 Aşırı Dip Fırsatı (-%{dd:.2f}) - Güçlü Alış Fırsatı")
                    elif dd >= 1.0: st.success(f"🔥 Güçlü Dip Fırsatı (-%{dd:.2f})")
                    elif dd >= 0.4: st.info(f"⚠️ Orta Dip Fırsatı (-%{dd:.2f})")
                    elif dd > 0: st.write(f"💤 Zayıf Dip Fırsatı (-%{dd:.2f})")
                    
                    if r.get("yapisalLongGuclu", False):
                        st.markdown("""
                        <div style='background-color: rgba(16, 185, 129, 0.15); padding: 10px; border-radius: 6px; border-left: 4px solid #10B981; margin-top: 8px; margin-bottom: 5px;'>
                            <span style='color: #34D399; font-weight: 700; font-size: 13px;'>🎯 YAPISAL LONG GÜÇLENMESİ AKTİF</span><br>
                            <span style='color: #CBD5E1; font-size: 11px;'>Sinyal mumunun tepesi yukarı kırılmadı veya üstünde kapanış gelmedi. Fiyat aşağı esnedikçe LONG yönlü maliyetlenme avantajı artıyor! (+2 Ek Skor uygulandı)</span>
                        </div>
                        """, unsafe_allow_html=True)
                
                if "topFeatures" in r and r["topFeatures"]:
                    feat_str = " • ".join([f"{k}: %{v:.1f}" for k, v in r["topFeatures"].items()])
                    st.caption(f"🧠 **Model Karar Etkenleri:** {feat_str}")

            with col2:
                st.metric(
                    label="Model Güveni", 
                    value=f"%{int(r['confidence'])}"
                )
                st.metric(
                    label="Fold Testi Win-Rate", 
                    value=f"%{int(r['winRate'])}", 
                    help="Bu başarı oranı rastgele değil; K-Fold TimeSeries yöntemiyle geçmişteki 5 farklı piyasa koşulunda çapraz test edilerek hesaplanmıştır."
                )
                
            with col3:
                st.metric("Güncel Fiyat", f"{r['price']:,.2f}", f"{r['change']:.2f}%")
                
            if st.button(f"📊 Grafiği İncele", key=f"chart_btn_{m['symbol']}_{m['category']}"):
                st.session_state.chart_open = m
                st.rerun()
