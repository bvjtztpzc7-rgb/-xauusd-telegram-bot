import os
import requests
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURAZIONE
# ============================================================

API_KEY = os.getenv("TWELVE_DATA_API_KEY")

if not API_KEY:
    raise RuntimeError("TWELVE_DATA_API_KEY non configurato")


# ============================================================
# SCARICA DATI
# ============================================================

def scarica_dati():
    url = "https://api.twelvedata.com/time_series"

    def richiesta(params):
        response = requests.get(url, params=params, timeout=30)
        data = response.json()

        if "status" in data and data["status"] == "error":
            raise RuntimeError(f"Errore Twelve Data: {data}")

        if "values" not in data:
            raise RuntimeError(f"Risposta Twelve Data inattesa: {data}")

        return data["values"]

    # ========================================================
    # BLOCCO 1 — ultime 5000 candele
    # ========================================================

    params1 = {
        "symbol": "XAU/USD",
        "interval": "5min",
        "outputsize": 5000,
        "apikey": API_KEY,
        "format": "JSON",
        "timezone": "UTC",
        "order": "desc"
    }

    valori1 = richiesta(params1)

    df1 = pd.DataFrame(valori1)

    df1["datetime"] = pd.to_datetime(
        df1["datetime"],
        utc=True
    )

    df1 = df1[
        ["datetime", "open", "high", "low", "close"]
    ].copy()

    for col in ["open", "high", "low", "close"]:
        df1[col] = pd.to_numeric(
            df1[col],
            errors="coerce"
        )

    df1 = df1.dropna()

    # Troviamo la candela più vecchia del primo blocco
    data_piu_vecchia = df1["datetime"].min()

    # ========================================================
    # BLOCCO 2 — 5000 candele precedenti
    # ========================================================

    data_fine_secondo_blocco = (
        data_piu_vecchia - pd.Timedelta(minutes=5)
    )

    params2 = {
        "symbol": "XAU/USD",
        "interval": "5min",
        "outputsize": 5000,
        "apikey": API_KEY,
        "format": "JSON",
        "timezone": "UTC",
        "end_date": data_fine_secondo_blocco.strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),
        "order": "desc"
    }

    valori2 = richiesta(params2)

    df2 = pd.DataFrame(valori2)

    df2["datetime"] = pd.to_datetime(
        df2["datetime"],
        utc=True
    )

    df2 = df2[
        ["datetime", "open", "high", "low", "close"]
    ].copy()

    for col in ["open", "high", "low", "close"]:
        df2[col] = pd.to_numeric(
            df2[col],
            errors="coerce"
        )

    df2 = df2.dropna()

    # ========================================================
    # UNIONE DEI DUE BLOCCHI
    # ========================================================

    df = pd.concat(
        [df1, df2],
        ignore_index=True
    )

    # Elimina eventuali duplicati
    df = df.drop_duplicates(
        subset=["datetime"]
    )

    # Ordine cronologico
    df = df.sort_values(
        "datetime"
    ).reset_index(drop=True)

    print(
        f"Candele scaricate: {len(df)}"
    )

    print(
        f"Da: {df['datetime'].iloc[0]}"
    )

    print(
        f"A: {df['datetime'].iloc[-1]}"
    )

    return df


# ============================================================
# INDICATORI
# ============================================================

def calcola_indicatori(df):

    df = df.copy()

    df["ema20"] = df["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    df["ema50"] = df["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    ema12 = df["close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = df["close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["macd"] = ema12 - ema26

    df["macd_signal"] = df["macd"].ewm(
        span=9,
        adjust=False
    ).mean()

    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    df["rsi"] = 100 - (
        100 / (1 + rs)
    )

    high_low = df["high"] - df["low"]

    high_close = (
        df["high"] -
        df["close"].shift()
    ).abs()

    low_close = (
        df["low"] -
        df["close"].shift()
    ).abs()

    true_range = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(axis=1)

    df["atr"] = true_range.rolling(14).mean()

    df["macd_prev"] = df["macd"].shift(1)
    df["signal_prev"] = df["macd_signal"].shift(1)

    return df


# ============================================================
# TENDENZA 15 MINUTI
# ============================================================

def aggiungi_trend_15m(df):

    temp = df.set_index("datetime")

    df15 = temp.resample("15min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna()

    df15["ema20"] = df15["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    df15["ema50"] = df15["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    df15["trend"] = np.where(
        df15["ema20"] > df15["ema50"],
        "BULLISH",
        "BEARISH"
    )

    result = df.copy()

    result["trend_15m"] = None

    for i in range(len(result)):

        timestamp = result.loc[i, "datetime"]

        periodo = timestamp.floor("15min")

        periodi_precedenti = df15.index[
            df15.index < periodo
        ]

        if len(periodi_precedenti) == 0:
            continue

        ultimo_periodo = periodi_precedenti[-1]

        result.loc[i, "trend_15m"] = df15.loc[
            ultimo_periodo,
            "trend"
        ]

    return result


# ============================================================
# GENERAZIONE SEGNALE
# ============================================================

def genera_segnale(row, strategia):

    if pd.isna(row["atr"]):
        return None

    trend = row["trend_15m"]

    # ========================================================
    # STRATEGIA A
    # ========================================================

    if strategia == "A":

        buy = (
            row["ema20"] > row["ema50"]
            and
            row["macd"] > row["macd_signal"]
            and
            30 < row["rsi"] < 65
            and
            trend == "BULLISH"
        )

        sell = (
            row["ema20"] < row["ema50"]
            and
            row["macd"] < row["macd_signal"]
            and
            30 < row["rsi"] < 65
            and
            trend == "BEARISH"
        )

    # ========================================================
    # STRATEGIA B
    # ========================================================

    elif strategia == "B":

        buy = (
            row["ema20"] > row["ema50"]
            and
            row["macd"] > row["macd_signal"]
            and
            30 < row["rsi"] < 70
            and
            trend == "BULLISH"
        )

        sell = (
            row["ema20"] < row["ema50"]
            and
            row["macd"] < row["macd_signal"]
            and
            30 < row["rsi"] < 70
            and
            trend == "BEARISH"
        )

    # ========================================================
    # STRATEGIA C
    # ========================================================

    elif strategia == "C":

        bullish_cross = (
            row["macd_prev"] <= row["signal_prev"]
            and
            row["macd"] > row["macd_signal"]
        )

        bearish_cross = (
            row["macd_prev"] >= row["signal_prev"]
            and
            row["macd"] < row["macd_signal"]
        )

        buy = (
            row["ema20"] > row["ema50"]
            and
            bullish_cross
            and
            30 < row["rsi"] < 70
            and
            trend == "BULLISH"
        )

        sell = (
            row["ema20"] < row["ema50"]
            and
            bearish_cross
            and
            30 < row["rsi"] < 70
            and
            trend == "BEARISH"
        )

    else:
        return None

    if buy:
        return "BUY"

    if sell:
        return "SELL"

    return None


# ============================================================
# # ============================================================
# BACKTEST
# ============================================================

def esegui_backtest(df, strategia):

    operazioni = []

    posizione = None

    # ========================================================
    # CONTATORI DIAGNOSTICI
    # ========================================================

    conteggi = {
        "EMA": 0,
        "MACD": 0,
        "RSI": 0,
        "TREND": 0,
        "EMA_MACD": 0,
        "EMA_MACD_RSI": 0,
        "EMA_MACD_RSI_TREND": 0
    }

    for i in range(1, len(df) - 1):

        row = df.iloc[i]

        # ----------------------------------------------------
        # POSIZIONE APERTA
        # ----------------------------------------------------

        if posizione is not None:

            high = row["high"]
            low = row["low"]

            entry = posizione["entry"]
            sl = posizione["sl"]
            tp = posizione["tp"]

            risultato = None
            exit_price = None

            if posizione["tipo"] == "BUY":

                if low <= sl:
                    risultato = "SL"
                    exit_price = sl

                elif high >= tp:
                    risultato = "TP"
                    exit_price = tp

            else:

                if high >= sl:
                    risultato = "SL"
                    exit_price = sl

                elif low <= tp:
                    risultato = "TP"
                    exit_price = tp

            if risultato is not None:

                if posizione["tipo"] == "BUY":
                    profitto = exit_price - entry
                else:
                    profitto = entry - exit_price

                operazioni.append({
                    "tipo": posizione["tipo"],
                    "risultato": risultato,
                    "profitto": profitto,
                    "datetime": posizione["datetime"]
                })

                posizione = None

            continue

        # ----------------------------------------------------
        # CONDIZIONI DIAGNOSTICHE
        # ----------------------------------------------------

        if pd.isna(row["atr"]):
            continue

        trend = row["trend_15m"]

        ema_bull = row["ema20"] > row["ema50"]
        ema_bear = row["ema20"] < row["ema50"]

        macd_bull = row["macd"] > row["macd_signal"]
        macd_bear = row["macd"] < row["macd_signal"]

        if strategia == "A":
            rsi_ok = 30 < row["rsi"] < 65

        elif strategia == "B":
            rsi_ok = 30 < row["rsi"] < 70

        elif strategia == "C":
            continue

        else:
            continue

        # ----------------------------------------------------
        # CONTROLLIAMO LE CONDIZIONI BUY
        # ----------------------------------------------------

        if ema_bull:
            conteggi["EMA"] += 1

        if macd_bull:
            conteggi["MACD"] += 1

        if rsi_ok:
            conteggi["RSI"] += 1

        if trend == "BULLISH":
            conteggi["TREND"] += 1

        if ema_bull and macd_bull:
            conteggi["EMA_MACD"] += 1

        if (
            ema_bull
            and macd_bull
            and rsi_ok
        ):
            conteggi["EMA_MACD_RSI"] += 1

        if (
            ema_bull
            and macd_bull
            and rsi_ok
            and trend == "BULLISH"
        ):
            conteggi["EMA_MACD_RSI_TREND"] += 1

        # ----------------------------------------------------
        # SEGNALE
        # ----------------------------------------------------

        segnale = genera_segnale(
            row,
            strategia
        )

        if segnale is None:
            continue

        entry = df.iloc[i + 1]["open"]

        atr = row["atr"]

        if segnale == "BUY":

            sl = entry - (
                1.5 * atr
            )

            tp = entry + (
                2.0 * atr
            )

        else:

            sl = entry + (
                1.5 * atr
            )

            tp = entry - (
                2.0 * atr
            )

        posizione = {
            "tipo": segnale,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "datetime": df.iloc[i + 1]["datetime"]
        }

    # ========================================================
    # STAMPA DIAGNOSTICA
    # ========================================================

    print()
    print("🔎 DIAGNOSTICA BUY — STRATEGIA", strategia)
    print("-" * 50)
    print("EMA20 > EMA50:", conteggi["EMA"])
    print("MACD > SIGNAL:", conteggi["MACD"])
    print("RSI OK:", conteggi["RSI"])
    print("TREND BULLISH:", conteggi["TREND"])
    print("EMA + MACD:", conteggi["EMA_MACD"])
    print("EMA + MACD + RSI:", conteggi["EMA_MACD_RSI"])
    print(
        "EMA + MACD + RSI + TREND:",
        conteggi["EMA_MACD_RSI_TREND"]
    )
    print("-" * 50)

    return operazioni

# ============================================================
# METRICHE
# ============================================================

def calcola_metriche(operazioni):

    if not operazioni:

        return {
            "totale": 0,
            "tp": 0,
            "sl": 0,
            "win_rate": 0,
            "profitto": 0,
            "profit_factor": 0,
            "media_win": 0,
            "media_loss": 0,
            "max_drawdown": 0,
            "max_win_streak": 0,
            "max_loss_streak": 0,
            "buy": 0,
            "buy_tp": 0,
            "buy_sl": 0,
            "sell": 0,
            "sell_tp": 0,
            "sell_sl": 0
        }

    totale = len(operazioni)

    vincenti = [
        x["profitto"]
        for x in operazioni
        if x["risultato"] == "TP"
    ]

    perdenti = [
        x["profitto"]
        for x in operazioni
        if x["risultato"] == "SL"
    ]

    tp = len(vincenti)
    sl = len(perdenti)

    win_rate = (
        tp / totale
    ) * 100

    profitto = sum(
        x["profitto"]
        for x in operazioni
    )

    gross_profit = sum(vincenti)

    gross_loss = abs(
        sum(perdenti)
    )

    if gross_loss > 0:
        profit_factor = (
            gross_profit /
            gross_loss
        )
    else:
        profit_factor = float("inf")

    media_win = (
        np.mean(vincenti)
        if vincenti
        else 0
    )

    media_loss = (
        np.mean(perdenti)
        if perdenti
        else 0
    )

    # --------------------------------------------------------
    # DRAWDOWN
    # --------------------------------------------------------

    capitale = 0
    massimo = 0
    max_drawdown = 0

    for op in operazioni:

        capitale += op["profitto"]

        if capitale > massimo:
            massimo = capitale

        drawdown = massimo - capitale

        if drawdown > max_drawdown:
            max_drawdown = drawdown

    # --------------------------------------------------------
    # SERIE
    # --------------------------------------------------------

    max_win_streak = 0
    max_loss_streak = 0

    win_streak = 0
    loss_streak = 0

    for op in operazioni:

        if op["risultato"] == "TP":

            win_streak += 1
            loss_streak = 0

        else:

            loss_streak += 1
            win_streak = 0

        max_win_streak = max(
            max_win_streak,
            win_streak
        )

        max_loss_streak = max(
            max_loss_streak,
            loss_streak
        )

    # --------------------------------------------------------
    # BUY / SELL
    # --------------------------------------------------------

    buy_ops = [
        x for x in operazioni
        if x["tipo"] == "BUY"
    ]

    sell_ops = [
        x for x in operazioni
        if x["tipo"] == "SELL"
    ]

    buy_tp = sum(
        x["risultato"] == "TP"
        for x in buy_ops
    )

    buy_sl = sum(
        x["risultato"] == "SL"
        for x in buy_ops
    )

    sell_tp = sum(
        x["risultato"] == "TP"
        for x in sell_ops
    )

    sell_sl = sum(
        x["risultato"] == "SL"
        for x in sell_ops
    )

    return {
        "totale": totale,
        "tp": tp,
        "sl": sl,
        "win_rate": win_rate,
        "profitto": profitto,
        "profit_factor": profit_factor,
        "media_win": media_win,
        "media_loss": media_loss,
        "max_drawdown": max_drawdown,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "buy": len(buy_ops),
        "buy_tp": buy_tp,
        "buy_sl": buy_sl,
        "sell": len(sell_ops),
        "sell_tp": sell_tp,
        "sell_sl": sell_sl
    }


# ============================================================
# STAMPA RISULTATI
# ============================================================

def stampa_risultati(codice, nome, metriche):

    print()
    print("=" * 65)

    print(f"📌 STRATEGIA {codice}")
    print(nome)

    print("-" * 65)

    print(
        f"Operazioni: {metriche['totale']}"
    )

    print(
        f"TP: {metriche['tp']}"
    )

    print(
        f"SL: {metriche['sl']}"
    )

    print(
        f"Win rate: "
        f"{metriche['win_rate']:.2f}%"
    )

    print(
        f"Risultato prezzo: "
        f"{metriche['profitto']:.2f}"
    )

    print(
        f"Profit Factor: "
        f"{metriche['profit_factor']:.2f}"
    )

    print(
        f"Media vincita: "
        f"{metriche['media_win']:.2f}"
    )

    print(
        f"Media perdita: "
        f"{metriche['media_loss']:.2f}"
    )

    print(
        f"Drawdown massimo: "
        f"{metriche['max_drawdown']:.2f}"
    )

    print(
        f"Max serie vittorie: "
        f"{metriche['max_win_streak']}"
    )

    print(
        f"Max serie perdite: "
        f"{metriche['max_loss_streak']}"
    )

    print()
    print("📈 BUY")

    print(
        f"Operazioni: {metriche['buy']}"
    )

    print(
        f"TP: {metriche['buy_tp']}"
    )

    print(
        f"SL: {metriche['buy_sl']}"
    )

    print()
    print("📉 SELL")

    print(
        f"Operazioni: {metriche['sell']}"
    )

    print(
        f"TP: {metriche['sell_tp']}"
    )

    print(
        f"SL: {metriche['sell_sl']}"
    )


# ============================================================
# ESECUZIONE SU UN PERIODO
# ============================================================

def analizza_periodo(df, titolo):

    print()
    print("#" * 65)
    print(f"📊 {titolo}")
    print("#" * 65)

    print(
        f"Candele: {len(df)}"
    )

    print(
        f"Da: {df.iloc[0]['datetime']}"
    )

    print(
        f"A: {df.iloc[-1]['datetime']}"
    )

    strategie = {
        "A": "Strategia attuale — RSI 30-65",
        "B": "RSI ampliato — RSI 30-70",
        "C": "Incrocio MACD — RSI 30-70"
    }

    for codice, nome in strategie.items():

        operazioni = esegui_backtest(
            df,
            codice
        )

        metriche = calcola_metriche(
            operazioni
        )

        stampa_risultati(
            codice,
            nome,
            metriche
        )


# ============================================================
# PRINCIPALE
# ============================================================

print()
print("=" * 65)
print("🤖 XAU/USD — BACKTEST PAPER/DEMO")
print("📊 TEST SVILUPPO 70% / VERIFICA 30%")
print("=" * 65)

df = scarica_dati()

print()
print(
    f"📊 Candele totali scaricate: {len(df)}"
)

print(
    f"📅 Periodo completo: "
    f"{df.iloc[0]['datetime']} → "
    f"{df.iloc[-1]['datetime']}"
)

# ============================================================
# INDICATORI
# ============================================================

df = calcola_indicatori(df)

df = aggiungi_trend_15m(df)

# Eliminiamo le righe iniziali senza indicatori
df = df.dropna(
    subset=[
        "ema20",
        "ema50",
        "macd",
        "macd_signal",
        "rsi",
        "atr"
    ]
).reset_index(drop=True)

# ============================================================
# DIVISIONE 70 / 30
# ============================================================

punto_divisione = int(
    len(df) * 0.70
)

df_sviluppo = df.iloc[
    :punto_divisione
].copy()

df_verifica = df.iloc[
    punto_divisione:
].copy()

# ============================================================
# SVILUPPO
# ============================================================

analizza_periodo(
    df_sviluppo,
    "FASE 1 — SVILUPPO 70%"
)

# ============================================================
# VERIFICA
# ============================================================

analizza_periodo(
    df_verifica,
    "FASE 2 — VERIFICA 30%"
)

# ============================================================
# FINE
# ============================================================

print()
print("=" * 65)
print("✅ BACKTEST TERMINATO")
print("=" * 65)

print()
print(
    "⚠️ Simulazione storica PAPER/DEMO: "
    "non rappresenta risultati finanziari reali "
    "e non considera tutti i costi di esecuzione."
)
