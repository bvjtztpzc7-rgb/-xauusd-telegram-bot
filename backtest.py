import os
import requests
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURAZIONE
# ============================================================

API_KEY = os.getenv("TWELVE_DATA_API_KEY")

if not API_KEY:
    raise RuntimeError("TWELVE_DATA_API_KEY non configurata")


# ============================================================
# PARAMETRI STRATEGIA ATTUALE
# ============================================================

SL_ATR = 1.5
TP_ATR = 2.5

RSI_MIN = 30
RSI_MAX = 65


# ============================================================
# SCARICA DATI
# ============================================================

def scarica_dati():

    url = "https://api.twelvedata.com/time_series"

    def richiesta(params):

        response = requests.get(
            url,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        if data.get("status") == "error":
            raise RuntimeError(
                f"Errore Twelve Data: {data}"
            )

        if "values" not in data:
            raise RuntimeError(
                f"Risposta Twelve Data inattesa: {data}"
            )

        return data["values"]

    # --------------------------------------------------------
    # BLOCCO 1
    # --------------------------------------------------------

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

    for col in ["open", "high", "low", "close"]:

        df1[col] = pd.to_numeric(
            df1[col],
            errors="coerce"
        )

    df1 = df1[
        [
            "datetime",
            "open",
            "high",
            "low",
            "close"
        ]
    ].dropna()

    data_piu_vecchia = df1["datetime"].min()

    # --------------------------------------------------------
    # BLOCCO 2
    # --------------------------------------------------------

    data_fine_secondo_blocco = (
        data_piu_vecchia -
        pd.Timedelta(minutes=5)
    )

    params2 = {
        "symbol": "XAU/USD",
        "interval": "5min",
        "outputsize": 5000,
        "apikey": API_KEY,
        "format": "JSON",
        "timezone": "UTC",
        "end_date":
            data_fine_secondo_blocco.strftime(
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

    for col in ["open", "high", "low", "close"]:

        df2[col] = pd.to_numeric(
            df2[col],
            errors="coerce"
        )

    df2 = df2[
        [
            "datetime",
            "open",
            "high",
            "low",
            "close"
        ]
    ].dropna()

    # --------------------------------------------------------
    # UNIONE
    # --------------------------------------------------------

    df = pd.concat(
        [df1, df2],
        ignore_index=True
    )

    df = (
        df
        .drop_duplicates("datetime")
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    print()
    print(f"📊 Candele scaricate: {len(df)}")
    print(f"📅 Da: {df['datetime'].iloc[0]}")
    print(f"📅 A:  {df['datetime'].iloc[-1]}")

    return df


# ============================================================
# INDICATORI
# ============================================================

def calcola_indicatori(df):

    df = df.copy()

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    df["ema20"] = df["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    df["ema50"] = df["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # RSI 14
    # --------------------------------------------------------

    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    df["rsi"] = 100 - (
        100 / (1 + rs)
    )

    # --------------------------------------------------------
    # ATR 14
    # --------------------------------------------------------

    high_low = (
        df["high"] -
        df["low"]
    )

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

    # --------------------------------------------------------
    # MACD precedente
    # --------------------------------------------------------

    df["macd_prev"] = df["macd"].shift(1)

    df["signal_prev"] = (
        df["macd_signal"].shift(1)
    )

    # --------------------------------------------------------
    # CARATTERISTICHE DELLA CANDELA
    # --------------------------------------------------------

    df["candle_range"] = (
        df["high"] -
        df["low"]
    )

    df["candle_body"] = (
        df["close"] -
        df["open"]
    ).abs()

    df["body_ratio"] = np.where(
        df["candle_range"] > 0,
        df["candle_body"] /
        df["candle_range"],
        0
    )

    # --------------------------------------------------------
    # MOVIMENTI PRECEDENTI
    # --------------------------------------------------------

    df["move_1"] = (
        df["close"] -
        df["close"].shift(1)
    )

    df["move_3"] = (
        df["close"] -
        df["close"].shift(3)
    )

    df["move_5"] = (
        df["close"] -
        df["close"].shift(5)
    )

    # --------------------------------------------------------
    # DISTANZE INDICATORI
    # --------------------------------------------------------

    df["ema_distance"] = (
        df["ema20"] -
        df["ema50"]
    )

    df["ema_distance_atr"] = np.where(
        df["atr"] > 0,
        abs(df["ema_distance"]) /
        df["atr"],
        np.nan
    )

    df["macd_distance"] = (
        df["macd"] -
        df["macd_signal"]
    )

    return df


# ============================================================
# TREND 15 MINUTI
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

        timestamp = result.loc[
            i,
            "datetime"
        ]

        periodo = timestamp.floor("15min")

        periodi_precedenti = df15.index[
            df15.index < periodo
        ]

        if len(periodi_precedenti) == 0:
            continue

        ultimo_periodo = (
            periodi_precedenti[-1]
        )

        result.loc[
            i,
            "trend_15m"
        ] = df15.loc[
            ultimo_periodo,
            "trend"
        ]

    return result


# ============================================================
# GENERA SEGNALE
# ============================================================

def genera_segnale(row):

    if pd.isna(row["atr"]):
        return None

    if pd.isna(row["trend_15m"]):
        return None

    ema_bull = (
        row["ema20"] >
        row["ema50"]
    )

    ema_bear = (
        row["ema20"] <
        row["ema50"]
    )

    macd_bull = (
        row["macd"] >
        row["macd_signal"]
    )

    macd_bear = (
        row["macd"] <
        row["macd_signal"]
    )

    rsi_ok = (
        RSI_MIN <
        row["rsi"] <
        RSI_MAX
    )

    trend_bull = (
        row["trend_15m"] == "BULLISH"
    )

    trend_bear = (
        row["trend_15m"] == "BEARISH"
    )

    if (
        ema_bull
        and macd_bull
        and rsi_ok
        and trend_bull
    ):
        return "BUY"

    if (
        ema_bear
        and macd_bear
        and rsi_ok
        and trend_bear
    ):
        return "SELL"

    return None


# ============================================================
# BACKTEST AVANZATO
# ============================================================

def esegui_backtest(df):

    operazioni = []

    posizione = None

    casi_ambigui = 0

    for i in range(1, len(df) - 1):

        row = df.iloc[i]

        # ====================================================
        # GESTIONE POSIZIONE
        # ====================================================

        if posizione is not None:

            high = float(row["high"])
            low = float(row["low"])

            entry = posizione["entry"]
            sl = posizione["sl"]
            tp = posizione["tp"]

            risultato = None
            exit_price = None

            # ------------------------------------------------
            # BUY
            # ------------------------------------------------

            if posizione["tipo"] == "BUY":

                tocca_sl = low <= sl
                tocca_tp = high >= tp

                if tocca_sl and tocca_tp:

                    # Con dati OHLC non possiamo sapere
                    # quale livello sia stato raggiunto prima.
                    casi_ambigui += 1

                    risultato = "SL/TP_AMBIGUO"
                    exit_price = sl

                elif tocca_sl:

                    risultato = "SL"
                    exit_price = sl

                elif tocca_tp:

                    risultato = "TP"
                    exit_price = tp

            # ------------------------------------------------
            # SELL
            # ------------------------------------------------

            else:

                tocca_sl = high >= sl
                tocca_tp = low <= tp

                if tocca_sl and tocca_tp:

                    casi_ambigui += 1

                    risultato = "SL/TP_AMBIGUO"
                    exit_price = sl

                elif tocca_sl:

                    risultato = "SL"
                    exit_price = sl

                elif tocca_tp:

                    risultato = "TP"
                    exit_price = tp

            # ------------------------------------------------
            # CHIUSURA
            # ------------------------------------------------

            if risultato is not None:

                if posizione["tipo"] == "BUY":

                    profitto = (
                        exit_price -
                        entry
                    )

                else:

                    profitto = (
                        entry -
                        exit_price
                    )

                operazioni.append({
                    "tipo": posizione["tipo"],
                    "risultato": risultato,
                    "profitto": float(profitto),
                    "datetime": posizione["datetime"],

                    "entry": posizione["entry"],
                    "sl": posizione["sl"],
                    "tp": posizione["tp"],

                    "rsi": posizione["rsi"],
                    "atr": posizione["atr"],

                    "ema20": posizione["ema20"],
                    "ema50": posizione["ema50"],
                    "ema_distance":
                        posizione["ema_distance"],
                    "ema_distance_atr":
                        posizione["ema_distance_atr"],

                    "macd": posizione["macd"],
                    "macd_signal":
                        posizione["macd_signal"],
                    "macd_distance":
                        posizione["macd_distance"],

                    "trend_15m":
                        posizione["trend_15m"],

                    "candle_range":
                        posizione["candle_range"],
                    "body_ratio":
                        posizione["body_ratio"],

                    "move_1":
                        posizione["move_1"],
                    "move_3":
                        posizione["move_3"],
                    "move_5":
                        posizione["move_5"]
                })

                posizione = None

            continue

        # ====================================================
        # DATI NON VALIDI
        # ====================================================

        if pd.isna(row["atr"]):
            continue

        if pd.isna(row["rsi"]):
            continue

        if pd.isna(row["trend_15m"]):
            continue

        # ====================================================
        # SEGNALE
        # ====================================================

        segnale = genera_segnale(row)

        if segnale is None:
            continue

        # ====================================================
        # ENTRATA
        # ====================================================

        next_candle = df.iloc[i + 1]

        entry = float(
            next_candle["open"]
        )

        atr = float(row["atr"])

        # ====================================================
        # SL / TP
        # ====================================================

        if segnale == "BUY":

            sl = (
                entry -
                SL_ATR * atr
            )

            tp = (
                entry +
                TP_ATR * atr
            )

        else:

            sl = (
                entry +
                SL_ATR * atr
            )

            tp = (
                entry -
                TP_ATR * atr
            )

        # ====================================================
        # SALVA POSIZIONE
        # ====================================================

        posizione = {

            "tipo": segnale,

            "entry": entry,
            "sl": sl,
            "tp": tp,

            "datetime":
                next_candle["datetime"],

            "rsi":
                float(row["rsi"]),

            "atr":
                atr,

            "ema20":
                float(row["ema20"]),

            "ema50":
                float(row["ema50"]),

            "ema_distance":
                float(row["ema_distance"]),

            "ema_distance_atr":
                float(row["ema_distance_atr"]),

            "macd":
                float(row["macd"]),

            "macd_signal":
                float(row["macd_signal"]),

            "macd_distance":
                float(row["macd_distance"]),

            "trend_15m":
                row["trend_15m"],

            "candle_range":
                float(row["candle_range"]),

            "body_ratio":
                float(row["body_ratio"]),

            "move_1":
                float(row["move_1"])
                if not pd.isna(row["move_1"])
                else 0,

            "move_3":
                float(row["move_3"])
                if not pd.isna(row["move_3"])
                else 0,

            "move_5":
                float(row["move_5"])
                if not pd.isna(row["move_5"])
                else 0
        }

    return pd.DataFrame(operazioni), casi_ambigui


# ============================================================
# METRICHE
# ============================================================

def calcola_metriche(trades):

    if len(trades) == 0:

        return {
            "operazioni": 0,
            "tp": 0,
            "sl": 0,
            "win_rate": 0,
            "risultato": 0,
            "profit_factor": 0,
            "drawdown": 0
        }

    validi = trades[
        trades["risultato"].isin(
            ["TP", "SL"]
        )
    ]

    tp = (
        validi["risultato"] == "TP"
    ).sum()

    sl = (
        validi["risultato"] == "SL"
    ).sum()

    totale = len(validi)

    win_rate = (
        tp / totale * 100
        if totale
        else 0
    )

    profitti = validi["profitto"].tolist()

    risultato = sum(profitti)

    vincite = [
        x for x in profitti
        if x > 0
    ]

    perdite = [
        x for x in profitti
        if x < 0
    ]

    profitto_vincite = sum(vincite)

    profitto_perdite = abs(
        sum(perdite)
    )

    if profitto_perdite > 0:

        profit_factor = (
            profitto_vincite /
            profitto_perdite
        )

    else:

        profit_factor = 0

    # --------------------------------------------------------
    # DRAWDOWN
    # --------------------------------------------------------

    equity = 0
    massimo = 0
    drawdown_max = 0

    for p in profitti:

        equity += p

        massimo = max(
            massimo,
            equity
        )

        drawdown = (
            massimo -
            equity
        )

        drawdown_max = max(
            drawdown_max,
            drawdown
        )

    return {
        "operazioni": totale,
        "tp": int(tp),
        "sl": int(sl),
        "win_rate": win_rate,
        "risultato": risultato,
        "profit_factor": profit_factor,
        "drawdown": drawdown_max
    }


# ============================================================
# ANALISI DEI GRUPPI
# ============================================================

def analizza_gruppo(
    trades,
    nome,
    maschera
):

    gruppo = trades[maschera].copy()

    if len(gruppo) == 0:

        print(
            f"{nome:<35} → nessun trade"
        )

        return

    metriche = calcola_metriche(
        gruppo
    )

    print(
        f"{nome:<35} | "
        f"N={metriche['operazioni']:>3} | "
        f"WR={metriche['win_rate']:>6.2f}% | "
        f"PF={metriche['profit_factor']:>5.2f} | "
        f"Ris={metriche['risultato']:>8.2f}"
    )


# ============================================================
# ANALISI CARATTERISTICHE
# ============================================================

def analizza_caratteristiche(trades):

    if len(trades) == 0:
        return

    print()
    print("=" * 100)
    print("🔬 ANALISI CARATTERISTICHE DEI TRADE")
    print("=" * 100)

    print()
    print("RSI")
    print("-" * 100)

    analizza_gruppo(
        trades,
        "RSI 30-40",
        trades["rsi"].between(30, 40)
    )

    analizza_gruppo(
        trades,
        "RSI 40-50",
        trades["rsi"].between(40, 50)
    )

    analizza_gruppo(
        trades,
        "RSI 50-60",
        trades["rsi"].between(50, 60)
    )

    analizza_gruppo(
        trades,
        "RSI 60-65",
        trades["rsi"].between(60, 65)
    )

    print()
    print("DISTANZA EMA / ATR")
    print("-" * 100)

    analizza_gruppo(
        trades,
        "EMA distance < 0.25 ATR",
        trades["ema_distance_atr"] < 0.25
    )

    analizza_gruppo(
        trades,
        "EMA distance 0.25-0.50 ATR",
        trades["ema_distance_atr"].between(
            0.25,
            0.50
        )
    )

    analizza_gruppo(
        trades,
        "EMA distance 0.50-1.00 ATR",
        trades["ema_distance_atr"].between(
            0.50,
            1.00
        )
    )

    analizza_gruppo(
        trades,
        "EMA distance > 1.00 ATR",
        trades["ema_distance_atr"] > 1.00
    )

    print()
    print("FORZA MACD")
    print("-" * 100)

    analizza_gruppo(
        trades,
        "MACD distance < 0.10",
        trades["macd_distance"].abs() < 0.10
    )

    analizza_gruppo(
        trades,
        "MACD distance 0.10-0.30",
        trades["macd_distance"].abs().between(
            0.10,
            0.30
        )
    )

    analizza_gruppo(
        trades,
        "MACD distance > 0.30",
        trades["macd_distance"].abs() > 0.30
    )

    print()
    print("MOVIMENTO PRECEDENTE — 5 CANDELE")
    print("-" * 100)

    analizza_gruppo(
        trades,
        "Movimento 5 negativo",
        trades["move_5"] < 0
    )

    analizza_gruppo(
        trades,
        "Movimento 5 positivo",
        trades["move_5"] > 0
    )

    print()
    print("DIMENSIONE CANDELA")
    print("-" * 100)

    analizza_gruppo(
        trades,
        "Candela < 0.50 ATR",
        trades["candle_range"] <
        trades["atr"] * 0.50
    )

    analizza_gruppo(
        trades,
        "Candela 0.50-1.00 ATR",
        trades["candle_range"].between(
            trades["atr"] * 0.50,
            trades["atr"]
        )
    )

    analizza_gruppo(
        trades,
        "Candela > 1.00 ATR",
        trades["candle_range"] >
        trades["atr"]
    )

    print()
    print("BODY RATIO")
    print("-" * 100)

    analizza_gruppo(
        trades,
        "Body < 30%",
        trades["body_ratio"] < 0.30
    )

    analizza_gruppo(
        trades,
        "Body 30-60%",
        trades["body_ratio"].between(
            0.30,
            0.60
        )
    )

    analizza_gruppo(
        trades,
        "Body > 60%",
        trades["body_ratio"] > 0.60
    )

    print()
    print("DIREZIONE")
    print("-" * 100)

    analizza_gruppo(
        trades,
        "BUY",
        trades["tipo"] == "BUY"
    )

    analizza_gruppo(
        trades,
        "SELL",
        trades["tipo"] == "SELL"
    )


# ============================================================
# STAMPA METRICHE PRINCIPALI
# ============================================================

def stampa_metriche(nome, trades):

    metriche = calcola_metriche(
        trades
    )

    print()
    print("=" * 75)
    print(f"📊 {nome}")
    print("=" * 75)

    print(
        f"Operazioni:      {metriche['operazioni']}"
    )

    print(
        f"TP:              {metriche['tp']}"
    )

    print(
        f"SL:              {metriche['sl']}"
    )

    print(
        f"Win rate:        {metriche['win_rate']:.2f}%"
    )

    print(
        f"Risultato:       {metriche['risultato']:.2f}"
    )

    print(
        f"Profit Factor:   {metriche['profit_factor']:.2f}"
    )

    print(
        f"Drawdown:        {metriche['drawdown']:.2f}"
    )


# ============================================================
# ESECUZIONE FASE
# ============================================================

def esegui_fase(
    df,
    nome
):

    print()
    print("#" * 100)
    print(f"📊 {nome}")
    print("#" * 100)

    trades, ambigui = esegui_backtest(
        df
    )

    stampa_metriche(
        nome,
        trades
    )

    print()
    print(
        f"⚠️ Candele in cui SL e TP "
        f"sono stati toccati entrambi: {ambigui}"
    )

    analizza_caratteristiche(
        trades
    )

    return trades


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 100)
    print("🤖 XAU/USD — BACKTEST AVANZATO")
    print("=" * 100)

    print()
    print(
        f"Strategia: EMA20/50 + MACD + RSI {RSI_MIN}-{RSI_MAX} + Trend 15m"
    )

    print(
        f"SL = {SL_ATR} ATR"
    )

    print(
        f"TP = {TP_ATR} ATR"
    )

    print(
        "Ingresso = apertura della candela successiva"
    )

    # --------------------------------------------------------
    # DATI
    # --------------------------------------------------------

    df = scarica_dati()

    # --------------------------------------------------------
    # INDICATORI
    # --------------------------------------------------------

    print()
    print("📐 Calcolo indicatori...")

    df = calcola_indicatori(
        df
    )

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    print(
        "📈 Calcolo trend 15m..."
    )

    df = aggiungi_trend_15m(
        df
    )

    # --------------------------------------------------------
    # SPLIT 70 / 30
    # --------------------------------------------------------

    split_index = int(
        len(df) * 0.70
    )

    df_sviluppo = df.iloc[
        :split_index
    ].copy()

    df_verifica = df.iloc[
        split_index:
    ].copy()

    # --------------------------------------------------------
    # SVILUPPO
    # --------------------------------------------------------

    trades_sviluppo = esegui_fase(
        df_sviluppo,
        "FASE 1 — SVILUPPO 70%"
    )

    # --------------------------------------------------------
    # VERIFICA
    # --------------------------------------------------------

    trades_verifica = esegui_fase(
        df_verifica,
        "FASE 2 — VERIFICA 30%"
    )

    # --------------------------------------------------------
    # CONFRONTO
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("📊 CONFRONTO SVILUPPO / VERIFICA")
    print("=" * 100)

    m1 = calcola_metriche(
        trades_sviluppo
    )

    m2 = calcola_metriche(
        trades_verifica
    )

    print()
    print(
        f"{'Metrica':<25}"
        f"{'Sviluppo':>20}"
        f"{'Verifica':>20}"
    )

    print("-" * 65)

    print(
        f"{'Operazioni':<25}"
        f"{m1['operazioni']:>20}"
        f"{m2['operazioni']:>20}"
    )

    print(
        f"{'Win rate':<25}"
        f"{m1['win_rate']:>19.2f}%"
        f"{m2['win_rate']:>19.2f}%"
    )

    print(
        f"{'Profit Factor':<25}"
        f"{m1['profit_factor']:>20.2f}"
        f"{m2['profit_factor']:>20.2f}"
    )

    print(
        f"{'Risultato':<25}"
        f"{m1['risultato']:>20.2f}"
        f"{m2['risultato']:>20.2f}"
    )

    print(
        f"{'Drawdown':<25}"
        f"{m1['drawdown']:>20.2f}"
        f"{m2['drawdown']:>20.2f}"
    )

    # --------------------------------------------------------
    # ESPORTAZIONE
    # --------------------------------------------------------

    if len(trades_sviluppo) > 0:

        trades_sviluppo.to_csv(
            "trades_sviluppo.csv",
            index=False
        )

    if len(trades_verifica) > 0:

        trades_verifica.to_csv(
            "trades_verifica.csv",
            index=False
        )

    print()
    print("💾 File creati:")

    print(
        " - trades_sviluppo.csv"
    )

    print(
        " - trades_verifica.csv"
    )

    print()
    print("=" * 100)
    print("✅ BACKTEST AVANZATO TERMINATO")
    print("=" * 100)


# ============================================================
# AVVIO
# ============================================================

if __name__ == "__main__":
    main()
