import os
import traceback
import warnings

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")


# ============================================================
# V10 — AGGRESSIVE BUT PRUDENT
# TREND + BREAKOUT + MOMENTUM
# ============================================================

SYMBOL = "XAU/USD"
INTERVAL = "5min"

API_KEY = os.getenv("TWELVE_DATA_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not API_KEY:
    raise RuntimeError("TWELVE_DATA_API_KEY non configurato")

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN non configurato")

if not TELEGRAM_CHAT_ID:
    raise RuntimeError("TELEGRAM_CHAT_ID non configurato")


# ============================================================
# PARAMETRI V10 — FISSI
# ============================================================

EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 100

BREAKOUT_LOOKBACK = 24

BODY_RATIO_MIN = 0.60
CLOSE_LOCATION_MAX = 0.20

MOMENTUM_ATR_MIN = 0.25

ATR_PERIOD = 14
ATR_AVG_PERIOD = 50

RSI_PERIOD = 14

BUY_RSI_MIN = 52
BUY_RSI_MAX = 68

SELL_RSI_MIN = 32
SELL_RSI_MAX = 48

EMA_GAP_ATR_MIN = 0.15

SL_ATR = 1.00
TP_ATR = 2.50

COOLDOWN_MIN = 60

DATA_SIZE = 500


# ============================================================
# FUNZIONI TELEGRAM
# ============================================================

def send_telegram(message):

    url = (
        f"https://api.telegram.org/bot"
        f"{TELEGRAM_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }

    response = requests.post(
        url,
        json=payload,
        timeout=20
    )

    response.raise_for_status()


# ============================================================
# DOWNLOAD DATI
# ============================================================

def get_data():

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": DATA_SIZE,
        "apikey": API_KEY,
        "timezone": "UTC",
        "order": "ASC"
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    if "values" not in data:
        raise RuntimeError(
            f"Errore Twelve Data: {data}"
        )

    df = pd.DataFrame(data["values"])

    if df.empty:
        raise RuntimeError(
            "Twelve Data ha restituito dati vuoti."
        )

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True
    )

    df = df.sort_values(
        "datetime"
    ).reset_index(drop=True)

    for col in [
        "open",
        "high",
        "low",
        "close"
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close"
        ]
    ).reset_index(drop=True)

    return df


# ============================================================
# INDICATORI
# ============================================================

def calculate_indicators(df):

    df = df.copy()

    # --------------------------------------------------------
    # EMA 5m
    # --------------------------------------------------------

    df["ema20"] = df["close"].ewm(
        span=EMA_FAST,
        adjust=False
    ).mean()

    df["ema50"] = df["close"].ewm(
        span=EMA_MID,
        adjust=False
    ).mean()

    df["ema100"] = df["close"].ewm(
        span=EMA_SLOW,
        adjust=False
    ).mean()

    # --------------------------------------------------------
    # TRUE RANGE
    # --------------------------------------------------------

    previous_close = df["close"].shift(1)

    tr1 = (
        df["high"] -
        df["low"]
    )

    tr2 = (
        df["high"] -
        previous_close
    ).abs()

    tr3 = (
        df["low"] -
        previous_close
    ).abs()

    df["tr"] = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    df["atr"] = df["tr"].ewm(
        alpha=1 / ATR_PERIOD,
        adjust=False
    ).mean()

    df["atr_avg50"] = df["atr"].rolling(
        ATR_AVG_PERIOD
    ).mean()

    # --------------------------------------------------------
    # MOM6
    # --------------------------------------------------------

    df["mom6"] = (
        df["close"] -
        df["close"].shift(6)
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = df["close"].diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / RSI_PERIOD,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / RSI_PERIOD,
        adjust=False
    ).mean()

    rs = (
        avg_gain /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    df["rsi"] = (
        100 -
        (
            100 /
            (1 + rs)
        )
    )

    # --------------------------------------------------------
    # CANDLE BODY
    # --------------------------------------------------------

    candle_range = (
        df["high"] -
        df["low"]
    )

    df["body"] = (
        df["close"] -
        df["open"]
    ).abs()

    df["body_ratio"] = (
        df["body"] /
        candle_range.replace(
            0,
            np.nan
        )
    )

    # --------------------------------------------------------
    # CLOSE LOCATION
    # --------------------------------------------------------

    df["close_position"] = (
        df["close"] -
        df["low"]
    ) / candle_range.replace(
        0,
        np.nan
    )

    # --------------------------------------------------------
    # BREAKOUT LEVEL
    #
    # SHIFT(1) = non usa la candela corrente
    # --------------------------------------------------------

    df["previous_high"] = (
        df["high"]
        .rolling(BREAKOUT_LOOKBACK)
        .max()
        .shift(1)
    )

    df["previous_low"] = (
        df["low"]
        .rolling(BREAKOUT_LOOKBACK)
        .min()
        .shift(1)
    )

    # --------------------------------------------------------
    # EMA GAP NORMALIZZATO
    # --------------------------------------------------------

    df["ema_gap_atr"] = (
        (
            df["ema20"] -
            df["ema50"]
        )
        /
        df["atr"].replace(
            0,
            np.nan
        )
    )

    # --------------------------------------------------------
    # TREND 15m
    # --------------------------------------------------------

    temp = df.set_index(
        "datetime"
    )

    tf15 = temp.resample(
        "15min",
        label="left",
        closed="left"
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    })

    tf15 = tf15.dropna()

    tf15["ema20_15"] = (
        tf15["close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    tf15["ema50_15"] = (
        tf15["close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    tf15["trend15"] = np.where(
        tf15["ema20_15"] >
        tf15["ema50_15"],
        "BULLISH",
        "BEARISH"
    )

    # La candela 15m è utilizzabile
    # solo dopo la sua chiusura.
    tf15["available_at"] = (
        tf15.index +
        pd.Timedelta(
            minutes=15
        )
    )

    tf15 = tf15[
        [
            "available_at",
            "trend15"
        ]
    ].sort_values(
        "available_at"
    )

    # --------------------------------------------------------
    # MERGE SENZA LOOKAHEAD
    # --------------------------------------------------------

    df["entry_time"] = (
        df["datetime"] +
        pd.Timedelta(
            minutes=5
        )
    )

    df = pd.merge_asof(
        df.sort_values(
            "entry_time"
        ),
        tf15.sort_values(
            "available_at"
        ),
        left_on="entry_time",
        right_on="available_at",
        direction="backward"
    )

    df = df.drop(
        columns=[
            "available_at"
        ],
        errors="ignore"
    )

    return df


# ============================================================
# SIGNAL CHECK
# ============================================================

def check_signal(row):

    required = [
        "ema20",
        "ema50",
        "ema100",
        "atr",
        "atr_avg50",
        "mom6",
        "rsi",
        "body_ratio",
        "close_position",
        "previous_high",
        "previous_low",
        "ema_gap_atr",
        "trend15"
    ]

    for col in required:

        if pd.isna(row[col]):
            return None

    # ========================================================
    # BUY
    # ========================================================

    buy = (

        row["ema20"] >
        row["ema50"] >
        row["ema100"]

        and
        row["ema_gap_atr"] >=
        EMA_GAP_ATR_MIN

        and
        row["trend15"] ==
        "BULLISH"

        and
        row["close"] >
        row["previous_high"]

        and
        row["body_ratio"] >=
        BODY_RATIO_MIN

        and
        row["close_position"] >=
        (
            1 -
            CLOSE_LOCATION_MAX
        )

        and
        row["mom6"] >=
        (
            MOMENTUM_ATR_MIN *
            row["atr"]
        )

        and
        row["atr"] >
        row["atr_avg50"]

        and
        row["rsi"] >=
        BUY_RSI_MIN

        and
        row["rsi"] <=
        BUY_RSI_MAX
    )

    if buy:
        return "BUY"

    # ========================================================
    # SELL
    # ========================================================

    sell = (

        row["ema20"] <
        row["ema50"] <
        row["ema100"]

        and
        row["ema_gap_atr"] <=
        -EMA_GAP_ATR_MIN

        and
        row["trend15"] ==
        "BEARISH"

        and
        row["close"] <
        row["previous_low"]

        and
        row["body_ratio"] >=
        BODY_RATIO_MIN

        and
        row["close_position"] <=
        CLOSE_LOCATION_MAX

        and
        row["mom6"] <=
        (
            -MOMENTUM_ATR_MIN *
            row["atr"]
        )

        and
        row["atr"] >
        row["atr_avg50"]

        and
        row["rsi"] >=
        SELL_RSI_MIN

        and
        row["rsi"] <=
        SELL_RSI_MAX
    )

    if sell:
        return "SELL"

    return None


# ============================================================
# TELEGRAM MESSAGE
# ============================================================

def build_message(
    direction,
    row
):

    entry = float(
        row["close"]
    )

    atr = float(
        row["atr"]
    )

    if direction == "BUY":

        sl = (
            entry -
            SL_ATR * atr
        )

        tp = (
            entry +
            TP_ATR * atr
        )

        emoji = "🟢"

    else:

        sl = (
            entry +
            SL_ATR * atr
        )

        tp = (
            entry -
            TP_ATR * atr
        )

        emoji = "🔴"

    candle_time = (
        row["datetime"]
        .strftime(
            "%d/%m/%Y %H:%M"
        )
    )

    message = f"""
🧪 PAPER/DEMO — XAU/USD V10

{emoji} {direction}

⏰ Candela: {candle_time} UTC
💰 Entry: {entry:.2f}
🛑 SL: {sl:.2f}
🎯 TP: {tp:.2f}

RSI: {row["rsi"]:.2f}
ATR: {atr:.2f}
MOM6: {row["mom6"]:.2f}

EMA20: {row["ema20"]:.2f}
EMA50: {row["ema50"]:.2f}
EMA100: {row["ema100"]:.2f}

Body Ratio: {row["body_ratio"]:.2f}
Close Position: {row["close_position"]:.2f}

Trend 15m: {row["trend15"]}

Breakout: 24 candele
EMA Gap: 0.15 ATR
Momentum: 0.25 ATR
ATR > ATR(50)

SL: 1.00 ATR
TP: 2.50 ATR
R:R: 1:2.50

Cooldown: 60 min

⚠️ Segnale sperimentale PAPER/DEMO.
Nessun ordine reale viene eseguito.
"""

    return message.strip()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("V10 — AGGRESSIVE BUT PRUDENT")
    print("=" * 70)

    print("\nScaricamento dati...")

    df = get_data()

    print(
        f"Candele ricevute: {len(df)}"
    )

    print(
        f"Ultima candela disponibile: "
        f"{df['datetime'].iloc[-1]}"
    )

    # --------------------------------------------------------
    # IMPORTANTE:
    #
    # Twelve Data può restituire la candela ancora in corso.
    # Il bot deve utilizzare esclusivamente l'ultima
    # candela COMPLETAMENTE CHIUSA.
    #
    # Con intervallo 5m, controlliamo l'ora UTC.
    # --------------------------------------------------------

    now_utc = pd.Timestamp.now(
        tz="UTC"
    )

    df["candle_end"] = (
        df["datetime"] +
        pd.Timedelta(
            minutes=5
        )
    )

    closed_df = df[
        df["candle_end"] <= now_utc
    ].copy()

    if len(closed_df) < 150:

        raise RuntimeError(
            "Dati insufficienti dopo "
            "la rimozione della candela "
            "ancora aperta."
        )

    # --------------------------------------------------------
    # INDICATORI
    # --------------------------------------------------------

    print(
        "\nCalcolo indicatori V10..."
    )

    closed_df = calculate_indicators(
        closed_df
    )

    # --------------------------------------------------------
    # ULTIMA CANDLE CHIUSA
    # --------------------------------------------------------

    row = closed_df.iloc[-1]

    print("\n" + "-" * 70)

    print(
        f"Candela analizzata: "
        f"{row['datetime']}"
    )

    print(
        f"Close: "
        f"{row['close']:.2f}"
    )

    print(
        f"RSI: "
        f"{row['rsi']:.2f}"
    )

    print(
        f"ATR: "
        f"{row['atr']:.2f}"
    )

    print(
        f"Trend 15m: "
        f"{row['trend15']}"
    )

    print(
        f"Body Ratio: "
        f"{row['body_ratio']:.2f}"
    )

    print(
        f"MOM6: "
        f"{row['mom6']:.2f}"
    )

    print(
        f"EMA20: "
        f"{row['ema20']:.2f}"
    )

    print(
        f"EMA50: "
        f"{row['ema50']:.2f}"
    )

    print(
        f"EMA100: "
        f"{row['ema100']:.2f}"
    )

    print("-" * 70)

    # --------------------------------------------------------
    # SIGNAL
    # --------------------------------------------------------

    signal = check_signal(
        row
    )

    if signal is None:

        print(
            "Nessun segnale V10."
        )

        return

    # --------------------------------------------------------
    # COOLDOWN
    #
    # GitHub Actions esegue processi separati.
    # Per evitare segnali duplicati senza affidarsi
    # alla memoria del processo, controlliamo i segnali
    # precedenti direttamente sui dati recenti.
    # --------------------------------------------------------

    recent_start = max(
        0,
        len(closed_df) - 13
    )

    recent_df = closed_df.iloc[
        recent_start:-1
    ]

    previous_signal = None

    for _, previous_row in recent_df.iterrows():

        previous = check_signal(
            previous_row
        )

        if previous is not None:
            previous_signal = previous

    if previous_signal is not None:

        print(
            f"Segnale precedente "
            f"trovato: {previous_signal}"
        )

        print(
            "Cooldown V10 attivo. "
            "Nessun nuovo segnale."
        )

        return

    # --------------------------------------------------------
    # SEND
    # --------------------------------------------------------

    message = build_message(
        signal,
        row
    )

    print(
        f"\n🚨 SEGNALE V10: {signal}"
    )

    print(
        "\nInvio Telegram..."
    )

    send_telegram(
        message
    )

    print(
        "Telegram inviato correttamente."
    )


# ============================================================
# ERROR HANDLING
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as e:

        print(
            "\nERRORE:"
        )

        print(
            str(e)
        )

        traceback.print_exc()

        raise
