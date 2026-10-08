import os
import time
import math
import warnings
import requests

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURAZIONE
# ============================================================

SYMBOL = "XAU/USD"
INTERVAL = "5min"

API_KEY = os.getenv("TWELVE_DATA_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "ERRORE: TWELVE_DATA_API_KEY non configurato."
    )

OUTPUT_DIR = "backtest_results_v9_1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "v9_1_results.csv"
)

ROBUST_FILE = os.path.join(
    OUTPUT_DIR,
    "v9_1_robust.csv"
)

N_CANDLES = 9999


# ============================================================
# COSTI
# ============================================================

SPREAD = 0.05
SLIPPAGE = 0.05

ENTRY_COST = SPREAD + SLIPPAGE
EXIT_COST = SPREAD + SLIPPAGE


# ============================================================
# SPLIT
# ============================================================

DEV_PCT = 0.60
VAL_PCT = 0.20
TEST_PCT = 0.20


# ============================================================
# GRIGLIA V9.1
#
# Intenzionalmente mirata.
# ============================================================

BODY_RATIOS = [
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]

CLOSE_LOCATION_MAX = [
    0.20,
    0.30,
]

EMA_GAP_MIN = [
    0.00,
    0.10,
]

MOM_ATR_MIN = [
    0.00,
    0.10,
]

RSI_RANGES = [
    (30, 65),
    (35, 65),
]

TP_VALUES = [
    2.00,
    2.25,
    2.50,
    2.75,
]

SL_VALUES = [
    1.00,
    1.25,
    1.50,
]

HORIZONS = [
    30,
    60,
]

COOLDOWNS = [
    30,
    60,
]


# ============================================================
# DOWNLOAD
# ============================================================

def download_data():

    print("=" * 78)
    print("V9.1 - DOWNLOAD DATI")
    print("=" * 78)

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": N_CANDLES,
        "apikey": API_KEY,
        "timezone": "UTC",
        "order": "ASC",
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=30,
        )

        print(
            f"HTTP status: {response.status_code}"
        )

        response.raise_for_status()

        data = response.json()

    except requests.exceptions.RequestException as e:

        print()
        print("=" * 78)
        print("ERRORE RICHIESTA TWELVE DATA")
        print("=" * 78)
        print(str(e))
        print("=" * 78)

        raise

    except ValueError as e:

        print()
        print("=" * 78)
        print("ERRORE RISPOSTA JSON")
        print("=" * 78)
        print(str(e))
        print()
        print("Risposta ricevuta:")
        print(response.text[:3000])
        print("=" * 78)

        raise

    print("Risposta Twelve Data ricevuta.")

    if "status" in data:
        print(
            f"API status : {data.get('status')}"
        )

    if "code" in data:
        print(
            f"API code   : {data.get('code')}"
        )

    if "message" in data:
        print(
            f"API message: {data.get('message')}"
        )

    if "values" not in data:

        print()
        print("=" * 78)
        print("ERRORE: NESSUN CAMPO 'values'")
        print("=" * 78)
        print(data)
        print("=" * 78)

        raise RuntimeError(
            "Twelve Data non ha restituito candele."
        )

    df = pd.DataFrame(data["values"])

    if df.empty:
        raise RuntimeError(
            "Twelve Data ha restituito zero candele."
        )

    required = [
        "datetime",
        "open",
        "high",
        "low",
        "close",
    ]

    missing = [
        x for x in required
        if x not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Colonne mancanti: {missing}"
        )

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True,
    )

    for col in [
        "open",
        "high",
        "low",
        "close",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = (
        df[
            [
                "datetime",
                "open",
                "high",
                "low",
                "close",
            ]
        ]
        .dropna()
        .sort_values("datetime")
        .drop_duplicates("datetime")
        .reset_index(drop=True)
    )

    print(
        f"Candele scaricate: {len(df)}"
    )

    if len(df) < 3000:
        raise RuntimeError(
            f"Troppi pochi dati: {len(df)}"
        )

    print(
        "Periodo:"
        f" {df['datetime'].iloc[0]}"
        f" -> "
        f"{df['datetime'].iloc[-1]}"
    )

    return df


# ============================================================
# INDICATORI
# ============================================================

def ema(series, period):

    return series.ewm(
        span=period,
        adjust=False,
        min_periods=period,
    ).mean()


def rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = (
        avg_gain /
        avg_loss.replace(0, np.nan)
    )

    return 100 - (
        100 / (1 + rs)
    )


def atr(df, period=14):

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

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3,
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def macd(series):

    ema12 = ema(
        series,
        12,
    )

    ema26 = ema(
        series,
        26,
    )

    line = ema12 - ema26

    signal = line.ewm(
        span=9,
        adjust=False,
        min_periods=9,
    ).mean()

    return line, signal


# ============================================================
# INDICATORI 5M
# ============================================================

def prepare_5m(df):

    df = df.copy()

    df["ema20"] = ema(
        df["close"],
        20,
    )

    df["ema50"] = ema(
        df["close"],
        50,
    )

    df["ema100"] = ema(
        df["close"],
        100,
    )

    (
        df["macd"],
        df["macd_signal"],
    ) = macd(
        df["close"]
    )

    df["rsi"] = rsi(
        df["close"],
        14,
    )

    df["atr"] = atr(
        df,
        14,
    )

    # 6 candele 5m = 30 minuti
    df["mom6"] = (
        df["close"] -
        df["close"].shift(6)
    )

    candle_range = (
        df["high"] -
        df["low"]
    )

    candle_range = candle_range.replace(
        0,
        np.nan,
    )

    body = (
        df["close"] -
        df["open"]
    ).abs()

    df["body_ratio"] = (
        body /
        candle_range
    )

    # 0 = chiusura sul minimo
    # 1 = chiusura sul massimo
    df["close_location"] = (
        df["close"] -
        df["low"]
    ) / candle_range

    # Per SELL:
    # EMA50 - EMA20 > 0
    # maggiore = trend più distanziato
    df["ema_gap_atr"] = (
        (
            df["ema50"] -
            df["ema20"]
        ) /
        df["atr"]
    )

    # Per SELL:
    # MOM6 negativo.
    # Qui trasformato in forza positiva.
    df["mom_atr"] = (
        -df["mom6"] /
        df["atr"]
    )

    return df


# ============================================================
# 15M - NO LEAKAGE
# ============================================================

def prepare_15m(df):

    x = df.set_index(
        "datetime"
    )[
        [
            "open",
            "high",
            "low",
            "close",
        ]
    ]

    tf = x.resample(
        "15min",
        label="left",
        closed="left",
    ).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
    ).dropna()

    tf["ema20"] = ema(
        tf["close"],
        20,
    )

    tf["ema50"] = ema(
        tf["close"],
        50,
    )

    tf["trend_bearish"] = (
        tf["ema20"] <
        tf["ema50"]
    )

    # La candela 15m diventa disponibile
    # solo dopo 15 minuti.
    tf["available_at"] = (
        tf.index +
        pd.Timedelta(minutes=15)
    )

    return tf.reset_index()[
        [
            "available_at",
            "trend_bearish",
        ]
    ]


def merge_15m(df5, df15):

    left = df5.sort_values(
        "datetime"
    ).copy()

    right = df15.sort_values(
        "available_at"
    ).copy()

    right = right.rename(
        columns={
            "trend_bearish":
            "trend15_bearish"
        }
    )

    merged = pd.merge_asof(
        left,
        right,
        left_on="datetime",
        right_on="available_at",
        direction="backward",
        allow_exact_matches=True,
    )

    return merged


# ============================================================
# PREPARAZIONE COMPLETA
# ============================================================

def prepare_data(df):

    print()
    print("=" * 78)
    print("PREPARAZIONE DATI")
    print("=" * 78)

    df = prepare_5m(df)

    df15 = prepare_15m(
        df
    )

    df = merge_15m(
        df,
        df15,
    )

    # Eliminiamo le righe senza indicatori completi.
    df = df.dropna(
        subset=[
            "ema20",
            "ema50",
            "ema100",
            "macd",
            "macd_signal",
            "rsi",
            "atr",
            "mom6",
            "body_ratio",
            "close_location",
            "ema_gap_atr",
            "mom_atr",
            "trend15_bearish",
        ]
    ).reset_index(
        drop=True
    )

    print(
        f"Candele utilizzabili: {len(df)}"
    )

    return df


# ============================================================
# SPLIT
# ============================================================

def split_data(df):

    n = len(df)

    dev_end = int(
        n * DEV_PCT
    )

    val_end = int(
        n *
        (DEV_PCT + VAL_PCT)
    )

    dev = df.iloc[
        :dev_end
    ].copy()

    val = df.iloc[
        dev_end:val_end
    ].copy()

    test = df.iloc[
        val_end:
    ].copy()

    print()
    print("=" * 78)
    print("SPLIT DEV / VAL / TEST")
    print("=" * 78)

    print(
        f"DEV : {len(dev):5d} | "
        f"{dev['datetime'].iloc[0]} -> "
        f"{dev['datetime'].iloc[-1]}"
    )

    print(
        f"VAL : {len(val):5d} | "
        f"{val['datetime'].iloc[0]} -> "
        f"{val['datetime'].iloc[-1]}"
    )

    print(
        f"TEST: {len(test):5d} | "
        f"{test['datetime'].iloc[0]} -> "
        f"{test['datetime'].iloc[-1]}"
    )

    return dev, val, test


# ============================================================
# PRE-CALCO DELLE FINESTRE FUTURE
# ============================================================

def add_future_arrays(df):

    """
    Prepara le informazioni future necessarie
    per velocizzare enormemente il backtest.

    Non anticipa il futuro nel segnale:
    queste informazioni vengono usate SOLO
    dalla simulazione dopo l'ingresso.
    """

    df = df.copy()

    max_bars = (
        max(HORIZONS) //
        5
    )

    highs = (
        df["high"]
        .to_numpy(dtype=float)
    )

    lows = (
        df["low"]
        .to_numpy(dtype=float)
    )

    closes = (
        df["close"]
        .to_numpy(dtype=float)
    )

    n = len(df)

    future_high = np.full(
        (max_bars, n),
        np.nan,
    )

    future_low = np.full(
        (max_bars, n),
        np.nan,
    )

    future_close = np.full(
        (max_bars, n),
        np.nan,
    )

    for k in range(
        1,
        max_bars + 1,
    ):

        future_high[
            k - 1,
            :-k
        ] = highs[k:]

        future_low[
            k - 1,
            :-k
        ] = lows[k:]

        future_close[
            k - 1,
            :-k
        ] = closes[k:]

    return (
        df,
        future_high,
        future_low,
        future_close,
    )


# ============================================================
# SIGNAL MASK
# ============================================================

def signal_mask(
    df,
    body_ratio,
    close_location,
    ema_gap,
    mom_atr,
    rsi_low,
    rsi_high,
):

    mask = (
        (df["ema20"] < df["ema50"])
        &
        (df["ema50"] < df["ema100"])
        &
        (df["macd"] < df["macd_signal"])
        &
        (df["rsi"] >= rsi_low)
        &
        (df["rsi"] <= rsi_high)
        &
        (df["mom_atr"] >= mom_atr)
        &
        (df["body_ratio"] >= body_ratio)
        &
        (df["close_location"] <= close_location)
        &
        (df["ema_gap_atr"] >= ema_gap)
        &
        (df["trend15_bearish"] == True)
    )

    return mask.to_numpy(
        dtype=bool
    )


# ============================================================
# SIMULAZIONE VELOCE
# ============================================================

def simulate_fast(
    df,
    mask,
    future_high,
    future_low,
    future_close,
    tp_atr,
    sl_atr,
    horizon,
    cooldown,
):

    indices = np.flatnonzero(
        mask
    )

    if len(indices) == 0:
        return np.empty(
            0,
            dtype=float
        )

    horizon_bars = (
        horizon // 5
    )

    times = (
        df["datetime"]
        .to_numpy()
    )

    closes = (
        df["close"]
        .to_numpy(dtype=float)
    )

    atr_values = (
        df["atr"]
        .to_numpy(dtype=float)
    )

    results = []

    last_signal_idx = -10**9

    for idx in indices:

        # Cooldown in barre.
        bars_since = (
            idx -
            last_signal_idx
        )

        if bars_since * 5 < cooldown:
            continue

        entry = closes[idx]
        atr_value = atr_values[idx]

        if (
            not np.isfinite(entry)
            or
            not np.isfinite(atr_value)
            or
            atr_value <= 0
        ):
            continue

        sl_distance = (
            sl_atr *
            atr_value
        )

        tp_distance = (
            tp_atr *
            atr_value
        )

        effective_entry = (
            entry +
            ENTRY_COST
        )

        sl = (
            effective_entry +
            sl_distance
        )

        tp = (
            effective_entry -
            tp_distance
        )

        # ----------------------------------------------------
        # FUTURO
        # ----------------------------------------------------

        end = min(
            horizon_bars,
            len(future_high)
        )

        highs = future_high[
            :end,
            idx
        ]

        lows = future_low[
            :end,
            idx
        ]

        closes_future = future_close[
            :end,
            idx
        ]

        if len(highs) == 0:
            continue

        outcome = None
        exit_price = None

        for k in range(
            len(highs)
        ):

            hit_sl = (
                highs[k] >= sl
            )

            hit_tp = (
                lows[k] <= tp
            )

            # Conservative:
            # se entrambi nella stessa candela,
            # consideriamo SL.
            if hit_sl and hit_tp:

                outcome = "SL"
                exit_price = sl
                break

            if hit_sl:

                outcome = "SL"
                exit_price = sl
                break

            if hit_tp:

                outcome = "TP"
                exit_price = tp
                break

        # ----------------------------------------------------
        # TIME EXIT
        # ----------------------------------------------------

        if outcome is None:

            valid_close = (
                closes_future[
                    ~np.isnan(
                        closes_future
                    )
                ]
            )

            if len(valid_close) == 0:
                continue

            exit_price = (
                valid_close[-1]
            )

            outcome = "TIME"

        # ----------------------------------------------------
        # R
        # ----------------------------------------------------

        if outcome == "TP":

            gross_r = (
                tp_atr /
                sl_atr
            )

        elif outcome == "SL":

            gross_r = -1.0

        else:

            pnl = (
                effective_entry -
                exit_price
            )

            gross_r = (
                pnl /
                sl_distance
            )

        cost_r = (
            EXIT_COST /
            sl_distance
        )

        net_r = (
            gross_r -
            cost_r
        )

        results.append(
            net_r
        )

        last_signal_idx = idx

    return np.asarray(
        results,
        dtype=float
    )


# ============================================================
# STATISTICHE
# ============================================================

def stats(r):

    if len(r) == 0:

        return {
            "trades": 0,
            "win_rate": np.nan,
            "total_R": np.nan,
            "avg_R": np.nan,
            "PF": np.nan,
            "DD": np.nan,
        }

    wins = r[r > 0]
    losses = r[r < 0]

    gross_profit = (
        wins.sum()
    )

    gross_loss = abs(
        losses.sum()
    )

    if gross_loss > 0:
        pf = (
            gross_profit /
            gross_loss
        )
    else:
        pf = np.inf

    equity = np.cumsum(r)

    running_max = np.maximum.accumulate(
        np.concatenate(
            [
                [0.0],
                equity,
            ]
        )
    )[1:]

    dd = (
        running_max -
        equity
    )

    max_dd = (
        dd.max()
        if len(dd)
        else 0
    )

    return {
        "trades": len(r),
        "win_rate": (
            len(wins) /
            len(r) *
            100
        ),
        "total_R": r.sum(),
        "avg_R": r.mean(),
        "PF": pf,
        "DD": max_dd,
    }


# ============================================================
# VALUTAZIONE
# ============================================================

def evaluate(
    params,
    dev,
    val,
    test,
    dev_future,
    val_future,
    test_future,
):

    body = params["body_ratio"]
    close_loc = params["close_location"]
    ema_gap = params["ema_gap"]
    mom_atr = params["mom_atr"]

    rsi_low = params["rsi_low"]
    rsi_high = params["rsi_high"]

    tp = params["tp"]
    sl = params["sl"]

    horizon = params["horizon"]
    cooldown = params["cooldown"]

    # --------------------------------------------------------
    # SIGNAL
    # --------------------------------------------------------

    dev_mask = signal_mask(
        dev,
        body,
        close_loc,
        ema_gap,
        mom_atr,
        rsi_low,
        rsi_high,
    )

    val_mask = signal_mask(
        val,
        body,
        close_loc,
        ema_gap,
        mom_atr,
        rsi_low,
        rsi_high,
    )

    test_mask = signal_mask(
        test,
        body,
        close_loc,
        ema_gap,
        mom_atr,
        rsi_low,
        rsi_high,
    )

    # --------------------------------------------------------
    # SIMULAZIONE
    # --------------------------------------------------------

    dev_r = simulate_fast(
        dev,
        dev_mask,
        dev_future[0],
        dev_future[1],
        dev_future[2],
        tp,
        sl,
        horizon,
        cooldown,
    )

    val_r = simulate_fast(
        val,
        val_mask,
        val_future[0],
        val_future[1],
        val_future[2],
        tp,
        sl,
        horizon,
        cooldown,
    )

    test_r = simulate_fast(
        test,
        test_mask,
        test_future[0],
        test_future[1],
        test_future[2],
        tp,
        sl,
        horizon,
        cooldown,
    )

    ds = stats(dev_r)
    vs = stats(val_r)
    ts = stats(test_r)

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    score = robust_score(
        ds,
        vs,
        ts,
    )

    return {
        **params,

        "DEV_trades": ds["trades"],
        "DEV_win": ds["win_rate"],
        "DEV_total_R": ds["total_R"],
        "DEV_avg_R": ds["avg_R"],
        "DEV_PF": ds["PF"],
        "DEV_DD": ds["DD"],

        "VAL_trades": vs["trades"],
        "VAL_win": vs["win_rate"],
        "VAL_total_R": vs["total_R"],
        "VAL_avg_R": vs["avg_R"],
        "VAL_PF": vs["PF"],
        "VAL_DD": vs["DD"],

        "TEST_trades": ts["trades"],
        "TEST_win": ts["win_rate"],
        "TEST_total_R": ts["total_R"],
        "TEST_avg_R": ts["avg_R"],
        "TEST_PF": ts["PF"],
        "TEST_DD": ts["DD"],

        "ROBUST_SCORE": score,
    }


# ============================================================
# ROBUST SCORE
# ============================================================

def robust_score(
    dev,
    val,
    test,
):

    # Minimo numero di operazioni
    if (
        dev["trades"] < 30
        or val["trades"] < 15
        or test["trades"] < 15
    ):
        return -999999.0

    values = [
        dev["avg_R"],
        val["avg_R"],
        test["avg_R"],
    ]

    if not all(
        np.isfinite(x)
        for x in values
    ):
        return -999999.0

    positive_periods = sum(
        x > 0
        for x in values
    )

    pf_values = [
        dev["PF"],
        val["PF"],
        test["PF"],
    ]

    pf_positive = sum(
        x > 1
        for x in pf_values
        if np.isfinite(x)
    )

    # --------------------------------------------------------
    # BONUS / PENALTY
    # --------------------------------------------------------

    consistency = (
        0.40 * dev["avg_R"]
        +
        0.30 * val["avg_R"]
        +
        0.30 * test["avg_R"]
    )

    pf_score = (
        0.40 * min(dev["PF"], 3)
        +
        0.30 * min(val["PF"], 3)
        +
        0.30 * min(test["PF"], 3)
    )

    score = (
        consistency * 100
        +
        pf_score * 10
        +
        positive_periods * 10
        +
        pf_positive * 5
    )

    # Penalizza fortemente i periodi negativi
    if dev["avg_R"] <= 0:
        score -= 30

    if val["avg_R"] <= 0:
        score -= 30

    if test["avg_R"] <= 0:
        score -= 30

    if dev["PF"] <= 1:
        score -= 15

    if val["PF"] <= 1:
        score -= 15

    if test["PF"] <= 1:
        score -= 15

    return score


# ============================================================
# SCANNER
# ============================================================

def run_scanner(
    dev,
    val,
    test,
):

    print()
    print("=" * 78)
    print("V9.1 - TARGETED ROBUSTNESS SCANNER")
    print("=" * 78)

    total = (
        len(BODY_RATIOS)
        *
        len(CLOSE_LOCATION_MAX)
        *
        len(EMA_GAP_MIN)
        *
        len(MOM_ATR_MIN)
        *
        len(RSI_RANGES)
        *
        len(TP_VALUES)
        *
        len(SL_VALUES)
        *
        len(HORIZONS)
        *
        len(COOLDOWNS)
    )

    print(
        f"Configurazioni teoriche: {total:,}"
    )

    print(
        "Direzione: SELL"
    )

    print(
        "Motivo: V8/V8.1 ha mostrato "
        "il setup SELL come area più interessante."
    )

    # --------------------------------------------------------
    # FUTURE ARRAYS
    # --------------------------------------------------------

    (
        dev,
        dev_high,
        dev_low,
        dev_close,
    ) = add_future_arrays(dev)

    (
        val,
        val_high,
        val_low,
        val_close,
    ) = add_future_arrays(val)

    (
        test,
        test_high,
        test_low,
        test_close,
    ) = add_future_arrays(test)

    dev_future = (
        dev_high,
        dev_low,
        dev_close,
    )

    val_future = (
        val_high,
        val_low,
        val_close,
    )

    test_future = (
        test_high,
        test_low,
        test_close,
    )

    results = []

    counter = 0

    start = time.time()

    # --------------------------------------------------------
    # LOOP
    # --------------------------------------------------------

    for body in BODY_RATIOS:

        for close_loc in CLOSE_LOCATION_MAX:

            for ema_gap in EMA_GAP_MIN:

                for mom_atr in MOM_ATR_MIN:

                    for rsi_low, rsi_high in RSI_RANGES:

                        for tp in TP_VALUES:

                            for sl in SL_VALUES:

                                # TP deve essere superiore
                                # alla distanza SL.
                                if tp <= sl:
                                    continue

                                for horizon in HORIZONS:

                                    for cooldown in COOLDOWNS:

                                        counter += 1

                                        params = {
                                            "direction": "SELL",

                                            "body_ratio": body,

                                            "close_location": close_loc,

                                            "ema_gap": ema_gap,

                                            "mom_atr": mom_atr,

                                            "rsi_low": rsi_low,

                                            "rsi_high": rsi_high,

                                            "tp": tp,

                                            "sl": sl,

                                            "horizon": horizon,

                                            "cooldown": cooldown,
                                        }

                                        result = evaluate(
                                            params,
                                            dev,
                                            val,
                                            test,
                                            dev_future,
                                            val_future,
                                            test_future,
                                        )

                                        results.append(
                                            result
                                        )

                                        # ------------------------------------------------
                                        # PROGRESS
                                        # ------------------------------------------------

                                        if (
                                            counter % 250 == 0
                                        ):

                                            elapsed = (
                                                time.time()
                                                - start
                                            )

                                            rate = (
                                                counter /
                                                elapsed
                                                if elapsed > 0
                                                else 0
                                            )

                                            remaining = (
                                                (
                                                    total -
                                                    counter
                                                )
                                                /
                                                rate
                                                if rate > 0
                                                else 0
                                            )

                                            print(
                                                f"[{counter:,}/{total:,}] "
                                                f"{counter / total * 100:5.1f}% | "
                                                f"{rate:.1f} cfg/s | "
                                                f"ETA {remaining / 60:.1f} min"
                                            )

    return pd.DataFrame(
        results
    )


# ============================================================
# FILTRO ROBUSTEZ
# ============================================================

def filter_robust(results):

    robust = results[
        (results["DEV_trades"] >= 30)
        &
        (results["VAL_trades"] >= 15)
        &
        (results["TEST_trades"] >= 15)

        &
        (results["DEV_avg_R"] > 0)
        &
        (results["VAL_avg_R"] > 0)
        &
        (results["TEST_avg_R"] > 0)

        &
        (results["DEV_PF"] > 1)
        &
        (results["VAL_PF"] > 1)
        &
        (results["TEST_PF"] > 1)
    ].copy()

    if robust.empty:
        return robust

    robust["AVG_R_RANGE"] = (
        robust[
            [
                "DEV_avg_R",
                "VAL_avg_R",
                "TEST_avg_R",
            ]
        ].max(axis=1)
        -
        robust[
            [
                "DEV_avg_R",
                "VAL_avg_R",
                "TEST_avg_R",
            ]
        ].min(axis=1)
    )

    robust = robust.sort_values(
        [
            "ROBUST_SCORE",
            "VAL_avg_R",
            "TEST_avg_R",
        ],
        ascending=False,
    )

    return robust


# ============================================================
# REPORT
# ============================================================

def print_report(
    results,
    robust,
):

    print()
    print("=" * 78)
    print("RISULTATI V9.1")
    print("=" * 78)

    print(
        f"Configurazioni analizzate: "
        f"{len(results):,}"
    )

    print(
        f"Configurazioni robuste: "
        f"{len(robust):,}"
    )

    cols = [
        "direction",

        "body_ratio",
        "close_location",
        "ema_gap",
        "mom_atr",

        "rsi_low",
        "rsi_high",

        "tp",
        "sl",
        "horizon",
        "cooldown",

        "DEV_trades",
        "DEV_avg_R",
        "DEV_PF",

        "VAL_trades",
        "VAL_avg_R",
        "VAL_PF",

        "TEST_trades",
        "TEST_avg_R",
        "TEST_PF",

        "ROBUST_SCORE",
    ]

    top = results.sort_values(
        "ROBUST_SCORE",
        ascending=False,
    ).head(20)

    print()
    print("=" * 78)
    print("TOP 20 ROBUST SCORE")
    print("=" * 78)

    print(
        top[cols].to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}"
        )
    )

    print()

    if len(robust) > 0:

        print("=" * 78)
        print("CONFIGURAZIONI ROBUSTE")
        print("=" * 78)

        print(
            robust[
                cols
            ].head(30).to_string(
                index=False,
                float_format=lambda x:
                f"{x:.4f}"
            )
        )

    else:

        print("=" * 78)
        print("NESSUNA CONFIGURAZIONE ROBUSTA")
        print("=" * 78)

        print(
            "Nessuna configurazione ha superato "
            "tutti i requisiti DEV + VAL + TEST."
        )

        print(
            "Questo è un risultato utile: "
            "non trasferiamo una configurazione "
            "fragile dentro bot.py."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    start = time.time()

    print()
    print("=" * 78)
    print("XAU/USD BACKTEST V9.1")
    print("TARGETED ROBUSTNESS VALIDATION")
    print("=" * 78)

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    df = download_data()

    # --------------------------------------------------------
    # INDICATORI
    # --------------------------------------------------------

    df = prepare_data(
        df
    )

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    dev, val, test = split_data(
        df
    )

    # --------------------------------------------------------
    # SCANNER
    # --------------------------------------------------------

    results = run_scanner(
        dev,
        val,
        test,
    )

    # --------------------------------------------------------
    # SALVATAGGIO
    # --------------------------------------------------------

    results = results.sort_values(
        "ROBUST_SCORE",
        ascending=False,
    )

    results.to_csv(
        RESULTS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # ROBUST
    # --------------------------------------------------------

    robust = filter_robust(
        results
    )

    robust.to_csv(
        ROBUST_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print_report(
        results,
        robust,
    )

    elapsed = (
        time.time() -
        start
    )

    print()
    print("=" * 78)
    print("FINE V9.1")
    print("=" * 78)

    print(
        f"Tempo totale: "
        f"{elapsed / 60:.2f} minuti"
    )

    print()
    print(
        f"Risultati completi: "
        f"{RESULTS_FILE}"
    )

    print(
        f"Solo robusti: "
        f"{ROBUST_FILE}"
    )

    print()
    print(
        "NON modificare ancora bot.py."
    )

    print(
        "Prima analizziamo DEV + VAL + TEST."
    )


# ============================================================
# AVVIO
# ============================================================

if __name__ == "__main__":
    main()
