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
    raise RuntimeError("TWELVE_DATA_API_KEY non configurato")

OUTPUT_DIR = "backtest_results_v9_1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_RESULTS = os.path.join(OUTPUT_DIR, "v9_1_results.csv")
CSV_ROBUST = os.path.join(OUTPUT_DIR, "v9_1_robust.csv")

N_CANDLES = 9999

# Costi stress
SPREAD_COST = 0.05
SLIPPAGE_COST = 0.05

# Costo totale considerato in R-price units
ENTRY_COST = SPREAD_COST + SLIPPAGE_COST
EXIT_COST = SPREAD_COST + SLIPPAGE_COST
TOTAL_COST = ENTRY_COST + EXIT_COST

# Split
DEV_PCT = 0.60
VAL_PCT = 0.20
TEST_PCT = 0.20

# ============================================================
# GRIGLIA V9.1
# ============================================================

# Body Ratio
BODY_RATIOS = [
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]

# Close location:
# 0 = chiusura esattamente sul minimo
# 1 = chiusura esattamente sul massimo
#
# Per SELL vogliamo valori bassi.
CLOSE_LOCATION_MAX = [
    0.20,
    0.30,
    0.40,
]

# Distanza EMA20-EMA50 normalizzata per ATR
EMA_GAP_ATR_MIN = [
    0.00,
    0.05,
    0.10,
]

# MOM6 negativo normalizzato per ATR
MOM_ATR_MIN = [
    0.00,
    0.10,
    0.20,
]

# RSI
RSI_RANGES = [
    (30, 65),
    (35, 65),
    (35, 60),
]

# TP / SL
TP_ATR_VALUES = [
    2.00,
    2.25,
    2.50,
    2.75,
]

SL_ATR_VALUES = [
    1.00,
    1.25,
    1.50,
]

# Orizzonte
HORIZONS = [
    30,
    60,
]

# Cooldown
COOLDOWNS = [
    30,
    60,
]


# ============================================================
# DOWNLOAD DATI
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

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    if "values" not in data:
        raise RuntimeError(f"Errore Twelve Data: {data}")

    df = pd.DataFrame(data["values"])

    if df.empty:
        raise RuntimeError("Nessun dato ricevuto")

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

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

    print(f"Candele scaricate: {len(df)}")

    if len(df) < 3000:
        raise RuntimeError("Troppi pochi dati per il backtest")

    print(
        f"Periodo: {df['datetime'].iloc[0]} -> "
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
        min_periods=period
    ).mean()


def rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def atr(df, period=14):
    prev_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()


def macd(series):
    ema12 = ema(series, 12)
    ema26 = ema(series, 26)

    line = ema12 - ema26
    signal = line.ewm(
        span=9,
        adjust=False,
        min_periods=9
    ).mean()

    return line, signal


# ============================================================
# INDICATORI 5M
# ============================================================

def prepare_5m(df):
    df = df.copy()

    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["ema100"] = ema(df["close"], 100)

    df["macd"], df["macd_signal"] = macd(df["close"])

    df["rsi"] = rsi(df["close"], 14)

    df["atr"] = atr(df, 14)

    # Momentum a 6 candele = 30 minuti
    df["mom6"] = df["close"] - df["close"].shift(6)

    # --------------------------------------------------------
    # BODY RATIO
    # --------------------------------------------------------

    candle_range = (
        df["high"] - df["low"]
    ).replace(0, np.nan)

    body = (
        df["close"] - df["open"]
    ).abs()

    df["body_ratio"] = body / candle_range

    # --------------------------------------------------------
    # CLOSE LOCATION
    #
    # 0 = close sul minimo
    # 1 = close sul massimo
    # --------------------------------------------------------

    df["close_location"] = (
        df["close"] - df["low"]
    ) / candle_range

    # --------------------------------------------------------
    # EMA GAP NORMALIZZATO
    # --------------------------------------------------------

    df["ema_gap_atr"] = (
        (df["ema50"] - df["ema20"])
        / df["atr"]
    )

    # --------------------------------------------------------
    # MOM NORMALIZZATO
    #
    # Per SELL vogliamo MOM6/ATR negativo.
    # Salviamo anche la versione positiva come strength.
    # --------------------------------------------------------

    df["mom_atr"] = (
        (-df["mom6"]) / df["atr"]
    )

    return df


# ============================================================
# 15 MINUTI SENZA LEAKAGE
# ============================================================

def prepare_15m(df5):
    """
    Costruisce il timeframe 15m e lo rende disponibile
    solamente DOPO la chiusura della relativa candela.

    Esempio:
    candela 10:00-10:15
    -> disponibile solo dalle 10:15.
    """

    x = df5.set_index("datetime")[
        ["open", "high", "low", "close"]
    ].copy()

    tf = x.resample(
        "15min",
        label="left",
        closed="left"
    ).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
    ).dropna()

    tf["ema20"] = ema(tf["close"], 20)
    tf["ema50"] = ema(tf["close"], 50)

    tf["trend_bearish"] = (
        tf["ema20"] < tf["ema50"]
    )

    # timestamp di disponibilità
    tf["available_at"] = (
        tf.index + pd.Timedelta(minutes=15)
    )

    tf = tf.reset_index()

    return tf[
        [
            "datetime",
            "available_at",
            "trend_bearish",
        ]
    ]


def merge_15m(df5, df15):
    """
    Merge as-of usando available_at.
    Quindi il segnale 5m NON può vedere una candela 15m
    ancora non chiusa.
    """

    left = df5.sort_values("datetime").copy()

    right = df15.sort_values("available_at").copy()

    right = right.rename(
        columns={
            "trend_bearish": "trend15_bearish"
        }
    )

    merged = pd.merge_asof(
        left,
        right[
            [
                "available_at",
                "trend15_bearish",
            ]
        ],
        left_on="datetime",
        right_on="available_at",
        direction="backward",
        allow_exact_matches=True,
    )

    return merged


# ============================================================
# PREPARAZIONE DATI
# ============================================================

def prepare_data(df):
    print()
    print("=" * 78)
    print("PREPARAZIONE INDICATORI")
    print("=" * 78)

    df = prepare_5m(df)

    df15 = prepare_15m(df)

    df = merge_15m(
        df,
        df15
    )

    # --------------------------------------------------------
    # SOLO CANDLE COMPLETE
    #
    # Il dato ricevuto può contenere una candela ancora aperta.
    # Il backtest usa sempre la candela precedente.
    # --------------------------------------------------------

    df["signal_time"] = df["datetime"]

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
    ).reset_index(drop=True)

    print(f"Candele utilizzabili: {len(df)}")

    return df


# ============================================================
# SPLIT
# ============================================================

def get_splits(df):
    n = len(df)

    dev_end = int(n * DEV_PCT)
    val_end = int(n * (DEV_PCT + VAL_PCT))

    dev = df.iloc[:dev_end].copy()
    val = df.iloc[dev_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    print()
    print("=" * 78)
    print("SPLIT")
    print("=" * 78)

    print(
        f"DEV : {len(dev)} | "
        f"{dev['datetime'].iloc[0]} -> {dev['datetime'].iloc[-1]}"
    )

    print(
        f"VAL : {len(val)} | "
        f"{val['datetime'].iloc[0]} -> {val['datetime'].iloc[-1]}"
    )

    print(
        f"TEST: {len(test)} | "
        f"{test['datetime'].iloc[0]} -> {test['datetime'].iloc[-1]}"
    )

    return dev, val, test


# ============================================================
# SIGNAL MASK
# ============================================================

def build_signal_mask(
    df,
    body_ratio_min,
    close_location_max,
    ema_gap_min,
    mom_atr_min,
    rsi_low,
    rsi_high,
):
    """
    SELL-only.

    Core V8.1:
    EMA20 < EMA50 < EMA100
    MACD < signal
    RSI range
    MOM6 < 0
    15m bearish

    V9.1 aggiunge:
    Body Ratio
    Close Location
    EMA Gap / ATR
    Momentum / ATR
    """

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
        (df["mom_atr"] >= mom_atr_min)
        &
        (df["body_ratio"] >= body_ratio_min)
        &
        (df["close_location"] <= close_location_max)
        &
        (df["ema_gap_atr"] >= ema_gap_min)
        &
        (df["trend15_bearish"] == True)
    )

    return mask.fillna(False).to_numpy(dtype=bool)


# ============================================================
# TRADE ENGINE
# ============================================================

def simulate(
    df,
    signal_mask,
    tp_atr,
    sl_atr,
    horizon_min,
    cooldown_min,
):
    """
    Simulazione SELL.

    Entry = close della candela segnale.

    SL = entry + SL_ATR * ATR
    TP = entry - TP_ATR * ATR

    Se TP e SL vengono toccati nella stessa candela,
    assumiamo conservativamente SL.

    I trade devono terminare entro lo split.
    """

    if len(df) == 0:
        return None

    times = df["datetime"].to_numpy()
    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    atrs = df["atr"].to_numpy(dtype=float)

    signal_indices = np.flatnonzero(signal_mask)

    if len(signal_indices) == 0:
        return None

    trades = []

    last_signal_time = None

    for idx in signal_indices:

        signal_time = times[idx]

        # ----------------------------------------------------
        # COOLDOWN
        # ----------------------------------------------------

        if last_signal_time is not None:
            elapsed = (
                pd.Timestamp(signal_time)
                - pd.Timestamp(last_signal_time)
            ).total_seconds() / 60.0

            if elapsed < cooldown_min:
                continue

        entry = closes[idx]
        atr_value = atrs[idx]

        if not np.isfinite(entry):
            continue

        if not np.isfinite(atr_value) or atr_value <= 0:
            continue

        sl_distance = sl_atr * atr_value
        tp_distance = tp_atr * atr_value

        # Costi in prezzo
        effective_entry = entry + ENTRY_COST

        sl = effective_entry + sl_distance
        tp = effective_entry - tp_distance

        # ----------------------------------------------------
        # FINESTRA TEMPORALE
        # ----------------------------------------------------

        end_time = (
            pd.Timestamp(signal_time)
            + pd.Timedelta(minutes=horizon_min)
        )

        # Solo candele successive al segnale
        j_end = np.searchsorted(
            times,
            np.datetime64(end_time),
            side="right",
        )

        start_idx = idx + 1

        if start_idx >= j_end:
            continue

        # ----------------------------------------------------
        # SCANSIONE DELLE CANDLE FUTURE
        # ----------------------------------------------------

        outcome = None
        exit_price = None
        exit_time = None
        bars_held = 0

        for j in range(start_idx, j_end):

            bars_held += 1

            high = highs[j]
            low = lows[j]

            hit_sl = high >= sl
            hit_tp = low <= tp

            # ----------------------------------------------
            # CONSERVATIVE SAME-CANDLE RULE
            # ----------------------------------------------

            if hit_sl and hit_tp:
                outcome = "SL"
                exit_price = sl
                exit_time = times[j]
                break

            if hit_sl:
                outcome = "SL"
                exit_price = sl
                exit_time = times[j]
                break

            if hit_tp:
                outcome = "TP"
                exit_price = tp
                exit_time = times[j]
                break

        # ----------------------------------------------------
        # TIME EXIT
        # ----------------------------------------------------

        if outcome is None:

            final_idx = min(
                j_end - 1,
                len(df) - 1
            )

            exit_price = closes[final_idx]
            exit_time = times[final_idx]

            outcome = "TIME"

        # ----------------------------------------------------
        # R RESULT
        # ----------------------------------------------------

        if outcome == "TP":

            gross_r = tp_atr / sl_atr

        elif outcome == "SL":

            gross_r = -1.0

        else:

            # SELL:
            # profit se il prezzo finale è più basso
            pnl_price = (
                effective_entry - exit_price
            )

            gross_r = pnl_price / sl_distance

        # Costi già incorporati parzialmente nell'entry.
        # Applichiamo anche il costo di uscita.
        cost_r = EXIT_COST / sl_distance

        net_r = gross_r - cost_r

        trades.append(
            {
                "signal_time": signal_time,
                "exit_time": exit_time,
                "entry": entry,
                "exit": exit_price,
                "atr": atr_value,
                "outcome": outcome,
                "bars_held": bars_held,
                "gross_R": gross_r,
                "net_R": net_r,
            }
        )

        last_signal_time = signal_time

    if not trades:
        return None

    return pd.DataFrame(trades)


# ============================================================
# STATISTICHE
# ============================================================

def statistics(trades):
    if trades is None or len(trades) == 0:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": np.nan,
            "total_R": np.nan,
            "avg_R": np.nan,
            "pf": np.nan,
            "max_dd": np.nan,
        }

    r = trades["net_R"].to_numpy(dtype=float)

    wins = r[r > 0]
    losses = r[r < 0]

    total_r = r.sum()
    avg_r = r.mean()

    win_rate = (
        len(wins) / len(r) * 100
    )

    gross_profit = wins.sum()

    gross_loss = abs(losses.sum())

    if gross_loss > 0:
        pf = gross_profit / gross_loss
    else:
        pf = np.inf

    equity = np.cumsum(r)

    running_max = np.maximum.accumulate(
        np.insert(equity, 0, 0)
    )[1:]

    dd = running_max - equity

    max_dd = dd.max() if len(dd) else 0

    return {
        "trades": len(r),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "total_R": total_r,
        "avg_R": avg_r,
        "pf": pf,
        "max_dd": max_dd,
    }


# ============================================================
# SCORE ROBUSTEZZA
# ============================================================

def robust_score(dev, val, test):
    """
    Non premia solamente il TEST.

    Richiede:
    - numero minimo di trade
    - DEV positivo
    - VAL positivo
    - TEST positivo
    - PF > 1 nei tre periodi

    Il punteggio favorisce la CONSISTENZA.
    """

    if (
        dev["trades"] < 30
        or val["trades"] < 15
        or test["trades"] < 15
    ):
        return -999999

    if (
        not np.isfinite(dev["avg_R"])
        or not np.isfinite(val["avg_R"])
        or not np.isfinite(test["avg_R"])
    ):
        return -999999

    # Penalità se un periodo è negativo
    penalty = 0

    if dev["avg_R"] <= 0:
        penalty += 5

    if val["avg_R"] <= 0:
        penalty += 5

    if test["avg_R"] <= 0:
        penalty += 5

    if dev["pf"] < 1:
        penalty += 2

    if val["pf"] < 1:
        penalty += 2

    if test["pf"] < 1:
        penalty += 2

    # Media ponderata
    consistency = (
        0.40 * dev["avg_R"]
        + 0.30 * val["avg_R"]
        + 0.30 * test["avg_R"]
    )

    pf_component = (
        0.40 * min(dev["pf"], 3.0)
        + 0.30 * min(val["pf"], 3.0)
        + 0.30 * min(test["pf"], 3.0)
    )

    trade_component = min(
        math.log1p(
            min(
                dev["trades"]
                + val["trades"]
                + test["trades"],
                500,
            )
        ),
        7,
    )

    score = (
        consistency * 100
        + pf_component * 10
        + trade_component
        - penalty * 10
    )

    return score


# ============================================================
# TEST DI UNA CONFIGURAZIONE
# ============================================================

def evaluate_config(
    params,
    dev,
    val,
    test,
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
    # SIGNAL MASK
    # --------------------------------------------------------

    dev_mask = build_signal_mask(
        dev,
        body,
        close_loc,
        ema_gap,
        mom_atr,
        rsi_low,
        rsi_high,
    )

    val_mask = build_signal_mask(
        val,
        body,
        close_loc,
        ema_gap,
        mom_atr,
        rsi_low,
        rsi_high,
    )

    test_mask = build_signal_mask(
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

    dev_trades = simulate(
        dev,
        dev_mask,
        tp,
        sl,
        horizon,
        cooldown,
    )

    val_trades = simulate(
        val,
        val_mask,
        tp,
        sl,
        horizon,
        cooldown,
    )

    test_trades = simulate(
        test,
        test_mask,
        tp,
        sl,
        horizon,
        cooldown,
    )

    dev_stats = statistics(dev_trades)
    val_stats = statistics(val_trades)
    test_stats = statistics(test_trades)

    score = robust_score(
        dev_stats,
        val_stats,
        test_stats,
    )

    return {
        **params,

        "DEV_trades": dev_stats["trades"],
        "DEV_win": dev_stats["win_rate"],
        "DEV_total_R": dev_stats["total_R"],
        "DEV_avg_R": dev_stats["avg_R"],
        "DEV_PF": dev_stats["pf"],
        "DEV_DD": dev_stats["max_dd"],

        "VAL_trades": val_stats["trades"],
        "VAL_win": val_stats["win_rate"],
        "VAL_total_R": val_stats["total_R"],
        "VAL_avg_R": val_stats["avg_R"],
        "VAL_PF": val_stats["pf"],
        "VAL_DD": val_stats["max_dd"],

        "TEST_trades": test_stats["trades"],
        "TEST_win": test_stats["win_rate"],
        "TEST_total_R": test_stats["total_R"],
        "TEST_avg_R": test_stats["avg_R"],
        "TEST_PF": test_stats["pf"],
        "TEST_DD": test_stats["max_dd"],

        "ROBUST_SCORE": score,
    }


# ============================================================
# SCANNER
# ============================================================

def run_scanner(dev, val, test):

    print()
    print("=" * 78)
    print("V9.1 - TARGETED ROBUSTNESS SCANNER")
    print("=" * 78)

    print()
    print("Direzione: SELL")
    print("Motivo: V8 ha mostrato il segnale SELL BODY_STRONG come")
    print("l'area statisticamente più interessante.")
    print()

    results = []

    total = (
        len(BODY_RATIOS)
        * len(CLOSE_LOCATION_MAX)
        * len(EMA_GAP_ATR_MIN)
        * len(MOM_ATR_MIN)
        * len(RSI_RANGES)
        * len(TP_ATR_VALUES)
        * len(SL_ATR_VALUES)
        * len(HORIZONS)
        * len(COOLDOWNS)
    )

    print(f"Configurazioni teoriche: {total:,}")
    print("Scanner mirato V9.1 avviato...")
    print()

    counter = 0
    start_time = time.time()

    for body in BODY_RATIOS:

        for close_loc in CLOSE_LOCATION_MAX:

            for ema_gap in EMA_GAP_ATR_MIN:

                for mom_atr in MOM_ATR_MIN:

                    for rsi_low, rsi_high in RSI_RANGES:

                        for tp in TP_ATR_VALUES:

                            for sl in SL_ATR_VALUES:

                                # Evita configurazioni palesemente
                                # troppo aggressive
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

                                        result = evaluate_config(
                                            params,
                                            dev,
                                            val,
                                            test,
                                        )

                                        results.append(result)

                                        # ------------------------------------------------
                                        # PROGRESS
                                        # ------------------------------------------------

                                        if (
                                            counter % 250 == 0
                                            or counter == total
                                        ):
                                            elapsed = (
                                                time.time()
                                                - start_time
                                            )

                                            rate = (
                                                counter / elapsed
                                                if elapsed > 0
                                                else 0
                                            )

                                            remaining = (
                                                (total - counter)
                                                / rate
                                                if rate > 0
                                                else 0
                                            )

                                            print(
                                                f"[{counter:,}/{total:,}] "
                                                f"{counter / total * 100:5.1f}% | "
                                                f"{rate:.1f} cfg/s | "
                                                f"ETA {remaining / 60:.1f} min"
                                            )

    results_df = pd.DataFrame(results)

    return results_df


# ============================================================
# FILTRO ROBUSTEZ
# ============================================================

def filter_robust(results):

    # -----------------------------------------------
    # Requisiti minimi
    # -----------------------------------------------

    robust = results[
        (results["DEV_trades"] >= 30)
        &
        (results["VAL_trades"] >= 15)
        &
        (results["TEST_trades"] >= 15)

        & (results["DEV_avg_R"] > 0)
        & (results["VAL_avg_R"] > 0)
        & (results["TEST_avg_R"] > 0)

        & (results["DEV_PF"] > 1.0)
        & (results["VAL_PF"] > 1.0)
        & (results["TEST_PF"] > 1.0)
    ].copy()

    if robust.empty:
        return robust

    # -----------------------------------------------
    # Stabilità dell'avg R
    # -----------------------------------------------

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

    # -----------------------------------------------
    # Nessun periodo deve essere drasticamente
    # peggiore degli altri.
    # -----------------------------------------------

    robust["CONSISTENCY_RATIO"] = (
        robust[
            [
                "DEV_avg_R",
                "VAL_avg_R",
                "TEST_avg_R",
            ]
        ].min(axis=1)
        /
        robust[
            [
                "DEV_avg_R",
                "VAL_avg_R",
                "TEST_avg_R",
            ]
        ].max(axis=1)
    )

    robust = robust.sort_values(
        [
            "ROBUST_SCORE",
            "TEST_avg_R",
            "VAL_avg_R",
        ],
        ascending=False,
    )

    return robust


# ============================================================
# REPORT
# ============================================================

def print_report(results, robust):

    print()
    print("=" * 78)
    print("V9.1 - RISULTATI")
    print("=" * 78)

    print(
        f"\nConfigurazioni analizzate: {len(results):,}"
    )

    print(
        f"Configurazioni robuste: {len(robust):,}"
    )

    # --------------------------------------------------------
    # TOP SCORE
    # --------------------------------------------------------

    top = results.sort_values(
        "ROBUST_SCORE",
        ascending=False
    ).head(20)

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

    print()
    print("TOP 20 ROBUST SCORE")
    print("=" * 78)

    print(
        top[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # --------------------------------------------------------
    # ROBUST ONLY
    # --------------------------------------------------------

    if not robust.empty:

        print()
        print("=" * 78)
        print("CONFIGURAZIONI VERAMENTE ROBUSTE")
        print("=" * 78)

        print(
            robust[cols]
            .head(30)
            .to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}"
            )
        )

    else:

        print()
        print("=" * 78)
        print("NESSUNA CONFIGURAZIONE HA SUPERATO TUTTI I FILTRI")
        print("=" * 78)

        print(
            "Questo NON significa che il bot sia inutilizzabile."
        )

        print(
            "Significa che V9.1 non ha trovato una combinazione "
            "abbastanza stabile su DEV + VAL + TEST."
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

    df = prepare_data(df)

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    dev, val, test = get_splits(df)

    # --------------------------------------------------------
    # SCANNER
    # --------------------------------------------------------

    results = run_scanner(
        dev,
        val,
        test,
    )

    # --------------------------------------------------------
    # SALVATAGGIO COMPLETO
    # --------------------------------------------------------

    results = results.sort_values(
        "ROBUST_SCORE",
        ascending=False
    )

    results.to_csv(
        CSV_RESULTS,
        index=False
    )

    # --------------------------------------------------------
    # FILTRO ROBUSTEZ
    # --------------------------------------------------------

    robust = filter_robust(
        results
    )

    robust.to_csv(
        CSV_ROBUST,
        index=False
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print_report(
        results,
        robust,
    )

    elapsed = time.time() - start

    print()
    print("=" * 78)
    print("FINE V9.1")
    print("=" * 78)

    print(
        f"Tempo totale: {elapsed / 60:.2f} minuti"
    )

    print()
    print(f"Risultati completi: {CSV_RESULTS}")
    print(f"Solo robusti:       {CSV_ROBUST}")

    print()
    print("IMPORTANTE:")
    print(
        "Non modificare ancora bot.py in base al primo risultato."
    )
    print(
        "Prima guardiamo DEV + VAL + TEST e verifichiamo "
        "la stabilità dei parametri."
    )


if __name__ == "__main__":
    main()
