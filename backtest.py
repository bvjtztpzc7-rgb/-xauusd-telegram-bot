import os
import time
import warnings
from itertools import product

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURAZIONE
# ============================================================

SYMBOL = "XAU/USD"
INTERVAL = "5min"

API_KEY = os.getenv("TWELVE_DATA_API_KEY")

if not API_KEY:
    raise RuntimeError("TWELVE_DATA_API_KEY non configurato")

OUTPUT_DIR = "backtest_results_v6"

TOTAL_CANDLES = 10000
BLOCK_SIZE = 5000

DEV_RATIO = 0.70

ATR_PERIOD = 14

MIN_TRADES_DEV = 30
MIN_TRADES_VER = 15

# Numero massimo di combinazioni
MAX_COMBINATIONS = 10000

# Orizzonte in candele da 5 minuti
HORIZONS = [
    3,      # 15m
    6,      # 30m
    12,     # 1h
    24,     # 2h
    48,     # 4h
    96,     # 8h
    288,    # 24h
]

EMA_STRENGTHS = [
    0.0,
    0.05,
    0.10,
    0.15,
    0.20,
]

RSI_RANGES = [
    (0, 100),
    (30, 70),
    (30, 65),
    (35, 65),
    (35, 60),
    (40, 60),
]

ATR_REGIMES = [
    "ALL",
    "NORMAL",
    "HIGH",
]

TREND_MODES = [
    "NONE",
    "15M",
    "1H",
    "15M_1H",
]

MOMENTUM_MODES = [
    "NONE",
    "MOM6",
    "MOM12",
]

SIGNAL_MODES = [
    "ALL",
    "FIRST_IN_EPISODE",
    "NON_OVERLAPPING",
]

TP_VALUES = [
    0.75,
    1.0,
    1.25,
    1.5,
    2.0,
    2.5,
    3.0,
]

SL_VALUES = [
    0.75,
    1.0,
    1.25,
    1.5,
    2.0,
    2.5,
]


# ============================================================
# DIRECTORY
# ============================================================

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DOWNLOAD
# ============================================================

def download_block(outputsize=5000, end_date=None):

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": outputsize,
        "timezone": "UTC",
        "apikey": API_KEY,
        "order": "ASC",
    }

    if end_date is not None:
        params["end_date"] = end_date

    response = requests.get(
        url,
        params=params,
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Twelve Data HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )

    data = response.json()

    if "values" not in data:
        raise RuntimeError(
            f"Risposta Twelve Data non valida: {data}"
        )

    df = pd.DataFrame(data["values"])

    required = [
        "datetime",
        "open",
        "high",
        "low",
        "close",
    ]

    for column in required:
        if column not in df.columns:
            raise RuntimeError(
                f"Colonna mancante: {column}"
            )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "datetime",
            "open",
            "high",
            "low",
            "close",
        ]
    )

    return df


def download_data():

    print("=" * 70)
    print("DOWNLOAD DATI")
    print("=" * 70)

    blocks = []

    first = download_block(BLOCK_SIZE)

    blocks.append(first)

    print(
        f"Blocco 1: {len(first)} candele | "
        f"{first['datetime'].min()} -> "
        f"{first['datetime'].max()}"
    )

    if len(first) >= BLOCK_SIZE:

        oldest = first["datetime"].min()

        second = download_block(
            BLOCK_SIZE,
            end_date=oldest.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        )

        blocks.append(second)

        print(
            f"Blocco 2: {len(second)} candele | "
            f"{second['datetime'].min()} -> "
            f"{second['datetime'].max()}"
        )

    df = pd.concat(
        blocks,
        ignore_index=True,
    )

    df = (
        df
        .drop_duplicates(
            subset=["datetime"]
        )
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    if len(df) > TOTAL_CANDLES:
        df = df.tail(
            TOTAL_CANDLES
        ).reset_index(drop=True)

    print()
    print(f"Candele finali: {len(df)}")
    print(f"Da: {df['datetime'].iloc[0]}")
    print(f"A : {df['datetime'].iloc[-1]}")

    return df


# ============================================================
# INDICATORI
# ============================================================

def add_indicators(df):

    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # EMA
    df["ema20"] = close.ewm(
        span=20,
        adjust=False,
    ).mean()

    df["ema50"] = close.ewm(
        span=50,
        adjust=False,
    ).mean()

    df["ema100"] = close.ewm(
        span=100,
        adjust=False,
    ).mean()

    # Distanza EMA %
    df["ema_distance"] = (
        (
            df["ema20"]
            - df["ema50"]
        ).abs()
        / close
        * 100
    )

    # RSI
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    rs = (
        avg_gain
        / avg_loss.replace(0, np.nan)
    )

    df["rsi"] = (
        100
        - (100 / (1 + rs))
    )

    # MACD
    ema12 = close.ewm(
        span=12,
        adjust=False,
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False,
    ).mean()

    df["macd"] = (
        ema12 - ema26
    )

    df["macd_signal"] = (
        df["macd"]
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    df["macd_hist"] = (
        df["macd"]
        - df["macd_signal"]
    )

    # ATR
    previous_close = close.shift(1)

    tr1 = high - low
    tr2 = (
        high - previous_close
    ).abs()
    tr3 = (
        low - previous_close
    ).abs()

    tr = pd.concat(
        [
            tr1,
            tr2,
            tr3,
        ],
        axis=1,
    ).max(axis=1)

    df["atr"] = (
        tr
        .rolling(ATR_PERIOD)
        .mean()
    )

    df["atr_pct"] = (
        df["atr"]
        / close
        * 100
    )

    df["atr_pct_median"] = (
        df["atr_pct"]
        .rolling(
            200,
            min_periods=50,
        )
        .median()
    )

    # Momentum
    df["mom6"] = (
        close.pct_change(6)
        * 100
    )

    df["mom12"] = (
        close.pct_change(12)
        * 100
    )

    # Candela
    df["body"] = (
        close - df["open"]
    ).abs()

    df["range"] = (
        high - low
    )

    df["body_ratio"] = (
        df["body"]
        / df["range"].replace(
            0,
            np.nan,
        )
    )

    df["hour"] = (
        df["datetime"].dt.hour
    )

    return df


# ============================================================
# MULTI TIMEFRAME
# ============================================================

def add_higher_timeframes(df):

    df = df.copy()

    base = df.set_index(
        "datetime"
    )

    aggregation = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }

    # --------------------------------------------------------
    # 15 MINUTI
    # --------------------------------------------------------

    tf15 = (
        base
        .resample("15min")
        .agg(aggregation)
        .dropna()
    )

    tf15["ema20"] = (
        tf15["close"]
        .ewm(
            span=20,
            adjust=False,
        )
        .mean()
    )

    tf15["ema50"] = (
        tf15["close"]
        .ewm(
            span=50,
            adjust=False,
        )
        .mean()
    )

    tf15["trend"] = np.where(
        tf15["ema20"]
        > tf15["ema50"],
        1,
        np.where(
            tf15["ema20"]
            < tf15["ema50"],
            -1,
            0,
        ),
    )

    # IMPORTANTE:
    # usiamo solo la candela 15m già chiusa.
    tf15["trend"] = (
        tf15["trend"].shift(1)
    )

    tf15 = tf15[
        ["trend"]
    ].rename(
        columns={
            "trend": "trend_15m"
        }
    )

    # --------------------------------------------------------
    # 1 ORA
    # --------------------------------------------------------

    tf1h = (
        base
        .resample("1h")
        .agg(aggregation)
        .dropna()
    )

    tf1h["ema20"] = (
        tf1h["close"]
        .ewm(
            span=20,
            adjust=False,
        )
        .mean()
    )

    tf1h["ema50"] = (
        tf1h["close"]
        .ewm(
            span=50,
            adjust=False,
        )
        .mean()
    )

    tf1h["trend"] = np.where(
        tf1h["ema20"]
        > tf1h["ema50"],
        1,
        np.where(
            tf1h["ema20"]
            < tf1h["ema50"],
            -1,
            0,
        ),
    )

    tf1h["trend"] = (
        tf1h["trend"].shift(1)
    )

    tf1h = tf1h[
        ["trend"]
    ].rename(
        columns={
            "trend": "trend_1h"
        }
    )

    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    result = df.set_index(
        "datetime"
    )

    result = pd.merge_asof(
        result.sort_index(),
        tf15.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )

    result = pd.merge_asof(
        result.sort_index(),
        tf1h.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )

    return (
        result
        .reset_index()
    )


# ============================================================
# FILTRI
# ============================================================

def ema_strength_ok(
    row,
    threshold,
):

    if threshold <= 0:
        return True

    return (
        row["ema_distance"]
        >= threshold
    )


def atr_regime_ok(
    row,
    regime,
):

    if regime == "ALL":
        return True

    if pd.isna(
        row["atr_pct"]
    ) or pd.isna(
        row["atr_pct_median"]
    ):
        return False

    if regime == "NORMAL":
        return (
            row["atr_pct"]
            <= row["atr_pct_median"]
        )

    if regime == "HIGH":
        return (
            row["atr_pct"]
            > row["atr_pct_median"]
        )

    return True


def trend_ok(
    row,
    direction,
    mode,
):

    if mode == "NONE":
        return True

    required = (
        1
        if direction == "BUY"
        else -1
    )

    if mode == "15M":
        return (
            row["trend_15m"]
            == required
        )

    if mode == "1H":
        return (
            row["trend_1h"]
            == required
        )

    if mode == "15M_1H":
        return (
            row["trend_15m"]
            == required
            and
            row["trend_1h"]
            == required
        )

    return True


def momentum_ok(
    row,
    direction,
    mode,
):

    if mode == "NONE":
        return True

    if direction == "BUY":

        if mode == "MOM6":
            return row["mom6"] > 0

        if mode == "MOM12":
            return row["mom12"] > 0

    if direction == "SELL":

        if mode == "MOM6":
            return row["mom6"] < 0

        if mode == "MOM12":
            return row["mom12"] < 0

    return True


# ============================================================
# DIREZIONE BASE
# ============================================================

def base_direction(row):

    values = [
        row["ema20"],
        row["ema50"],
        row["macd"],
        row["macd_signal"],
        row["rsi"],
        row["atr"],
    ]

    if any(
        pd.isna(x)
        for x in values
    ):
        return None

    # BUY
    if (
        row["ema20"]
        > row["ema50"]
        and
        row["macd"]
        > row["macd_signal"]
        and
        30
        < row["rsi"]
        < 70
    ):
        return "BUY"

    # SELL
    if (
        row["ema20"]
        < row["ema50"]
        and
        row["macd"]
        < row["macd_signal"]
        and
        30
        < row["rsi"]
        < 70
    ):
        return "SELL"

    return None


# ============================================================
# SEGNALE COMPLETO
# ============================================================

def signal_allowed(
    row,
    direction,
    cfg,
):

    if direction is None:
        return False

    rsi_low = (
        cfg["rsi_range"][0]
    )

    rsi_high = (
        cfg["rsi_range"][1]
    )

    if pd.isna(
        row["rsi"]
    ):
        return False

    if not (
        rsi_low
        < row["rsi"]
        < rsi_high
    ):
        return False

    if not ema_strength_ok(
        row,
        cfg["ema_strength"],
    ):
        return False

    if not atr_regime_ok(
        row,
        cfg["atr_regime"],
    ):
        return False

    if not trend_ok(
        row,
        direction,
        cfg["trend_mode"],
    ):
        return False

    if not momentum_ok(
        row,
        direction,
        cfg["momentum_mode"],
    ):
        return False

    if cfg["body_filter"]:

        if (
            pd.isna(
                row["body_ratio"]
            )
            or
            row["body_ratio"] < 0.25
        ):
            return False

    return True


# ============================================================
# GENERA SEGNALI
# ============================================================

def build_signals(
    df,
    cfg,
):

    signals = []

    last_signal_index = {
        "BUY": -10000,
        "SELL": -10000,
    }

    previous_direction = None

    for i in range(
        len(df) - 1
    ):

        row = df.iloc[i]

        direction = (
            base_direction(row)
        )

        if direction is None:
            previous_direction = None
            continue

        if direction != cfg["direction"]:
            previous_direction = None
            continue

        if not signal_allowed(
            row,
            direction,
            cfg,
        ):
            previous_direction = None
            continue

        mode = (
            cfg["signal_mode"]
        )

        # ----------------------------------------------------
        # FIRST IN EPISODE
        # ----------------------------------------------------

        if mode == "FIRST_IN_EPISODE":

            if (
                previous_direction
                == direction
            ):
                continue

        # ----------------------------------------------------
        # NON OVERLAPPING
        # ----------------------------------------------------

        if mode == "NON_OVERLAPPING":

            if (
                i
                - last_signal_index[
                    direction
                ]
                < cfg["horizon"]
            ):
                continue

        signals.append(
            {
                "signal_index": i,
                "entry_index": i + 1,
                "direction": direction,
                "signal_time": row[
                    "datetime"
                ],
                "entry_price": df.iloc[
                    i + 1
                ]["open"],
                "atr": row["atr"],
                "rsi": row["rsi"],
                "ema_distance": row[
                    "ema_distance"
                ],
                "trend_15m": row[
                    "trend_15m"
                ],
                "trend_1h": row[
                    "trend_1h"
                ],
            }
        )

        last_signal_index[
            direction
        ] = i

        previous_direction = (
            direction
        )

    return signals


# ============================================================
# SIMULA TRADE
# ============================================================

def simulate_trade(
    df,
    signal,
    tp_atr,
    sl_atr,
    horizon,
):

    entry_idx = (
        signal["entry_index"]
    )

    if entry_idx >= len(df):
        return None

    entry = (
        signal["entry_price"]
    )

    atr = signal["atr"]

    direction = (
        signal["direction"]
    )

    if (
        pd.isna(entry)
        or pd.isna(atr)
        or atr <= 0
    ):
        return None

    tp_distance = (
        atr * tp_atr
    )

    sl_distance = (
        atr * sl_atr
    )

    if direction == "BUY":

        tp = (
            entry + tp_distance
        )

        sl = (
            entry - sl_distance
        )

    else:

        tp = (
            entry - tp_distance
        )

        sl = (
            entry + sl_distance
        )

    end_idx = min(
        entry_idx + horizon,
        len(df) - 1,
    )

    result = "TIMEOUT"

    exit_price = (
        df.iloc[end_idx]["close"]
    )

    exit_idx = end_idx

    for j in range(
        entry_idx,
        end_idx + 1,
    ):

        candle = df.iloc[j]

        high = candle["high"]
        low = candle["low"]

        if direction == "BUY":

            hit_sl = (
                low <= sl
            )

            hit_tp = (
                high >= tp
            )

        else:

            hit_sl = (
                high >= sl
            )

            hit_tp = (
                low <= tp
            )

        # Conservativo:
        # se TP e SL vengono toccati
        # nella stessa candela -> SL.
        if hit_sl and hit_tp:

            result = "SL"
            exit_price = sl
            exit_idx = j
            break

        if hit_sl:

            result = "SL"
            exit_price = sl
            exit_idx = j
            break

        if hit_tp:

            result = "TP"
            exit_price = tp
            exit_idx = j
            break

    if direction == "BUY":
        pnl = (
            exit_price - entry
        )
    else:
        pnl = (
            entry - exit_price
        )

    risk = sl_distance

    if risk <= 0:
        return None

    r_multiple = (
        pnl / risk
    )

    return {
        "signal_time": signal[
            "signal_time"
        ],
        "entry_time": df.iloc[
            entry_idx
        ]["datetime"],
        "exit_time": df.iloc[
            exit_idx
        ]["datetime"],
        "direction": direction,
        "entry": entry,
        "exit": exit_price,
        "tp": tp,
        "sl": sl,
        "atr": atr,
        "rsi": signal["rsi"],
        "ema_distance": signal[
            "ema_distance"
        ],
        "trend_15m": signal[
            "trend_15m"
        ],
        "trend_1h": signal[
            "trend_1h"
        ],
        "result": result,
        "R": r_multiple,
        "bars_held": (
            exit_idx - entry_idx
        ),
        "tp_atr": tp_atr,
        "sl_atr": sl_atr,
        "horizon": horizon,
    }


# ============================================================
# STATISTICHE
# ============================================================

def calculate_stats(
    trades
):

    if (
        trades is None
        or len(trades) == 0
    ):
        return {
            "trades": 0,
            "tp": 0,
            "sl": 0,
            "timeout": 0,
            "win_rate": np.nan,
            "avg_R": np.nan,
            "total_R": np.nan,
            "PF": np.nan,
            "max_DD_R": np.nan,
        }

    r = pd.Series(
        trades["R"].values,
        dtype=float,
    )

    wins = trades[
        trades["result"] == "TP"
    ]

    losses = trades[
        trades["result"] == "SL"
    ]

    gross_profit = (
        wins["R"].sum()
    )

    gross_loss = abs(
        losses["R"].sum()
    )

    if gross_loss > 0:

        pf = (
            gross_profit
            / gross_loss
        )

    else:

        if gross_profit > 0:
            pf = np.inf
        else:
            pf = 0

    equity = r.cumsum()

    running_max = (
        equity.cummax()
    )

    drawdown = (
        running_max - equity
    )

    max_dd = (
        drawdown.max()
        if len(drawdown)
        else 0
    )

    return {
        "trades": len(trades),

        "tp": int(
            (
                trades["result"]
                == "TP"
            ).sum()
        ),

        "sl": int(
            (
                trades["result"]
                == "SL"
            ).sum()
        ),

        "timeout": int(
            (
                trades["result"]
                == "TIMEOUT"
            ).sum()
        ),

        "win_rate": (
            len(wins)
            / len(trades)
            * 100
        ),

        "avg_R": r.mean(),

        "total_R": r.sum(),

        "PF": pf,

        "max_DD_R": max_dd,
    }


# ============================================================
# ROBUSTNESS SCORE
# ============================================================

def robustness_score(
    dev,
    ver,
):

    if (
        dev["trades"]
        < MIN_TRADES_DEV
        or
        ver["trades"]
        < MIN_TRADES_VER
    ):
        return -9999

    required = [
        dev["PF"],
        ver["PF"],
        dev["avg_R"],
        ver["avg_R"],
    ]

    if any(
        pd.isna(x)
        for x in required
    ):
        return -9999

    score = 0.0

    # PF
    score += (
        min(
            dev["PF"],
            2.0,
        )
        * 20
    )

    score += (
        min(
            ver["PF"],
            2.0,
        )
        * 35
    )

    # Avg R
    score += (
        max(
            -1,
            min(
                dev["avg_R"],
                0.5,
            ),
        )
        * 30
    )

    score += (
        max(
            -1,
            min(
                ver["avg_R"],
                0.5,
            ),
        )
        * 50
    )

    # Total R
    score += (
        max(
            -50,
            min(
                dev["total_R"],
                50,
            ),
        )
        * 0.10
    )

    score += (
        max(
            -50,
            min(
                ver["total_R"],
                50,
            ),
        )
        * 0.20
    )

    # Differenza PF
    pf_difference = abs(
        dev["PF"]
        - ver["PF"]
    )

    score -= (
        min(
            pf_difference,
            2,
        )
        * 5
    )

    # Drawdown
    score -= (
        min(
            dev["max_DD_R"],
            50,
        )
        * 0.20
    )

    score -= (
        min(
            ver["max_DD_R"],
            50,
        )
        * 0.40
    )

    # Bonus robustezza
    if (
        dev["PF"] > 1
        and
        ver["PF"] > 1
    ):
        score += 25

    if (
        dev["avg_R"] > 0
        and
        ver["avg_R"] > 0
    ):
        score += 25

    if (
        dev["total_R"] > 0
        and
        ver["total_R"] > 0
    ):
        score += 15

    return score


# ============================================================
# GENERA CONFIGURAZIONI
# ============================================================

def generate_configs():

    configs = []

    combinations = product(
        ["BUY", "SELL"],
        EMA_STRENGTHS,
        RSI_RANGES,
        ATR_REGIMES,
        TREND_MODES,
        MOMENTUM_MODES,
        SIGNAL_MODES,
        [False, True],
        HORIZONS,
        TP_VALUES,
        SL_VALUES,
    )

    for combo in combinations:

        (
            direction,
            ema_strength,
            rsi_range,
            atr_regime,
            trend_mode,
            momentum_mode,
            signal_mode,
            body_filter,
            horizon,
            tp,
            sl,
        ) = combo

        # --------------------------------------------
        # Riduzione combinazioni poco interessanti
        # --------------------------------------------

        if (
            signal_mode != "ALL"
            and horizon < 12
        ):
            continue

        if (
            trend_mode == "15M_1H"
            and momentum_mode != "NONE"
        ):
            continue

        if (
            atr_regime == "HIGH"
            and momentum_mode == "MOM12"
        ):
            continue

        configs.append(
            {
                "direction": direction,
                "ema_strength": ema_strength,
                "rsi_range": rsi_range,
                "atr_regime": atr_regime,
                "trend_mode": trend_mode,
                "momentum_mode": momentum_mode,
                "signal_mode": signal_mode,
                "body_filter": body_filter,
                "horizon": horizon,
                "tp": tp,
                "sl": sl,
            }
        )

        if (
            len(configs)
            >= MAX_COMBINATIONS
        ):
            break

    return configs


# ============================================================
# RUN CONFIG
# ============================================================

def run_configuration(
    df,
    cfg,
    split_index,
):

    signals = build_signals(
        df,
        cfg,
    )

    trades = []

    for signal in signals:

        trade = simulate_trade(
            df,
            signal,
            cfg["tp"],
            cfg["sl"],
            cfg["horizon"],
        )

        if trade is not None:
            trades.append(trade)

    if not trades:
        return None

    trades_df = pd.DataFrame(
        trades
    )

    split_time = df.iloc[
        split_index
    ]["datetime"]

    dev = trades_df[
        trades_df["signal_time"]
        < split_time
    ].copy()

    ver = trades_df[
        trades_df["signal_time"]
        >= split_time
    ].copy()

    dev_stats = calculate_stats(
        dev
    )

    ver_stats = calculate_stats(
        ver
    )

    score = robustness_score(
        dev_stats,
        ver_stats,
    )

    result = {

        "direction": cfg[
            "direction"
        ],

        "ema_strength": cfg[
            "ema_strength"
        ],

        "rsi_low": cfg[
            "rsi_range"
        ][0],

        "rsi_high": cfg[
            "rsi_range"
        ][1],

        "atr_regime": cfg[
            "atr_regime"
        ],

        "trend_mode": cfg[
            "trend_mode"
        ],

        "momentum_mode": cfg[
            "momentum_mode"
        ],

        "signal_mode": cfg[
            "signal_mode"
        ],

        "body_filter": cfg[
            "body_filter"
        ],

        "horizon_bars": cfg[
            "horizon"
        ],

        "horizon_minutes": (
            cfg["horizon"] * 5
        ),

        "TP_ATR": cfg["tp"],

        "SL_ATR": cfg["sl"],

        "DEV_trades": dev_stats[
            "trades"
        ],

        "DEV_win": dev_stats[
            "win_rate"
        ],

        "DEV_avg_R": dev_stats[
            "avg_R"
        ],

        "DEV_total_R": dev_stats[
            "total_R"
        ],

        "DEV_PF": dev_stats[
            "PF"
        ],

        "DEV_DD": dev_stats[
            "max_DD_R"
        ],

        "VER_trades": ver_stats[
            "trades"
        ],

        "VER_win": ver_stats[
            "win_rate"
        ],

        "VER_avg_R": ver_stats[
            "avg_R"
        ],

        "VER_total_R": ver_stats[
            "total_R"
        ],

        "VER_PF": ver_stats[
            "PF"
        ],

        "VER_DD": ver_stats[
            "max_DD_R"
        ],

        "robustness_score": score,
    }

    return result, trades_df


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    ensure_output_dir()

    print()
    print("=" * 70)
    print("BACKTEST V6 - ROBUSTNESS SCANNER")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    df = download_data()

    # --------------------------------------------------------
    # INDICATORI
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COSTRUZIONE INDICATORI")
    print("=" * 70)

    df = add_indicators(df)

    df = add_higher_timeframes(
        df
    )

    df = df.dropna(
        subset=[
            "ema20",
            "ema50",
            "rsi",
            "macd",
            "macd_signal",
            "atr",
            "trend_15m",
            "trend_1h",
        ]
    ).reset_index(
        drop=True
    )

    df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "candles_with_indicators.csv",
        ),
        index=False,
    )

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    split_index = int(
        len(df)
        * DEV_RATIO
    )

    split_time = df.iloc[
        split_index
    ]["datetime"]

    print()
    print(
        f"Development: 0 -> "
        f"{split_index}"
    )

    print(
        f"Verification: "
        f"{split_index} -> "
        f"{len(df)}"
    )

    print(
        f"Split time: "
        f"{split_time}"
    )

    # --------------------------------------------------------
    # CONFIGURAZIONI
    # --------------------------------------------------------

    configs = (
        generate_configs()
    )

    print()
    print(
        "Configurazioni da testare: "
        f"{len(configs)}"
    )

    # --------------------------------------------------------
    # SCANNER
    # --------------------------------------------------------

    results = []

    total = len(configs)

    print()
    print("=" * 70)
    print("SCANNER IN CORSO")
    print("=" * 70)

    for n, cfg in enumerate(
        configs,
        start=1,
    ):

        try:

            output = run_configuration(
                df,
                cfg,
                split_index,
            )

            if output is None:
                continue

            result, trades_df = output

            results.append(
                result
            )

        except Exception as error:

            print(
                f"Errore config {n}: "
                f"{error}"
            )

        if (
            n % 250 == 0
            or n == total
        ):

            elapsed = (
                time.time()
                - start_time
            )

            print(
                f"[{n}/{total}] "
                f"risultati={len(results)} "
                f"tempo={elapsed:.1f}s"
            )

    if not results:

        raise RuntimeError(
            "Nessuna configurazione valida trovata."
        )

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # RANKING
    # --------------------------------------------------------

    results_df = (
        results_df
        .sort_values(
            [
                "robustness_score",
                "VER_PF",
                "VER_avg_R",
                "VER_total_R",
            ],
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    results_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "robustness_ranking.csv",
        ),
        index=False,
    )

    # --------------------------------------------------------
    # ROBUST CANDIDATES
    # --------------------------------------------------------

    robust = results_df[
        (
            results_df[
                "DEV_trades"
            ]
            >= MIN_TRADES_DEV
        )
        &
        (
            results_df[
                "VER_trades"
            ]
            >= MIN_TRADES_VER
        )
        &
        (
            results_df[
                "DEV_PF"
            ]
            > 1.0
        )
        &
        (
            results_df[
                "VER_PF"
            ]
            > 1.0
        )
        &
        (
            results_df[
                "DEV_avg_R"
            ]
            > 0
        )
        &
        (
            results_df[
                "VER_avg_R"
            ]
            > 0
        )
        &
        (
            results_df[
                "DEV_total_R"
            ]
            > 0
        )
        &
        (
            results_df[
                "VER_total_R"
            ]
            > 0
        )
    ].copy()

    robust = (
        robust
        .sort_values(
            "robustness_score",
            ascending=False,
        )
    )

    robust.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "robust_candidates.csv",
        ),
        index=False,
    )

    # --------------------------------------------------------
    # TOP 30
    # --------------------------------------------------------

    results_df.head(
        30
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "top30.csv",
        ),
        index=False,
    )

    # --------------------------------------------------------
    # COLONNE DISPLAY
    # --------------------------------------------------------

    display_cols = [
        "direction",
        "ema_strength",
        "rsi_low",
        "rsi_high",
        "atr_regime",
        "trend_mode",
        "momentum_mode",
        "signal_mode",
        "body_filter",
        "horizon_minutes",
        "TP_ATR",
        "SL_ATR",
        "DEV_trades",
        "DEV_win",
        "DEV_avg_R",
        "DEV_PF",
        "VER_trades",
        "VER_win",
        "VER_avg_R",
        "VER_PF",
        "VER_total_R",
        "VER_DD",
        "robustness_score",
    ]

    # --------------------------------------------------------
    # TOP 20
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TOP 20 ROBUSTNESS SCANNER")
    print("=" * 70)

    print(
        results_df[
