"""
BFX Funds Scanner
=================
Free, educational chart-screenshot scanner inspired by ICT / Smart Money Concepts
(Power of Three / PO3, Judas Swing, liquidity sweeps, market structure).

HOW IT WORKS (no paid AI / no external API):
  1. Candle detection  - finds green/red candles in the screenshot by colour and
                         rebuilds approximate O/H/L/C values (in pixel space).
  2. SMC logic         - swing points, market structure (HH/HL vs LH/LL), trend drift,
                         liquidity sweep (Judas swing), displacement, break of structure,
                         PO3 accumulation and premium/discount.
  3. Scoring           - 7-point checklist for BOTH directions; the stronger side wins.
  4. Safety gates      - Neutral + "WAIT - NO TRADE" unless confidence >= 60% AND all the
                         information needed (sweep, price calibration, R:R) is present.

Educational tool only. Not financial advice.

Run:  streamlit run app.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
DISCLAIMER = (
    "Educational tool only. Not financial advice. Trading involves substantial risk of "
    "loss. Past performance does not guarantee future results. You are solely "
    "responsible for your trading decisions."
)
MIN_CONFIDENCE = 60      # below this -> always Neutral / WAIT
MIN_RR = 1.5             # minimum acceptable reward:risk
MIN_CANDLES = 20         # need at least this many detected candles

# Timeframe -> (macro timeframe, micro timeframe) used in the strategy label
TF_MAP = {
    "M1": ("M15", "M1"), "M5": ("H1", "M5"), "M15": ("H1", "M15"),
    "M30": ("H4", "M30"), "H1": ("H4", "H1"), "H4": ("D1", "H4"), "D1": ("W1", "D1"),
}


# --------------------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------------------
@dataclass
class Result:
    direction: str = "Neutral"            # "Buy", "Sell" or "Neutral"
    confidence: int = 0                   # 0-100
    grade: str = "0/7"                    # checklist score of the leading side
    lean: str = "Balanced"                # raw lean even if trade is not allowed
    entry: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    rr: Optional[float] = None
    entry_type: str = ""
    checks: list = field(default_factory=list)    # [(label, bool)]
    reasons: list = field(default_factory=list)   # why we are Neutral
    explanation: str = ""
    overlay: dict = field(default_factory=dict)   # normalised levels for annotation


# --------------------------------------------------------------------------------------
# 1. Candle detection from a screenshot
# --------------------------------------------------------------------------------------
def detect_candles(img: Image.Image):
    """Detect bullish/bearish candles by colour. Returns (data_dict | None, resized_img).

    Prices are returned in *normalised* units: 0 = lowest wick, 1 = highest wick.
    Works best on a cropped chart (candles only) with green/teal bull and red bear candles.
    """
    img = img.convert("RGB")
    if img.width > 1400:  # keep processing fast on big phone screenshots
        img = img.resize((1400, int(img.height * 1400 / img.width)), Image.LANCZOS)

    a = np.asarray(img).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    bull = (g - r > 25) & (g > 90) & (g >= b - 15)                       # green / teal
    bear = (r - g > 50) & (r - b > 30) & (r > 120) & (np.abs(g - b) < 60)  # red
    mask = bull | bear
    if mask.mean() > 0.35 or mask.sum() < 50:   # nothing / background too colourful
        return None, img

    # A column is "active" if it holds candle pixels; contiguous active columns = 1 candle
    active = mask.sum(axis=0) >= 2
    runs, x, w_img = [], 0, img.width
    while x < w_img:
        if active[x]:
            x0 = x
            while x < w_img and active[x]:
                x += 1
            runs.append((x0, x - 1))
        else:
            x += 1
    if not runs:
        return None, img

    # Drop runs far wider than typical (legends, labels, merged blobs)
    widths = np.array([x1 - x0 + 1 for x0, x1 in runs])
    keep = widths <= max(4, 3 * np.median(widths))
    runs = [rn for rn, k in zip(runs, keep) if k]

    rows = []  # (xc, hi_y, lo_y, body_top_y, body_bot_y, is_bull)
    for x0, x1 in runs:
        sub = mask[:, x0:x1 + 1]
        ys = np.where(sub.any(axis=1))[0]
        counts = sub.sum(axis=1)
        body = np.where(counts >= 0.6 * counts.max())[0]
        is_bull = bull[:, x0:x1 + 1].sum() >= bear[:, x0:x1 + 1].sum()
        rows.append(((x0 + x1) / 2, ys.min(), ys.max(), body.min(), body.max(), is_bull))
    if len(rows) < 3:
        return None, img

    arr = np.array(rows, dtype=float)
    y_top, y_bot = arr[:, 1].min(), arr[:, 2].max()
    if y_bot - y_top < 10:
        return None, img

    def norm(y):  # pixel y (down) -> normalised price (up)
        return (y_bot - y) / (y_bot - y_top)

    is_b = arr[:, 5] == 1
    hi, lo = norm(arr[:, 1]), norm(arr[:, 2])
    btop, bbot = norm(arr[:, 3]), norm(arr[:, 4])
    o = np.where(is_b, bbot, btop)   # bull: opens at body bottom / bear: opens at body top
    c = np.where(is_b, btop, bbot)
    data = dict(x=arr[:, 0], o=o, h=hi, l=lo, c=c, y_top=y_top, y_bot=y_bot)
    return data, img


# --------------------------------------------------------------------------------------
# 2. Smart-money building blocks
#    Everything below is written for the LONG orientation. For SHORT we simply flip the
#    chart upside-down (1 - price, swap highs/lows) and reuse the same code.
# --------------------------------------------------------------------------------------
def swing_points(arr: np.ndarray, n: int = 3, high: bool = True) -> list:
    """Fractal swing highs/lows: extreme of the surrounding +/- n candles."""
    idx = []
    for i in range(n, len(arr) - n):
        w = arr[i - n:i + n + 1]
        if (high and arr[i] >= w.max()) or (not high and arr[i] <= w.min()):
            if idx and i - idx[-1] <= n:  # merge neighbours, keep the more extreme
                if (high and arr[i] > arr[idx[-1]]) or (not high and arr[i] < arr[idx[-1]]):
                    idx[-1] = i
            else:
                idx.append(i)
    return idx


def find_sweep(h, l, c, lookback: int = 30):
    """Sell-side liquidity sweep (Judas swing): a candle pierces a prior swing low, then
    price closes back above it within 3 candles. Returns (index, swept_level, extreme)."""
    n = len(c)
    lows = swing_points(l, 3, False)
    for i in range(n - 1, max(3, n - lookback) - 1, -1):
        for s in reversed(lows):
            if s >= i:
                continue
            if l[i] < l[s] - 0.004 and np.any(c[i:i + 3] > l[s]):
                return i, l[s], l[i:min(n, i + 4)].min()
    return None


def side_eval(o, h, l, c, sweep, is_long: bool):
    """Score one direction on a 7-point checklist. Returns (score, [(label, bool)])."""
    n = len(c)
    sh, sl = swing_points(h, 3, True), swing_points(l, 3, False)
    avg_body = max(float(np.mean(np.abs(c - o))), 1e-6)

    # 1. Market structure (HH+HL for longs, LH+LL for shorts)
    structure = len(sh) >= 2 and len(sl) >= 2 and h[sh[-1]] > h[sh[-2]] and l[sl[-1]] > l[sl[-2]]
    # 2. Trend drift over the last ~40 candles
    k = min(n, 40)
    drift = np.polyfit(np.arange(k), c[-k:], 1)[0] * k
    # 3-6. Sweep-dependent checks
    disp = bos = acc = False
    if sweep is not None:
        i = sweep[0]
        # Displacement: strong candle in trade direction after the sweep
        disp = bool(np.any((c[i:] - o[i:]) >= 1.5 * avg_body))
        # Break of structure: close above the last swing high that came before the sweep
        prior = [s for s in sh if s < i]
        bos = bool(prior and np.any(c[i + 1:] > h[prior[-1]])) if i + 1 < n else False
        # PO3 accumulation: tight range in the ~12 candles before the manipulation
        if i >= 6:
            seg = slice(max(0, i - 12), i)
            acc = (h[seg].max() - l[seg].min()) <= 0.45
    # 7. Price location: discount for longs (flipped = premium for shorts)
    location = c[-1] < 0.5

    d = ("HH + HL" if is_long else "LH + LL")
    checks = [
        (f"Market structure aligned ({d})", bool(structure)),
        ("Trend drift aligned (last ~40 candles)", bool(drift > 0.08)),
        (f"Liquidity sweep / Judas swing ({'sell-side' if is_long else 'buy-side'} taken)", sweep is not None),
        ("Displacement candle after the sweep", disp),
        ("Break of structure after the sweep", bos),
        ("PO3 accumulation range before the sweep", bool(acc)),
        (f"Price in {'discount (<50%)' if is_long else 'premium (>50%)'} of the range", bool(location)),
    ]
    return sum(ok for _, ok in checks), checks


def build_levels(o, h, l, c, sweep):
    """Entry / SL / TP in normalised units (long orientation). Returns (dict|None, reason)."""
    i, _, extreme = sweep
    sl = extreme - 0.25 * float(np.mean(h - l))      # stop just beyond the sweep extreme
    if c[-1] <= sl:
        return None, "Price already closed beyond the stop level - setup invalidated"
    peak = h[i:].max()
    eq = extreme + 0.5 * (peak - extreme)             # 50% of the displacement leg
    entry = min(c[-1], eq)
    entry_type = "Limit - wait for pullback" if c[-1] > eq else "At / near current price"
    risk = entry - sl
    if risk <= 0:
        return None, "Invalid risk distance"
    # Target: nearest swing-high liquidity giving >= MIN_RR, else the range high
    cands = sorted(h[s] for s in swing_points(h, 3, True) if h[s] > entry)
    tp = next((t for t in cands if (t - entry) / risk >= MIN_RR), float(h.max()))
    if tp <= entry:
        return None, "No liquidity target beyond the entry"
    return dict(entry=entry, sl=sl, tp=tp, rr=(tp - entry) / risk, entry_type=entry_type), ""


# --------------------------------------------------------------------------------------
# 3. Full analysis
# --------------------------------------------------------------------------------------
def analyze(cd: dict, p_low: float, p_high: float) -> Result:
    res = Result()
    o, h, l, c = cd["o"], cd["h"], cd["l"], cd["c"]
    if len(c) < MIN_CANDLES:
        res.reasons.append(f"Only {len(c)} candles detected (need {MIN_CANDLES}+). "
                           "Crop the screenshot to the candles and use green/red candles.")
        res.explanation = "Not enough readable candle data - information incomplete."
        return res

    long_args = (o, h, l, c)
    short_args = (1 - o, 1 - l, 1 - h, 1 - c)       # upside-down chart for shorts
    sw_l = find_sweep(long_args[1], long_args[2], long_args[3])
    sw_s = find_sweep(short_args[1], short_args[2], short_args[3])
    if sw_l and sw_s:                               # keep only the most recent sweep
        sw_s, sw_l = (None, sw_l) if sw_l[0] >= sw_s[0] else (sw_s, None)

    score_l, chk_l = side_eval(*long_args, sw_l, True)
    score_s, chk_s = side_eval(*short_args, sw_s, False)
    go_long = score_l >= score_s
    win, lose = (score_l, score_s) if go_long else (score_s, score_l)
    args, sweep, checks = (long_args, sw_l, chk_l) if go_long else (short_args, sw_s, chk_s)

    # Confidence: winning score minus half of the opposing score (penalises conflict)
    conf = int(round(max(0.0, 100 * (win - 0.5 * lose) / 7)))
    res.grade, res.checks = f"{win}/7", checks
    res.lean = "Balanced" if score_l == score_s else ("Bullish" if go_long else "Bearish")

    # ---- Safety gates: anything missing -> Neutral / WAIT ----
    reasons = []
    if conf < MIN_CONFIDENCE:
        reasons.append(f"Confidence {conf}% is below the {MIN_CONFIDENCE}% minimum")
    if sweep is None:
        reasons.append("No recent liquidity sweep (Judas swing) detected")
    if not (p_high > p_low > 0):
        reasons.append("Price calibration missing - Entry / SL / TP cannot be calculated")

    lv = None
    if not reasons:
        lv, why = build_levels(*args, sweep)
        if lv is None:
            reasons.append(why)
        elif lv["rr"] < MIN_RR:
            reasons.append(f"Reward:Risk {lv['rr']:.2f} is below the {MIN_RR} minimum")
            lv = None

    if reasons and conf >= MIN_CONFIDENCE:
        conf = MIN_CONFIDENCE - 1          # incomplete info can never reach 60%+
    res.confidence, res.reasons = conf, reasons

    if lv is not None:
        back = (lambda x: x) if go_long else (lambda x: 1 - x)   # undo the flip
        to_price = lambda x: p_low + back(x) * (p_high - p_low)
        res.direction = "Buy" if go_long else "Sell"
        res.entry, res.sl, res.tp = to_price(lv["entry"]), to_price(lv["sl"]), to_price(lv["tp"])
        res.rr, res.entry_type = lv["rr"], lv["entry_type"]
        res.overlay = {"ENTRY": back(lv["entry"]), "SL": back(lv["sl"]), "TP": back(lv["tp"])}

    res.explanation = build_explanation(res, go_long, sweep)
    return res


def build_explanation(res: Result, go_long: bool, sweep) -> str:
    ok = {label: v for label, v in res.checks}
    keys = list(ok.keys())
    structure, drift, swept, disp, bos, acc, loc = (ok[k] for k in keys)
    bias = "bullish" if go_long else "bearish"
    parts = [
        f"**1. Macro bias:** the strongest lean is **{bias}** "
        f"({'structure and trend agree' if structure and drift else 'structure/trend only partly agree' if structure or drift else 'structure and trend do not confirm it'}).",
        f"**2. Manipulation (Judas swing):** "
        + (f"{'sell-side' if go_long else 'buy-side'} liquidity was swept and price rejected back inside the range"
           if swept else "no clean liquidity sweep was found in the recent candles") + ".",
        f"**3. Distribution:** "
        + ("displacement " + ("and a break of structure " if bos else "") + "followed the sweep"
           if disp else "no convincing displacement after the sweep yet") + ".",
        f"**4. PO3 context:** "
        + ("a tight accumulation range preceded the sweep" if acc else "no clear accumulation range before the move")
        + f"; price is currently {'in the favourable half' if loc else 'in the unfavourable half'} of the range.",
    ]
    if res.direction == "Neutral":
        parts.append("**Decision:** WAIT - " + "; ".join(res.reasons) + ".")
    else:
        parts.append(f"**Decision:** {res.direction.upper()} setup passes all safety gates "
                     f"(confidence {res.confidence}%, R:R {res.rr:.2f}).")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------------------
# 4. Annotated screenshot
# --------------------------------------------------------------------------------------
def annotate(img: Image.Image, cd: dict, res: Result) -> Image.Image:
    im = img.copy()
    d = ImageDraw.Draw(im)
    yt, yb = cd["y_top"], cd["y_bot"]
    to_y = lambda v: yb - v * (yb - yt)
    for i in swing_points(cd["h"], 3, True):    # swing highs (yellow)
        x, y = cd["x"][i], to_y(cd["h"][i])
        d.ellipse([x - 4, y - 4, x + 4, y + 4], outline=(255, 215, 0), width=2)
    for i in swing_points(cd["l"], 3, False):   # swing lows (cyan)
        x, y = cd["x"][i], to_y(cd["l"][i])
        d.ellipse([x - 4, y - 4, x + 4, y + 4], outline=(0, 220, 255), width=2)
    colors = {"ENTRY": (255, 255, 255), "SL": (255, 80, 80), "TP": (80, 255, 120)}
    for name, v in res.overlay.items():
        y = min(max(to_y(v), 0), im.height - 1)
        d.line([(0, y), (im.width, y)], fill=colors[name], width=2)
        d.text((6, max(y - 12, 0)), name, fill=colors[name])
    return im


# --------------------------------------------------------------------------------------
# 5. Helpers for the UI
# --------------------------------------------------------------------------------------
def fmt(p: Optional[float]) -> str:
    """Format a price with sensible decimals."""
    if p is None:
        return "-"
    if abs(p) >= 1000:
        return f"{p:,.2f}"
    return f"{p:.3f}" if abs(p) >= 10 else f"{p:.5f}"


def sidebar_calculator():
    """Position size calculator (sidebar)."""
    sb = st.sidebar
    sb.header("🧮 Position Size Calculator")
    ccy = sb.selectbox("Account currency", ["USD", "ZAR"])
    balance = sb.number_input(f"Account balance ({ccy})", min_value=0.0, value=1000.0 if ccy == "USD" else 20000.0,
                              step=100.0, key=f"bal_{ccy}")
    risk_pct = sb.number_input("Risk per trade (%)", min_value=0.1, max_value=2.0, value=0.5, step=0.1,
                               help="Capped at 2% - keep risk small.")
    inst = sb.selectbox("Instrument type", [
        "Forex (USD-quoted) - $10 / pip / lot",
        "XAUUSD Gold (pip = 0.10) - $10 / pip / lot",
        "BTCUSD (point = $1) - $1 / point / lot",
        "Custom pip value",
    ])
    pip_value = {"Forex": 10.0, "XAUUSD": 10.0, "BTCUSD": 1.0}.get(inst.split(" ")[0], None)
    if pip_value is None:
        pip_value = sb.number_input("Value of 1 pip/point per 1.00 lot (USD)", min_value=0.0001, value=10.0)
    sl_pips = sb.number_input("Stop loss distance (pips / points)", min_value=0.1, value=20.0, step=1.0)
    rate = 1.0
    if ccy == "ZAR":
        rate = sb.number_input("USD/ZAR rate (update to live rate)", min_value=1.0, value=18.0, step=0.1)

    risk_amt = balance * risk_pct / 100                      # in account currency
    risk_usd = risk_amt / rate                               # convert to USD
    raw_lots = risk_usd / (sl_pips * pip_value)
    lots = math.floor(raw_lots * 100) / 100                  # round DOWN to 0.01
    real_risk = lots * sl_pips * pip_value * rate            # risk after rounding

    c1, c2 = sb.columns(2)
    c1.metric("Lot size", f"{lots:.2f}")
    c2.metric("Risk amount", f"{ccy} {risk_amt:,.2f}")
    if lots < 0.01:
        sb.warning("Calculated size is below 0.01 lot - lower the stop distance or raise the balance.")
    else:
        sb.caption(f"Actual risk after rounding: {ccy} {real_risk:,.2f}. Pip values differ by broker - verify yours.")


def render_result(scan: dict):
    """Draw the result card."""
    res: Result = scan["res"]

    # Direction banner (always explicit: Long or Short)
    if res.direction == "Buy":
        st.success("## 🟢 BUY (LONG)")
    elif res.direction == "Sell":
        st.error("## 🔴 SELL (SHORT)")
    else:
        st.warning("## ⚪ NEUTRAL")

    c1, c2, c3 = st.columns(3)
    c1.metric("Confidence", f"{res.confidence}%")
    c2.metric("Setup Grade", res.grade)
    c3.metric("Risk : Reward", f"1 : {res.rr:.2f}" if res.rr else "-")

    c4, c5, c6 = st.columns(3)
    c4.metric("Entry", fmt(res.entry))
    c5.metric("Stop Loss", fmt(res.sl))
    c6.metric("Take Profit", fmt(res.tp))

    # Strategy label from timeframe
    macro, micro = TF_MAP.get(scan["tf"], (None, None))
    strategy = f"PO3 {macro} macro / {micro} micro" if macro else "PO3 / Judas swing (timeframe not set)"
    st.info(f"**Strategy:** {strategy}" + (f"  |  **Pair:** {scan['pair']}" if scan["pair"] else ""))

    # Trigger / invalidation / action
    if res.direction == "Neutral":
        st.info("**Next Trigger:** a clear liquidity sweep of a swing high/low, displacement away from it "
                "and a break of structure - then rescan with a fresh screenshot.")
        st.info("**Invalidation:** n/a - no valid setup.")
        st.warning("**Best Action Now:** ⛔ WAIT – NO TRADE")
        if res.reasons:
            st.caption("Why Neutral: " + " · ".join(res.reasons))
    else:
        side = "LONG" if res.direction == "Buy" else "SHORT"
        cmp_ = "above" if res.direction == "Buy" else "below"
        st.info(f"**Next Trigger:** price reaches the entry zone ({res.entry_type}) and shows micro-timeframe "
                f"confirmation (market-structure shift / fair value gap) while holding {cmp_} {fmt(res.sl)}.")
        st.info(f"**Invalidation:** a candle close beyond {fmt(res.sl)} cancels the {side} idea.")
        st.success(f"**Best Action Now:** Consider a **{side}** at {fmt(res.entry)} · SL {fmt(res.sl)} · "
                   f"TP {fmt(res.tp)}. Size it with the sidebar calculator and only after confirmation.")

    with st.expander("📊 Top-down analysis", expanded=True):
        st.markdown(res.explanation)
        st.markdown("**Checklist (leading side):**")
        st.markdown("\n".join(f"- {'✅' if ok else '⬜'} {label}" for label, ok in res.checks))

    st.image(scan["annotated"], caption="Detected swings (yellow = highs, cyan = lows) and trade levels",
             use_container_width=True)
    st.warning(DISCLAIMER)


# --------------------------------------------------------------------------------------
# 6. App
# --------------------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="BFX Funds Scanner", page_icon="📈", layout="centered",
                       initial_sidebar_state="expanded")
    st.markdown("<style>.block-container{padding-top:1.5rem;max-width:900px}</style>", unsafe_allow_html=True)

    # Sidebar: disclaimer is always visible, then the calculator
    st.sidebar.warning("⚠️ **DISCLAIMER**\n\n" + DISCLAIMER)
    sidebar_calculator()

    st.title("📈 BFX Funds Scanner")
    st.caption("Free educational chart scanner · ICT / Smart Money Concepts · supports Long (Buy) and Short (Sell)")
    st.warning(DISCLAIMER)

    file = st.file_uploader("Upload a chart screenshot (PNG / JPG)", type=["png", "jpg", "jpeg"])
    col_a, col_b = st.columns(2)
    pair = col_a.text_input("Pair (optional)", placeholder="BTCUSD, XAUUSD, EURUSD").strip().upper()
    tf = col_b.selectbox("Timeframe (optional)", ["Not set", "M1", "M5", "M15", "M30", "H1", "H4", "D1"])

    with st.expander("🎯 Price calibration (needed for Entry / SL / TP)"):
        st.caption("Crop the screenshot to the candles only. Then enter the price of the **highest wick** and "
                   "the **lowest wick** visible. Without these, the scanner stays Neutral.")
        k1, k2 = st.columns(2)
        p_high = k1.number_input("Highest wick price", min_value=0.0, value=0.0, format="%.5f")
        p_low = k2.number_input("Lowest wick price", min_value=0.0, value=0.0, format="%.5f")

    if st.button("🔍 Scan Chart", type="primary", use_container_width=True):
        if file is None:
            st.error("Please upload a chart screenshot first.")
        else:
            try:
                with st.spinner("Scanning chart..."):
                    img = Image.open(file)
                    cd, img = detect_candles(img)
                    if cd is None:
                        res = Result(reasons=["No candles detected - use a clean screenshot with green/red candles"],
                                     explanation="Could not read any candle data - information incomplete.")
                        annotated = img
                    else:
                        res = analyze(cd, p_low, p_high)
                        annotated = annotate(img, cd, res)
                st.session_state["scan"] = dict(res=res, annotated=annotated, pair=pair,
                                                tf=tf if tf != "Not set" else "")
            except Exception as e:  # any failure -> safe fallback
                st.session_state.pop("scan", None)
                st.error(f"Could not process this image ({e}). Result: WAIT – NO TRADE.")

    # Results persist across sidebar edits via session_state
    if "scan" in st.session_state:
        st.divider()
        render_result(st.session_state["scan"])

    st.divider()
    st.caption("BFX Funds Scanner uses colour-based candle detection, so accuracy depends on screenshot quality. "
               "Always verify on your own chart.")
    st.caption(DISCLAIMER)


if __name__ == "__main__":
    main()
