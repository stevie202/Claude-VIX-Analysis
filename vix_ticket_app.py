"""
VIX vs Support Ticket Correlation — Streamlit App
==================================================
Requirements:
    pip install streamlit pandas numpy scipy matplotlib openpyxl

Run:
    streamlit run vix_ticket_app.py
"""

import io
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats
import streamlit as st

warnings.filterwarnings("ignore")

# ── Colour palette ──────────────────────────────────────────────────────────────
C_BLUE  = "#1A5276"
C_RED   = "#C0392B"
C_TEAL  = "#148F77"
C_GREY  = "#717D7E"
C_BG    = "#F8F9FA"
C_AMBER = "#F39C12"

# Label dictionaries — used everywhere so list-index bugs are impossible
DOW_LABELS  = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
DOW_FULL    = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday",
               5: "Saturday", 6: "Sunday"}
MON_LABELS  = {1: "Jan", 2: "Feb", 3: "Mar",  4: "Apr",  5: "May",  6: "Jun",
               7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
MON_FULL    = {1: "January", 2: "February",  3: "March",    4: "April",
               5: "May",     6: "June",       7: "July",     8: "August",
               9: "September", 10: "October", 11: "November", 12: "December"}


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_vix(file) -> pd.DataFrame:
    df = pd.read_excel(file, engine="openpyxl")
    df.columns = [c.strip().upper() for c in df.columns]
    date_col = next((c for c in df.columns if "DATE" in c), df.columns[0])
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], dayfirst=False, infer_datetime_format=True)
    for col in ["OPEN", "HIGH", "LOW", "CLOSE"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["date", "CLOSE"]).sort_values("date").reset_index(drop=True)


@st.cache_data
def load_tickets(file) -> pd.DataFrame:
    df = pd.read_excel(file, engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]
    date_col = next((c for c in df.columns if c.lower() in ("date", "day")), df.columns[0])
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], dayfirst=False, infer_datetime_format=True)
    for col in ["ticket_count", "active_client_count", "avg_ticket_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if ("avg_ticket_count" not in df.columns
            and "ticket_count" in df.columns
            and "active_client_count" in df.columns):
        df["avg_ticket_count"] = df["ticket_count"] / df["active_client_count"]
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# Analysis helpers
# ══════════════════════════════════════════════════════════════════════════════

def deseasonalise(df: pd.DataFrame, col: str) -> pd.Series:
    """Multiplicative deseasonalisation: remove day-of-week + month-of-year effects."""
    grand_mean = df[col].mean()
    dow_means  = df.groupby("dow")[col].transform("mean")
    mon_means  = df.groupby("month_num")[col].transform("mean")
    dow_factor = dow_means / grand_mean
    mon_factor = mon_means / grand_mean
    return df[col] / (dow_factor * mon_factor)


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def compute_ccf(x: pd.Series, y: pd.Series, max_lag: int = 10) -> pd.DataFrame:
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a = x.iloc[:lag].reset_index(drop=True)
            b = y.iloc[-lag:].reset_index(drop=True)
        elif lag == 0:
            a = x.reset_index(drop=True)
            b = y.reset_index(drop=True)
        else:
            a = x.iloc[lag:].reset_index(drop=True)
            b = y.iloc[:-lag].reset_index(drop=True)
        valid = pd.concat([a, b], axis=1).dropna()
        if len(valid) < 10:
            rows.append({"lag": lag, "r": np.nan, "p": np.nan})
            continue
        r, p = stats.pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
        rows.append({"lag": lag, "r": r, "p": p})
    return pd.DataFrame(rows)


def build_corr_table(df: pd.DataFrame, ticket_col: str) -> pd.DataFrame:
    vix_vars   = ["CLOSE", "vix_range", "vix_change", "vix_pct"]
    vix_labels = ["VIX Close (level)", "VIX Range H-L", "VIX Day Change", "VIX % Change"]
    rows = []
    for var, lbl in zip(vix_vars, vix_labels):
        if var not in df.columns:
            continue
        clean = df[[var, ticket_col]].dropna()
        if len(clean) < 5:
            continue
        pr, pp = stats.pearsonr(clean[var], clean[ticket_col])
        sr, sp = stats.spearmanr(clean[var], clean[ticket_col])
        rows.append({
            "VIX Metric":   lbl,
            "Pearson r":    round(pr, 3),
            "Pearson p":    round(pp, 4),
            "Spearman r":   round(sr, 3),
            "Spearman p":   round(sp, 4),
            "Significant?": "Yes" if pp < 0.05 else "No",
        })
    return pd.DataFrame(rows)


def strength_label(r: float):
    a = abs(r)
    if a >= 0.6:   return "strong"
    elif a >= 0.4: return "moderate"
    elif a >= 0.2: return "weak"
    else:          return "negligible"


def strength_icon(r: float):
    a = abs(r)
    if a >= 0.6:   return "🔴"
    elif a >= 0.4: return "🟠"
    elif a >= 0.2: return "🟡"
    else:          return "⚪"


# ══════════════════════════════════════════════════════════════════════════════
# Charts
# ══════════════════════════════════════════════════════════════════════════════

def fig_timeseries(df: pd.DataFrame):
    fig, ax1 = plt.subplots(figsize=(13, 3.8), facecolor=C_BG)
    ax2 = ax1.twinx()
    ax1.plot(df["date"], df["tickets_z"], color=C_TEAL, lw=1.3, alpha=0.85,
             label="Tickets - deseasonalised (z-score)")
    ax2.plot(df["date"], df["vix_z"],     color=C_RED,  lw=1.0, alpha=0.70,
             label="VIX Close (z-score)")
    ax1.set_ylabel("Tickets z-score", color=C_TEAL, fontsize=10)
    ax2.set_ylabel("VIX z-score",     color=C_RED,  fontsize=10)
    ax1.set_title("Deseasonalised Tickets vs VIX Close", fontsize=12, fontweight="bold")
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labs  = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labs, loc="upper right", fontsize=8)
    ax1.set_facecolor(C_BG)
    fig.tight_layout()
    return fig


def fig_vix_history(df: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 3.5), facecolor=C_BG)
    ax1.hist(df["CLOSE"], bins=40, color=C_BLUE, edgecolor="white", alpha=0.8)
    ax1.axvline(df["CLOSE"].median(), color=C_RED, lw=2, ls="--",
                label=f"Median {df['CLOSE'].median():.1f}")
    ax1.set_title("VIX Close Distribution", fontsize=11, fontweight="bold")
    ax1.set_xlabel("VIX Close")
    ax1.set_ylabel("Frequency")
    ax1.legend(fontsize=8)
    ax1.set_facecolor(C_BG)
    ax2.plot(df["date"], df["CLOSE"], color=C_RED, lw=0.9, alpha=0.8)
    ax2.fill_between(df["date"], df["LOW"], df["HIGH"],
                     color=C_RED, alpha=0.15, label="Daily H-L range")
    ax2.set_title("VIX Over Time (with H-L Band)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("VIX")
    ax2.legend(fontsize=8)
    ax2.set_facecolor(C_BG)
    fig.tight_layout()
    return fig


def fig_ccf(ccf_df: pd.DataFrame, ci: float):
    valid = ccf_df.dropna(subset=["r"])
    best_lag = int(valid.loc[valid["r"].abs().idxmax(), "lag"])
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=C_BG)
    colors = [C_RED if l == best_lag else (C_BLUE if (r > 0 and not np.isnan(r)) else C_GREY)
              for l, r in zip(ccf_df["lag"], ccf_df["r"])]
    ax.bar(ccf_df["lag"], ccf_df["r"].fillna(0), color=colors,
           edgecolor="white", linewidth=0.4)
    ax.axhline( ci, ls="--", lw=1, color="black", alpha=0.5, label=f"95% CI +/-{ci:.3f}")
    ax.axhline(-ci, ls="--", lw=1, color="black", alpha=0.5)
    ax.axhline(0,   lw=0.5,  color="black")
    ax.set_xlabel("Lag (trading days) - negative = VIX leads tickets", fontsize=9)
    ax.set_ylabel("Pearson r", fontsize=9)
    ax.set_title(f"Cross-Correlation Function  —  Peak at lag {best_lag}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_facecolor(C_BG)
    fig.tight_layout()
    return fig, best_lag


def fig_scatter(df: pd.DataFrame, ticket_col: str, vix_col: str = "CLOSE"):
    clean = df[[vix_col, ticket_col, "dow"]].dropna()
    slope, intercept, r_val, p_val, _ = stats.linregress(clean[vix_col], clean[ticket_col])
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=C_BG)
    sc = ax.scatter(clean[vix_col], clean[ticket_col],
                    c=clean["dow"], cmap="cool", alpha=0.45, s=16, edgecolors="none")
    x_fit = np.linspace(clean[vix_col].min(), clean[vix_col].max(), 200)
    ax.plot(x_fit, intercept + slope * x_fit, color=C_RED, lw=2,
            label=f"OLS  r2={r_val**2:.3f}  p={p_val:.4f}")
    plt.colorbar(sc, ax=ax, label="Day of week (0=Mon)")
    ax.set_xlabel("VIX Close", fontsize=10)
    ax.set_ylabel("Avg tickets/client (deseasonalised)", fontsize=10)
    ax.set_title("Scatter: VIX Level vs Deseasonalised Tickets",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_facecolor(C_BG)
    fig.tight_layout()
    return fig, r_val, p_val, slope, intercept


def fig_seasonal(df: pd.DataFrame, raw_col: str):
    """
    Seasonal bar charts. Uses DOW_LABELS / MON_LABELS dicts (never list indexing)
    so missing days or months never cause an IndexError.
    """
    dow_means = df.groupby("dow")[raw_col].mean()          # index = 0..4 ints
    mon_means = df.groupby("month_num")[raw_col].mean()    # index = 1..12 ints

    # Safe label mapping — .get() returns the key as string if not found
    dow_x = [DOW_LABELS.get(int(k), str(k)) for k in dow_means.index]
    mon_x = [MON_LABELS.get(int(k), str(k)) for k in mon_means.index]

    # Peak position is a positional index into .values, not the pandas index value
    peak_dow_pos = int(np.argmax(dow_means.values))
    peak_mon_pos = int(np.argmax(mon_means.values))

    bar_colors_dow = [C_AMBER if i == peak_dow_pos else C_TEAL
                      for i in range(len(dow_means))]
    bar_colors_mon = [C_AMBER if i == peak_mon_pos else C_BLUE
                      for i in range(len(mon_means))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 3.5), facecolor=C_BG)

    ax1.bar(dow_x, dow_means.values, color=bar_colors_dow, edgecolor="white")
    ax1.axhline(dow_means.mean(), color=C_RED, lw=1.5, ls="--", label="Mean")
    ax1.set_title("Day-of-Week Seasonal Pattern", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Mean avg tickets/client", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.set_facecolor(C_BG)

    ax2.bar(mon_x, mon_means.values, color=bar_colors_mon, edgecolor="white")
    ax2.axhline(mon_means.mean(), color=C_RED, lw=1.5, ls="--", label="Mean")
    ax2.set_title("Annual Seasonal Pattern", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Mean avg tickets/client", fontsize=9)
    ax2.tick_params(axis="x", rotation=45)
    ax2.legend(fontsize=8)
    ax2.set_facecolor(C_BG)

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Plain-English summary section
# ══════════════════════════════════════════════════════════════════════════════

def render_summary(df, corr_tbl, ccf_df, r_val, p_val, slope, ticket_col_raw, max_lag):

    # Pull correlation values safely
    def get_r(metric_label):
        row = corr_tbl[corr_tbl["VIX Metric"] == metric_label]
        return float(row["Pearson r"].values[0]) if len(row) else 0.0

    def get_p(metric_label):
        row = corr_tbl[corr_tbl["VIX Metric"] == metric_label]
        return float(row["Pearson p"].values[0]) if len(row) else 1.0

    r_close  = get_r("VIX Close (level)")
    r_range  = get_r("VIX Range H-L")
    r_change = get_r("VIX Day Change")
    p_close  = get_p("VIX Close (level)")

    valid_ccf = ccf_df.dropna(subset=["r"])
    best_lag  = int(valid_ccf.loc[valid_ccf["r"].abs().idxmax(), "lag"])
    ci        = 1.96 / np.sqrt(len(df))
    sig_lags  = ccf_df[ccf_df["r"].abs() > ci]["lag"].dropna().astype(int).tolist()

    # Seasonal stats — use dict lookups, never list indexing
    dow_means = df.groupby("dow")[ticket_col_raw].mean()
    mon_means = df.groupby("month_num")[ticket_col_raw].mean()

    peak_dow_key = int(dow_means.idxmax())
    low_dow_key  = int(dow_means.idxmin())
    peak_mon_key = int(mon_means.idxmax())
    low_mon_key  = int(mon_means.idxmin())

    peak_dow   = DOW_FULL.get(peak_dow_key, str(peak_dow_key))
    low_dow    = DOW_FULL.get(low_dow_key,  str(low_dow_key))
    peak_month = MON_FULL.get(peak_mon_key, str(peak_mon_key))
    low_month  = MON_FULL.get(low_mon_key,  str(low_mon_key))

    dow_swing = (dow_means.max() - dow_means.min()) / dow_means.mean() * 100
    mon_swing = (mon_means.max() - mon_means.min()) / mon_means.mean() * 100

    vix_mean   = df["CLOSE"].mean()
    vix_median = df["CLOSE"].median()
    date_start = df["date"].min().strftime("%d %b %Y")
    date_end   = df["date"].max().strftime("%d %b %Y")
    n_days     = len(df)

    st.markdown("---")
    st.markdown(
        "<h2 style='color:#1A5276'>Plain-English Summary</h2>"
        "<p style='color:#717D7E'>A jargon-free explanation of what the analysis found.</p>",
        unsafe_allow_html=True,
    )

    # 1. What was analysed
    st.markdown("### 1 · What was analysed")
    st.info(
        f"This analysis compared **{n_days:,} trading days** of data "
        f"from **{date_start}** to **{date_end}**. "
        f"The VIX (often called the 'fear index') measures how nervous financial markets are — "
        f"a higher number means more uncertainty. The average VIX over this period was "
        f"**{vix_mean:.1f}** (median **{vix_median:.1f}**). "
        f"Before comparing to VIX, the ticket data was cleaned of its regular weekly and "
        f"seasonal patterns, so those rhythms don't create false results."
    )

    # 2. Seasonal patterns
    st.markdown("### 2 · Seasonal patterns in ticket volume")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"**Day-of-week pattern**\n\n"
            f"Ticket volumes follow a clear weekly rhythm. **{peak_dow}** is the busiest day "
            f"and **{low_dow}** is the quietest — a swing of roughly **{dow_swing:.0f}%** "
            f"between the two. This pattern was removed before any VIX comparison."
        )
    with col2:
        st.markdown(
            f"**Time-of-year pattern**\n\n"
            f"Volumes also vary through the year. **{peak_month}** tends to be the busiest month "
            f"and **{low_month}** the quietest — a swing of roughly **{mon_swing:.0f}%**. "
            f"This annual cycle was also stripped out before the VIX comparison."
        )

    # 3. Is there a relationship?
    st.markdown("### 3 · Is there a relationship between VIX and ticket volume?")
    strength = strength_label(r_close)
    p_str = "< 0.0001" if p_close < 0.0001 else f"= {p_close:.4f}"
    if p_close < 0.05:
        st.success(
            f"**Yes — there is a statistically significant {strength} relationship.** "
            f"When market volatility is higher, support ticket volume also tends to be higher, "
            f"even after removing seasonal patterns. The correlation is **r = {r_close:.3f}** "
            f"(p {p_str}), which means the result is very unlikely to be due to chance."
        )
    else:
        st.warning(
            f"**No clear relationship was found after removing seasonal patterns.** "
            f"The correlation coefficient is **r = {r_close:.3f}** (p {p_str}), "
            f"which is {strength} and not statistically reliable. "
            f"VIX does not appear to be a meaningful driver of your ticket volume."
        )

    # 4. Which VIX measure matters most?
    st.markdown("### 4 · Which aspect of VIX matters most?")
    metrics_info = [
        ("The overall VIX level (close price)", r_close,
         "Captures the broad fear regime — whether markets are generally calm or stressed over days/weeks."),
        ("The daily high-low range",            r_range,
         "Captures intraday panic — how wild the swings were within a single day."),
        ("The day-over-day change in VIX",      r_change,
         "Captures sudden moves — whether fear spiked or dropped compared to yesterday."),
    ]
    for name, r, explanation in metrics_info:
        direction = "positive" if r >= 0 else "negative"
        st.markdown(
            f"**{strength_icon(r)} {name}** — {strength_label(r)} {direction} "
            f"link (r = {r:.3f}). {explanation}"
        )
    st.markdown(
        "> **Rule of thumb:** if the *level* matters most, clients react to sustained stress. "
        "If *daily change* matters most, they react to sudden shocks. "
        "If neither is significant, VIX is probably not a useful predictor."
    )

    # 5. Timing
    st.markdown("### 5 · Does market volatility come before the ticket spike?")
    # r value at the peak lag
    best_lag_r = float(valid_ccf.loc[valid_ccf["lag"] == best_lag, "r"].values[0])
    best_lag_r_str = f"r = {best_lag_r:.3f}"
    if best_lag < 0:
        st.success(
            f"**VIX moves before tickets do.** The strongest correlation occurs at a lag of "
            f"**{abs(best_lag)} trading day(s)** ({best_lag_r_str}) — a rise in VIX today "
            f"tends to be followed by more support tickets roughly {abs(best_lag)} day(s) later. "
            f"This suggests VIX could work as an **early warning signal** for upcoming spikes."
        )
    elif best_lag == 0:
        st.info(
            f"**VIX and tickets move together on the same day** ({best_lag_r_str}). "
            "There is no meaningful lead or lag. VIX is unlikely to be a forecasting tool "
            "here, but it does confirm a same-day co-movement."
        )
    else:
        st.warning(
            f"**Ticket volume appears to move before VIX** (peak at lag +{best_lag} days, "
            f"{best_lag_r_str}). This is unusual and worth investigating — it could indicate "
            "a data anomaly or that clients react to something that later flows through to markets."
        )
    sig_lag_str = ", ".join(str(l) for l in sorted(sig_lags)) if sig_lags else "none"
    st.caption(f"Lags outside the 95% confidence band: {sig_lag_str}")

    # 6. Practical meaning
    st.markdown("### 6 · What does this mean in practice?")
    if p_close < 0.05 and abs(r_close) >= 0.2:
        r2_pct = r_val ** 2 * 100
        st.markdown(
            f"Each **1-point rise in VIX** is associated with a change of **{slope:+.4f}** "
            f"in the average number of tickets per client per day (seasonally adjusted).\n\n"
            f"To put that in context: if VIX rises from 20 to 30 (a significant stress event), "
            f"you might expect roughly a **{slope * 10:+.3f} tickets/client/day** change.\n\n"
            f"VIX explains about **{r2_pct:.1f}%** of the day-to-day variation in your "
            f"deseasonalised ticket volume. The remaining {100 - r2_pct:.1f}% is driven "
            f"by other factors not captured here."
        )
    else:
        st.markdown(
            "The relationship is too weak to be practically useful for forecasting. "
            "Other factors — such as product releases, onboarding cycles, or operational "
            "changes — are likely more important drivers of your ticket volume."
        )

    # 7. Next steps
    st.markdown("### 7 · Recommended next steps")
    recs = []
    if p_close < 0.05 and abs(r_close) >= 0.3 and best_lag < 0:
        recs.append(
            f"**Build a simple alert.** If VIX rises above a threshold (e.g. > 25 or 30), "
            f"flag the next {abs(best_lag)} day(s) as elevated-risk and consider "
            f"pre-emptive staffing or proactive client comms."
        )
    if p_close < 0.05 and abs(r_close) >= 0.2:
        recs.append(
            "**Add VIX as a feature in a forecasting model.** Even a modest correlation "
            "can improve predictions when combined with day-of-week, month, and client count."
        )
    if abs(r_range) > abs(r_close):
        recs.append(
            "**Focus on intraday VIX range (H-L) rather than the close level.** "
            "Your data suggests daily volatility swings matter more than where VIX settled."
        )
    recs.append(
        "**Investigate other drivers.** VIX explains only part of your ticket variation. "
        "Consider overlaying product release dates, onboarding waves, or outage events."
    )
    recs.append(
        "**Download the deseasonalised CSV** (in the Correlation Table tab) to run "
        "further analysis or share with stakeholders."
    )
    for i, rec in enumerate(recs, 1):
        st.markdown(f"{i}. {rec}")


# ══════════════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="VIX vs Support Tickets",
    page_icon="📊",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 VIX vs Tickets")
    st.markdown("---")
    st.markdown("### 📁 Upload your files")

    vix_file = st.file_uploader(
        "VIX history  (VIXhistory.xlsx)",
        type=["xlsx", "xls"],
        help="Expected columns: DATE, OPEN, HIGH, LOW, CLOSE",
    )
    ticket_file = st.file_uploader(
        "Jira tickets  (Jira Daily Ticket Count.xlsx)",
        type=["xlsx", "xls"],
        help="Expected columns: Date, ticket_count, active_client_count, avg_ticket_count",
    )

    st.markdown("---")
    st.markdown("### Settings")
    max_lag = st.slider("Max CCF lag (trading days)", 5, 20, 10)
    use_avg = st.radio(
        "Ticket metric",
        ["avg_ticket_count (per client)", "ticket_count (raw)"],
        index=0,
        help="Per-client metric adjusts for changes in client count over time.",
    )
    ticket_col_raw = "avg_ticket_count" if "avg" in use_avg else "ticket_count"

    st.markdown("---")
    st.markdown("### 📅 Year filter")
    st.caption("Select which years to include in the analysis.")
    # Placeholder — will be replaced after data loads
    year_filter_placeholder = st.empty()

    st.markdown("---")
    st.markdown("### Correlation guide")
    st.markdown(
        "| r | Strength |\n|---|---|\n"
        "| 0.6+ | Strong 🔴 |\n"
        "| 0.4–0.6 | Moderate 🟠 |\n"
        "| 0.2–0.4 | Weak 🟡 |\n"
        "| < 0.2 | Negligible ⚪ |"
    )
    st.markdown(
        "**Lag guide**\n"
        "- Negative → VIX leads tickets\n"
        "- Zero → simultaneous\n"
        "- Positive → tickets lead VIX"
    )

# ── Page header ──────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='color:#1A5276;margin-bottom:0'>📊 VIX vs Support Ticket Correlation</h1>"
    "<p style='color:#717D7E;font-size:1.05rem;margin-top:4px'>"
    "Does market volatility drive support ticket volume? Upload your files to find out."
    "</p>"
    "<hr style='border:1px solid #D5D8DC;margin:12px 0 20px 0'>",
    unsafe_allow_html=True,
)

vix_ready    = vix_file    is not None
ticket_ready = ticket_file is not None

# ══════════════════════════════════════════════════════════════════════════════
# Upload landing page
# ══════════════════════════════════════════════════════════════════════════════

if not vix_ready or not ticket_ready:

    st.markdown("## Step 1 — Upload your two Excel files")
    st.markdown(
        "Use the **sidebar on the left** to upload both files. "
        "The analysis will run automatically once both are loaded."
    )

    col_a, col_b = st.columns(2)

    def file_card(title, filename, columns, uploaded):
        border = "#1E8449" if uploaded else "#C0392B"
        status = '<strong style="color:#1E8449">Uploaded</strong>' \
                 if uploaded else '<em style="color:#C0392B">Waiting for upload...</em>'
        cols_html = "".join(f"<li>{c}</li>" for c in columns)
        return (
            f"<div style='border:2px solid {border};border-radius:10px;"
            f"padding:22px;background:#fff;min-height:210px'>"
            f"<h3 style='margin:0 0 6px 0'>{title}</h3>"
            f"<p style='color:#717D7E;margin:0 0 14px 0'>File: <code>{filename}</code></p>"
            f"<p style='margin:0 0 6px 0'><strong>Required columns:</strong></p>"
            f"<ul style='margin:0 0 14px 0;padding-left:18px;color:#444'>{cols_html}</ul>"
            f"<p style='font-size:1.05rem;margin:0'>{status}</p>"
            f"</div>"
        )

    with col_a:
        st.markdown(file_card(
            "📈 VIX History File", "VIXhistory.xlsx",
            ["DATE", "OPEN", "HIGH", "LOW", "CLOSE"], vix_ready,
        ), unsafe_allow_html=True)

    with col_b:
        st.markdown(file_card(
            "🎫 Jira Ticket Count File", "Jira Daily Ticket Count.xlsx",
            ["Date", "ticket_count", "active_client_count", "avg_ticket_count", "month"],
            ticket_ready,
        ), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## What this tool does")
    steps = [
        ("1. Removes seasonal noise",
         "Strips out predictable weekly and annual patterns so they can't create "
         "misleading correlations with VIX."),
        ("2. Tests four VIX measures",
         "Compares your tickets against VIX level, intraday range, day-over-day change, "
         "and percentage change — each tells a different story."),
        ("3. Checks timing with cross-correlation",
         "Determines whether VIX rises before ticket spikes (making it a leading indicator), "
         "at the same time, or after."),
        ("4. Quantifies the relationship",
         "Runs Pearson and Spearman correlation tests with p-values so you know whether "
         "any link is statistically meaningful or could be random chance."),
        ("5. Delivers a plain-English summary",
         "Translates every chart and statistic into clear findings and actionable next "
         "steps — no data science background needed."),
    ]
    for title, desc in steps:
        st.markdown(f"**{title}** — {desc}")

    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# Load & process
# ══════════════════════════════════════════════════════════════════════════════

with st.spinner("Loading and processing files..."):
    vix_df    = load_vix(vix_file)
    ticket_df = load_tickets(ticket_file)

overlap_start = max(vix_df["date"].min(), ticket_df["date"].min())
overlap_end   = min(vix_df["date"].max(), ticket_df["date"].max())

if overlap_start >= overlap_end:
    st.error("The two files have no overlapping date range. Please check your data.")
    st.stop()

df = (pd.merge(vix_df, ticket_df, on="date", how="inner")
        .sort_values("date")
        .reset_index(drop=True))

df["dow"]        = df["date"].dt.dayofweek   # 0=Mon, 4=Fri
df["month_num"]  = df["date"].dt.month       # 1=Jan, 12=Dec
df["vix_range"]  = df["HIGH"] - df["LOW"]
df["vix_change"] = df["CLOSE"].diff()
df["vix_pct"]    = df["CLOSE"].pct_change() * 100

if ticket_col_raw not in df.columns:
    st.error(f"Column '{ticket_col_raw}' not found in the ticket file. "
             f"Available columns: {list(df.columns)}")
    st.stop()

df["year"] = df["date"].dt.year

# ── Year filter (rendered inside the sidebar placeholder) ────────────────────────
all_years     = sorted(df["year"].unique().tolist())
with year_filter_placeholder:
    selected_years = st.multiselect(
        "Years to include",
        options=all_years,
        default=all_years,
        help="Deselect years to exclude them from all charts and statistics.",
    )

if not selected_years:
    st.warning("No years selected — please select at least one year in the sidebar.")
    st.stop()

df = df[df["year"].isin(selected_years)].reset_index(drop=True)

# Re-compute derived columns on the filtered slice
df["vix_change"] = df["CLOSE"].diff()   # diff needs recalc after row removal
df["vix_pct"]    = df["CLOSE"].pct_change() * 100

df["tickets_deseas"] = deseasonalise(df, ticket_col_raw)
df["tickets_z"]      = zscore(df["tickets_deseas"])
df["vix_z"]          = zscore(df["CLOSE"])
ticket_col = "tickets_deseas"


# ── Dataset overview ─────────────────────────────────────────────────────────────
st.subheader("Dataset Overview")
years_str = ", ".join(str(y) for y in sorted(selected_years))
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Trading days", f"{len(df):,}")
c2.metric("Years included", years_str)
c3.metric("Date range",
          f"{df['date'].min().strftime('%d %b %Y')} → "
          f"{df['date'].max().strftime('%d %b %Y')}")
c4.metric("VIX Close mean", f"{df['CLOSE'].mean():.1f}")
c5.metric("Avg tickets/client mean", f"{df[ticket_col_raw].mean():.3f}")

with st.expander("Preview merged data (first 10 rows)"):
    show_cols = ["date", "OPEN", "HIGH", "LOW", "CLOSE", "vix_range",
                 ticket_col_raw, "tickets_deseas"]
    st.dataframe(df[[c for c in show_cols if c in df.columns]].head(10),
                 use_container_width=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# Pre-compute shared results (computed once, used in tabs + summary)
# ══════════════════════════════════════════════════════════════════════════════

ccf_df   = compute_ccf(df["CLOSE"], df[ticket_col], max_lag=max_lag)
ci       = 1.96 / np.sqrt(len(df))
corr_tbl = build_corr_table(df, ticket_col)
_, r_val, p_val, slope, intercept = fig_scatter(df, ticket_col)


# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Time Series",
    "🔁 Cross-Correlation",
    "🔵 Scatter & OLS",
    "📅 Seasonality",
    "📊 Correlation Table",
])

# ── Tab 1: Time series ───────────────────────────────────────────────────────────
with tab1:
    st.markdown("#### Deseasonalised Tickets vs VIX — Full Period")
    st.caption(
        "Both series shown as z-scores (0 = long-run average, +1 = one standard deviation "
        "above average). This lets two very different scales share one axis."
    )
    st.pyplot(fig_timeseries(df), use_container_width=True)
    st.markdown("#### VIX History")
    st.pyplot(fig_vix_history(df), use_container_width=True)

# ── Tab 2: CCF ───────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### Cross-Correlation Function (VIX Close vs Deseasonalised Tickets)")
    st.caption(
        "Negative lags = VIX leads tickets. Red bar = peak lag. "
        "Dashed lines = 95% significance bands."
    )
    ccf_fig, best_lag = fig_ccf(ccf_df, ci)
    st.pyplot(ccf_fig, use_container_width=True)
    if best_lag < 0:
        st.success(f"Peak at lag **{best_lag}** — VIX leads tickets by "
                   f"{abs(best_lag)} trading day(s).")
    elif best_lag == 0:
        st.info("Peak at lag **0** — VIX and tickets move together on the same day.")
    else:
        st.warning(f"Peak at lag **+{best_lag}** — tickets appear to lead VIX.")
    with st.expander("CCF values table"):
        st.dataframe(ccf_df.style.format({"r": "{:.3f}", "p": "{:.4f}"}),
                     use_container_width=True)

# ── Tab 3: Scatter ───────────────────────────────────────────────────────────────
with tab3:
    st.markdown("#### VIX Close vs Deseasonalised Tickets — Scatter + OLS")
    st.caption("Points coloured by day-of-week (0=Mon...4=Fri). Red line = OLS fit.")
    scatter_fig, r_val, p_val, slope, intercept = fig_scatter(df, ticket_col)
    st.pyplot(scatter_fig, use_container_width=True)
    s1, s2, s3 = st.columns(3)
    s1.metric("Pearson r2", f"{r_val**2:.3f}")
    s2.metric("OLS slope (per VIX point)", f"{slope:+.5f}")
    s3.metric("p-value", "< 0.0001" if p_val < 0.0001 else f"{p_val:.4f}")
    if p_val < 0.05:
        st.success(
            f"Statistically significant. Each 1-point VIX rise is associated with a "
            f"{slope:+.5f} change in deseasonalised avg tickets/client."
        )
    else:
        st.warning(f"Not statistically significant at p < 0.05 (p={p_val:.4f}).")

# ── Tab 4: Seasonality ───────────────────────────────────────────────────────────
with tab4:
    st.markdown("#### Seasonal Patterns Removed Before Correlation")
    st.caption(
        "Amber bars = peak period. These effects were stripped from the ticket series "
        "before any VIX comparison to prevent spurious correlations."
    )
    st.pyplot(fig_seasonal(df, ticket_col_raw), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        dow_tbl = (df.groupby("dow")[ticket_col_raw]
                     .agg(["mean", "std", "count"])
                     .reset_index())
        dow_tbl["dow"] = dow_tbl["dow"].map(DOW_LABELS).fillna(
                             dow_tbl["dow"].astype(str))
        dow_tbl.columns = ["Day", "Mean", "Std Dev", "N"]
        st.dataframe(
            dow_tbl.style.format({"Mean": "{:.3f}", "Std Dev": "{:.3f}"}),
            use_container_width=True,
        )
    with col_b:
        mon_tbl = (df.groupby("month_num")[ticket_col_raw]
                     .agg(["mean", "std", "count"])
                     .reset_index())
        mon_tbl["month_num"] = mon_tbl["month_num"].map(MON_LABELS).fillna(
                                   mon_tbl["month_num"].astype(str))
        mon_tbl.columns = ["Month", "Mean", "Std Dev", "N"]
        st.dataframe(
            mon_tbl.style.format({"Mean": "{:.3f}", "Std Dev": "{:.3f}"}),
            use_container_width=True,
        )

# ── Tab 5: Correlation table ─────────────────────────────────────────────────────
with tab5:
    st.markdown("#### Pearson & Spearman Correlations — All VIX Metrics")
    st.caption(
        "Green = r >= 0.4 (moderate+), Yellow = 0.2–0.4, Red = < 0.2. "
        "Both tests shown for robustness."
    )

    def colour_r(val):
        try:
            v = abs(float(val))
            if v >= 0.4:   return "background-color:#D5F5E3"
            elif v >= 0.2: return "background-color:#FEF9E7"
            else:          return "background-color:#FDEDEC"
        except Exception:
            return ""

    styled = (corr_tbl.style
              .applymap(colour_r, subset=["Pearson r", "Spearman r"])
              .format({"Pearson r":  "{:.3f}", "Pearson p":  "{:.4f}",
                       "Spearman r": "{:.3f}", "Spearman p": "{:.4f}"}))
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("---")
    with st.expander("Download deseasonalised data as CSV"):
        csv_cols = ["date", "OPEN", "HIGH", "LOW", "CLOSE", "vix_range",
                    "vix_change", "vix_pct", ticket_col_raw, "tickets_deseas",
                    "tickets_z", "vix_z", "dow", "month_num"]
        csv_buf = io.StringIO()
        df[[c for c in csv_cols if c in df.columns]].to_csv(csv_buf, index=False)
        st.download_button("Download CSV", csv_buf.getvalue(),
                           "vix_tickets_deseasonalised.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# Plain-English Summary
# ══════════════════════════════════════════════════════════════════════════════

render_summary(df, corr_tbl, ccf_df, r_val, p_val, slope, ticket_col_raw, max_lag)
