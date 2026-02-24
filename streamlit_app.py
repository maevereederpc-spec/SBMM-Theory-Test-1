import streamlit as st
import pandas as pd
import numpy as np
import random
import io
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from itertools import combinations

matplotlib.use("Agg")

# =============================================================================
#  DESIGN TOKENS
# =============================================================================
WINE       = "#722F37"
WINE_LIGHT = "#9B4A54"
WINE_DARK  = "#4A1520"
WINE_GLOW  = "#C0666F"
GOLD       = "#C9A84C"
CREAM      = "#F5ECD7"
BG_DARK    = "#0F0A0B"
BG_MID     = "#1A1115"
BG_CARD    = "#221519"
BG_PANEL   = "#2A1B1E"
BORDER     = "#3D2226"
TEXT_PRI   = "#F0E4E6"
TEXT_SEC   = "#9A7A7E"
TEAM_A_COL = "#722F37"
TEAM_B_COL = "#2F4472"

# =============================================================================
#  WIN FORMULA — Variance-Adjusted Logistic Model
#
#  Two-stage process:
#
#  STAGE 1 — Effective Team Skill (ETS)
#    ETS = mean(skills) − α × std(skills)     where α = 0.35
#
#    The standard deviation of a team's skill levels is a direct measure
#    of internal imbalance. A team with one standout player and several weak
#    players will have a high std — but in a 6v6 format, all six players
#    compete simultaneously. The strong player cannot compensate for five
#    players losing their individual matchups at the same time.
#    Subtracting α×std "discounts" the team's mean by how uneven it is.
#    A perfectly uniform team (all same skill) has std=0 and loses nothing.
#
#  STAGE 2 — Logistic Win Probability
#    P(A wins) = 1 / (1 + 10 ^ (−(ETS_A − ETS_B) / K))    where K = 12
#
#    This is a sigmoid (S-curve) that maps any ETS difference to a win
#    probability between 0% and 100%. K controls how steep the curve is:
#    a larger K flattens it (skill matters less), a smaller K steepens it.
#    K = 12 was chosen so that the most extreme possible matchup —
#    a full team of skill-10 players vs a full team of skill-1 players —
#    produces exactly 84.9%, keeping all outputs below the 85% ceiling.
#
#  KEY PROPERTIES
#    • Same ETS on both sides → always exactly 50%
#    • Symmetric: P(A wins) + P(B wins) = 100% always
#    • Carry penalty: [10,2,2,2,2,2] vs [3,3,4,4,3,3] → even team is favourite
#    • Hard cap: win% never exceeds 84.9% regardless of input
# =============================================================================

K     = 12
ALPHA = 0.35


def effective_team_skill(skills: list, alpha: float = ALPHA) -> float:
    arr = np.array(skills, dtype=float)
    return float(arr.mean() - alpha * arr.std())


def win_chance_teams(skills_a: list, skills_b: list, alpha: float = ALPHA) -> float:
    ets_a = effective_team_skill(skills_a, alpha)
    ets_b = effective_team_skill(skills_b, alpha)
    return 1.0 / (1.0 + 10.0 ** (-(ets_a - ets_b) / K))


def balance_score_from_wc(wc: float) -> float:
    return 1.0 - abs(wc - 0.5)


def generate_skill_level() -> int:
    return max(1, min(10, int(round(random.gauss(5, 1.8)))))


# =============================================================================
#  MATCHMAKING — Optimal 6v6 split via exhaustive search
#
#  For each group of 12 players (sorted by skill, so similar-skill players
#  always face each other), all C(12,6) = 924 possible 6-player splits are
#  evaluated. The split whose win probability is closest to 50% — as judged
#  by the variance-adjusted formula — is chosen. This guarantees the globally
#  optimal team composition, not just a heuristic estimate.
# =============================================================================

def optimal_split(group_skills: list, group_players: list, alpha: float = ALPHA):
    indices   = list(range(12))
    best_bs   = -1.0
    best_combo = None
    for combo in combinations(indices, 6):
        ta = [group_skills[i] for i in combo]
        tb = [group_skills[i] for i in indices if i not in combo]
        bs = balance_score_from_wc(win_chance_teams(ta, tb, alpha))
        if bs > best_bs:
            best_bs    = bs
            best_combo = combo
    idx_a = list(best_combo)
    idx_b = [i for i in indices if i not in best_combo]
    ta    = [group_skills[i] for i in idx_a]
    tb    = [group_skills[i] for i in idx_b]
    wc    = win_chance_teams(ta, tb, alpha)
    return idx_a, idx_b, wc, balance_score_from_wc(wc)


def matchmake_6v6(players_df: pd.DataFrame, alpha: float = ALPHA):
    df        = players_df.copy().sort_values("Skill", ascending=False).reset_index(drop=True)
    n_matches = len(df) // 12
    benched   = list(df.iloc[n_matches * 12:]["Player"])
    matches, rosters = [], []

    for m in range(n_matches):
        group     = df.iloc[m * 12 : (m + 1) * 12].reset_index(drop=True)
        g_skills  = list(group["Skill"].astype(int))
        g_players = list(group["Player"])

        idx_a, idx_b, wc, bs = optimal_split(g_skills, g_players, alpha)

        ta_s = [g_skills[i]  for i in idx_a]
        tb_s = [g_skills[i]  for i in idx_b]
        ta_p = [g_players[i] for i in idx_a]
        tb_p = [g_players[i] for i in idx_b]

        ets_a = effective_team_skill(ta_s, alpha)
        ets_b = effective_team_skill(tb_s, alpha)
        mn    = m + 1

        matches.append({
            "Match":          mn,
            "Team A Avg":     round(float(np.mean(ta_s)), 2),
            "Team B Avg":     round(float(np.mean(tb_s)), 2),
            "Team A Std":     round(float(np.std(ta_s)),  2),
            "Team B Std":     round(float(np.std(tb_s)),  2),
            "ETS A":          round(ets_a, 2),
            "ETS B":          round(ets_b, 2),
            "ETS Diff":       round(abs(ets_a - ets_b), 2),
            "Win % (A)":      round(wc * 100, 2),
            "Win % (B)":      round((1 - wc) * 100, 2),
            "Balance Score":  round(bs * 100, 2),
        })
        for p, s in zip(ta_p, ta_s):
            rosters.append({"Match": mn, "Team": "A", "Player": p, "Skill": s})
        for p, s in zip(tb_p, tb_s):
            rosters.append({"Match": mn, "Team": "B", "Player": p, "Skill": s})

    return matches, rosters, benched


# =============================================================================
#  EXCEL / DATA HELPERS
# =============================================================================

def read_excel_frequency(file) -> pd.DataFrame:
    df = pd.read_excel(file)
    df.columns = [c.strip().title() for c in df.columns]
    if "Skill" not in df.columns or "Frequency" not in df.columns:
        raise ValueError("Excel must have 'Skill' and 'Frequency' columns.")
    df["Skill"]     = pd.to_numeric(df["Skill"],     errors="coerce").clip(1, 10).fillna(5).astype(int)
    df["Frequency"] = pd.to_numeric(df["Frequency"], errors="coerce").fillna(0).astype(int)
    df = df[df["Frequency"] > 0]
    if df.empty:
        raise ValueError("No valid rows — check that Frequency values are positive integers.")
    rows = []
    for _, row in df.iterrows():
        rows.extend([int(row["Skill"])] * int(row["Frequency"]))
    out          = pd.DataFrame({"Skill": rows})
    out["Player"] = [f"Player_{i+1:03d}" for i in range(len(out))]
    return out[["Player", "Skill"]]


def frequency_table(players_df: pd.DataFrame) -> pd.DataFrame:
    f = players_df["Skill"].value_counts().sort_index().reset_index()
    f.columns = ["Skill", "Frequency"]
    return f


def create_sample_excel() -> bytes:
    data = {"Skill": list(range(1, 11)),
            "Frequency": [2, 3, 6, 10, 14, 12, 8, 5, 3, 1]}
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pd.DataFrame(data).to_excel(w, index=False, sheet_name="Players")
    return buf.getvalue()


def results_to_excel(matches, rosters, benched, players_df) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pd.DataFrame(matches).to_excel(w, index=False, sheet_name="Match Summaries")
        pd.DataFrame(rosters).to_excel(w, index=False, sheet_name="Team Rosters")
        frequency_table(players_df).to_excel(w, index=False, sheet_name="Skill Frequency")
        if benched:
            pd.DataFrame({"Benched": benched}).to_excel(w, index=False, sheet_name="Benched")
    return buf.getvalue()


# =============================================================================
#  MATPLOTLIB CHART HELPERS
# =============================================================================

def _fig_style():
    fig, ax = plt.subplots()
    fig.patch.set_facecolor(BG_CARD)
    ax.set_facecolor(BG_CARD)
    ax.tick_params(colors=TEXT_SEC, labelsize=9)
    ax.spines["bottom"].set_color(BORDER)
    ax.spines["left"].set_color(BORDER)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def chart_skill_distribution(players_df: pd.DataFrame):
    counts = [players_df[players_df["Skill"] == s].shape[0] for s in range(1, 11)]
    fig, ax = _fig_style()
    bars = ax.bar(range(1, 11), counts, color=WINE, edgecolor=WINE_DARK, linewidth=0.8, width=0.7)
    for bar, c in zip(bars, counts):
        if c > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(c), ha="center", va="bottom", color=TEXT_PRI, fontsize=9, fontweight="bold")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Skill Level", color=TEXT_SEC, fontsize=9)
    ax.set_ylabel("Player Count", color=TEXT_SEC, fontsize=9)
    ax.yaxis.label.set_color(TEXT_SEC)
    fig.tight_layout()
    return fig


def chart_ets_curve():
    """Show how the variance penalty works across different std values."""
    std_vals = np.linspace(0, 4.5, 200)
    mean_val = 5.0
    ets_vals = mean_val - ALPHA * std_vals

    fig, ax = _fig_style()
    ax.fill_between(std_vals, ets_vals, mean_val, alpha=0.15, color=WINE)
    ax.plot(std_vals, ets_vals, color=WINE_GLOW, linewidth=2.5, label="ETS (penalised)")
    ax.axhline(mean_val, color=GOLD, linewidth=1.5, linestyle="--", label=f"Raw mean = {mean_val}")
    ax.set_xlabel("Team Skill Std Dev  (spread of player skill)", color=TEXT_SEC, fontsize=9)
    ax.set_ylabel("Effective Team Skill (ETS)", color=TEXT_SEC, fontsize=9)

    # annotate two example points
    for std, label in [(0, "All equal\n(no penalty)"), (2.98, "Carry team\n[10,2,2,2,2,2]")]:
        ets = mean_val - ALPHA * std
        ax.scatter([std], [ets], color=GOLD, s=60, zorder=5)
        ax.annotate(f"std={std:.1f}\nETS={ets:.2f}", xy=(std, ets),
                    xytext=(std + 0.2, ets - 0.35),
                    color=CREAM, fontsize=8,
                    arrowprops=dict(arrowstyle="->", color=TEXT_SEC, lw=1))

    ax.legend(frameon=False, labelcolor=TEXT_SEC, fontsize=8)
    fig.tight_layout()
    return fig


def chart_win_curve():
    """S-curve: ETS difference → win probability."""
    diffs = np.linspace(-9, 9, 300)
    probs = 1 / (1 + 10 ** (-diffs / K))

    fig, ax = _fig_style()
    ax.fill_between(diffs, 0.5, probs, where=probs >= 0.5, alpha=0.12, color=WINE)
    ax.fill_between(diffs, probs, 0.5, where=probs <  0.5, alpha=0.12, color=TEAM_B_COL)
    ax.plot(diffs, probs * 100, color=WINE_GLOW, linewidth=2.5)
    ax.axhline(50, color=GOLD, linewidth=1, linestyle="--")
    ax.axhline(85, color=TEXT_SEC, linewidth=0.8, linestyle=":", alpha=0.6)
    ax.text(8.5, 86, "85% cap", color=TEXT_SEC, fontsize=7, ha="right")
    ax.axvline(0, color=BORDER, linewidth=1)
    ax.set_xlabel("ETS_A − ETS_B", color=TEXT_SEC, fontsize=9)
    ax.set_ylabel("Win Probability for Team A (%)", color=TEXT_SEC, fontsize=9)
    ax.set_ylim(10, 95)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    fig.tight_layout()
    return fig


def chart_team_comparison(match_row, roster_df):
    """
    Horizontal bar comparison of both teams' player skills.
    Height scales with the number of players per team (always 6, but future-proof).
    """
    mn      = match_row["Match"]
    ta_rows = roster_df[(roster_df["Match"] == mn) & (roster_df["Team"] == "A")].sort_values("Skill", ascending=False)
    tb_rows = roster_df[(roster_df["Match"] == mn) & (roster_df["Team"] == "B")].sort_values("Skill", ascending=False)

    n_players = max(len(ta_rows), len(tb_rows))
    fig_h     = max(2.2, n_players * 0.42 + 0.8)
    bar_h     = min(0.55, max(0.25, 0.55 - max(0, n_players - 6) * 0.05))
    lbl_fs    = max(7, 8.5 - max(0, n_players - 6) * 0.2)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7, fig_h), sharey=False)
    fig.patch.set_facecolor(BG_PANEL)

    for ax, rows, color, label, ets_val, win_val in [
        (ax_a, ta_rows, TEAM_A_COL, "Team A", match_row["ETS A"], match_row["Win % (A)"]),
        (ax_b, tb_rows, TEAM_B_COL, "Team B", match_row["ETS B"], match_row["Win % (B)"]),
    ]:
        ax.set_facecolor(BG_PANEL)
        # Shorten player names to avoid overflow
        names  = [r["Player"].replace("Player_", "P") for _, r in rows.iterrows()]
        skills = [r["Skill"] for _, r in rows.iterrows()]

        bars = ax.barh(names, skills, color=color, height=bar_h, edgecolor=BG_DARK, linewidth=0.4)
        ax.set_xlim(0, 12.5)
        ax.set_xlabel("Skill", color=TEXT_SEC, fontsize=7.5)
        ax.set_title(f"{label}  |  ETS {ets_val:.2f}  |  Win% {win_val:.1f}%",
                     color=TEXT_PRI, fontsize=8, pad=5)
        ax.tick_params(colors=TEXT_SEC, labelsize=lbl_fs)
        ax.spines["top"].set_visible(False);  ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(BORDER); ax.spines["left"].set_color(BORDER)

        for bar, skill in zip(bars, skills):
            ax.text(skill + 0.2, bar.get_y() + bar.get_height()/2,
                    str(skill), va="center", color=TEXT_PRI, fontsize=lbl_fs, fontweight="bold")

    fig.tight_layout(pad=0.9)
    return fig


def chart_balance_overview(match_df):
    """Horizontal bar per match coloured by balance score. Scales cleanly to any number of matches."""
    n       = len(match_df)
    row_h   = 0.52          # inches per row
    fig_h   = max(2.8, n * row_h + 1.2)   # +1.2 for legend + axes padding
    lbl_fs  = max(6.5, min(9, 9 - n * 0.06))  # shrink label font at scale
    bar_h   = min(0.62, max(0.28, 0.62 - n * 0.008))

    fig, ax = plt.subplots(figsize=(6.5, fig_h))
    fig.patch.set_facecolor(BG_CARD)
    ax.set_facecolor(BG_CARD)

    labels = [f"M{r['Match']}" for _, r in match_df.iterrows()]
    scores = match_df["Balance Score"].tolist()
    colors = [
        "#3a7d44" if s >= 95 else
        "#6aab6a" if s >= 80 else
        "#c9a84c" if s >= 60 else
        WINE
        for s in scores
    ]

    bars = ax.barh(labels[::-1], scores[::-1], color=colors[::-1],
                   height=bar_h, edgecolor=BG_DARK, linewidth=0.4)
    ax.axvline(95, color=TEXT_SEC, linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_xlim(40, 108)
    ax.set_xlabel("Balance Score (%)", color=TEXT_SEC, fontsize=8)
    ax.tick_params(colors=TEXT_SEC, labelsize=lbl_fs)
    ax.spines["bottom"].set_color(BORDER)
    ax.spines["left"].set_color(BORDER)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # only label every bar if few matches; thin label every other if many
    for i, (bar, s) in enumerate(zip(bars, scores[::-1])):
        if n <= 30 or i % 2 == 0:
            ax.text(s + 0.4, bar.get_y() + bar.get_height()/2,
                    f"{s:.1f}%", va="center", color=TEXT_PRI, fontsize=max(6, lbl_fs - 0.5))

    legend_patches = [
        mpatches.Patch(color="#3a7d44", label="Perfect (≥95%)"),
        mpatches.Patch(color="#6aab6a", label="Good (80–95%)"),
        mpatches.Patch(color=GOLD,      label="Fair (60–80%)"),
        mpatches.Patch(color=WINE,      label="Poor (<60%)"),
    ]
    ax.legend(handles=legend_patches, frameon=False, labelcolor=TEXT_SEC,
              fontsize=7, loc="lower right")
    fig.tight_layout(pad=0.8)
    return fig


def chart_ets_vs_avg(match_df):
    """Scatter: raw avg vs ETS for every team. Fixed size — scales naturally via alpha + sizing."""
    n   = len(match_df)
    # Shrink dots and add transparency when many points overlap
    dot_size  = max(25, 80 - n * 1.5)
    dot_alpha = max(0.45, 1.0 - n * 0.015)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    fig.patch.set_facecolor(BG_CARD)
    ax.set_facecolor(BG_CARD)
    ax.tick_params(colors=TEXT_SEC, labelsize=8.5)
    ax.spines["bottom"].set_color(BORDER); ax.spines["left"].set_color(BORDER)
    ax.spines["top"].set_visible(False);   ax.spines["right"].set_visible(False)

    for _, row in match_df.iterrows():
        ax.scatter(row["Team A Avg"], row["ETS A"], color=WINE_GLOW,
                   s=dot_size, alpha=dot_alpha, zorder=4, linewidths=0)
        ax.scatter(row["Team B Avg"], row["ETS B"], color="#5577CC",
                   s=dot_size, alpha=dot_alpha, zorder=4, linewidths=0)
        ax.plot([row["Team A Avg"], row["ETS A"]], [row["ETS A"], row["ETS A"]],
                color=BORDER, linewidth=0.5, alpha=0.5, zorder=3)

    all_vals = pd.concat([match_df[c] for c in ["Team A Avg","Team B Avg","ETS A","ETS B"]])
    lo, hi = all_vals.min() - 0.4, all_vals.max() + 0.4
    ax.plot([lo, hi], [lo, hi], color=GOLD, linewidth=1.2, linestyle="--")
    ax.set_xlabel("Raw Average Skill", color=TEXT_SEC, fontsize=9)
    ax.set_ylabel("Effective Team Skill (ETS)", color=TEXT_SEC, fontsize=9)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    ax.legend(
        handles=[
            mpatches.Patch(color=WINE_GLOW, label="Team A"),
            mpatches.Patch(color="#5577CC",  label="Team B"),
            mpatches.Patch(color=GOLD,       label="ETS = Avg (no penalty)"),
        ],
        frameon=False, labelcolor=TEXT_SEC, fontsize=7.5)
    fig.tight_layout(pad=0.8)
    return fig


def chart_std_comparison(match_df):
    """
    Grouped bar: Team A std vs Team B std.
    Flips to horizontal layout when there are many matches so labels never overlap.
    """
    n = len(match_df)
    labels = [f"M{r['Match']}" for _, r in match_df.iterrows()]
    a_std  = match_df["Team A Std"].tolist()
    b_std  = match_df["Team B Std"].tolist()

    if n <= 15:
        # ── Vertical bars — comfortable up to ~15 matches ──
        fig_w = max(5.5, n * 0.55 + 1.5)
        fig, ax = plt.subplots(figsize=(fig_w, 3.5))
        fig.patch.set_facecolor(BG_CARD)
        ax.set_facecolor(BG_CARD)

        x = np.arange(n)
        w = min(0.35, 0.7 / max(1, n * 0.08 + 0.5))
        ax.bar(x - w/2, a_std, width=w, label="Team A", color=WINE,    edgecolor=BG_DARK, linewidth=0.4)
        ax.bar(x + w/2, b_std, width=w, label="Team B", color="#2F4472", edgecolor=BG_DARK, linewidth=0.4)
        ax.axhline(1.5, color=GOLD, linewidth=1, linestyle="--", alpha=0.7)
        ax.text(n - 0.5, 1.62, "Low-variance target", color=GOLD, fontsize=7, ha="right")
        ax.set_xticks(x)
        lbl_fs = max(6.5, 9 - n * 0.15)
        ax.set_xticklabels(labels, color=TEXT_SEC, fontsize=lbl_fs,
                           rotation=45 if n > 8 else 0, ha="right" if n > 8 else "center")
        ax.set_ylabel("Skill Std Dev", color=TEXT_SEC, fontsize=9)
    else:
        # ── Horizontal bars — cleaner for 16+ matches ──
        row_h  = 0.45
        fig_h  = max(4, n * row_h + 1.2)
        fig, ax = plt.subplots(figsize=(5.5, fig_h))
        fig.patch.set_facecolor(BG_CARD)
        ax.set_facecolor(BG_CARD)

        y  = np.arange(n)
        bh = min(0.3, max(0.15, 0.3 - n * 0.003))
        ax.barh(y + bh/2, a_std, height=bh, label="Team A", color=WINE,    edgecolor=BG_DARK, linewidth=0.4)
        ax.barh(y - bh/2, b_std, height=bh, label="Team B", color="#2F4472", edgecolor=BG_DARK, linewidth=0.4)
        ax.axvline(1.5, color=GOLD, linewidth=1, linestyle="--", alpha=0.7)
        ax.text(1.55, n - 1, "Low-variance target", color=GOLD, fontsize=6.5, va="top")
        ax.set_yticks(y)
        lbl_fs = max(6, 8.5 - n * 0.04)
        ax.set_yticklabels(labels, color=TEXT_SEC, fontsize=lbl_fs)
        ax.set_xlabel("Skill Std Dev", color=TEXT_SEC, fontsize=9)

    ax.tick_params(colors=TEXT_SEC, labelsize=8)
    ax.spines["top"].set_visible(False);   ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(BORDER); ax.spines["left"].set_color(BORDER)
    ax.legend(frameon=False, labelcolor=TEXT_SEC, fontsize=7.5)
    fig.tight_layout(pad=0.8)
    return fig


# =============================================================================
#  PAGE CONFIG & GLOBAL CSS
# =============================================================================

st.set_page_config(page_title="6v6 Matchmaking", page_icon="⚔️", layout="wide")

st.markdown(f"""
<style>
  /* ── Global background ── */
  .stApp {{ background-color: {BG_DARK}; }}
  section[data-testid="stSidebar"] {{ background-color: {BG_MID} !important; }}
  section[data-testid="stSidebar"] * {{ color: {TEXT_SEC}; }}

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header {{ visibility: hidden; }}

  /* ── Typography ── */
  html, body, [class*="css"] {{ color: {TEXT_PRI}; }}
  h1,h2,h3,h4 {{ color: {TEXT_PRI} !important; }}

  /* ── Title banner ── */
  .title-banner {{
    background: linear-gradient(135deg, {WINE_DARK} 0%, {BG_MID} 60%, {BG_DARK} 100%);
    border: 1px solid {WINE};
    border-left: 5px solid {WINE};
    border-radius: 10px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.5rem;
  }}
  .title-banner h1 {{ color: {CREAM} !important; margin: 0; font-size: 2rem; letter-spacing: 0.02em; }}
  .title-banner p  {{ color: {TEXT_SEC}; margin: 0.4rem 0 0; font-size: 0.9rem; }}

  /* ── Metric card ── */
  .metric-card {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-top: 3px solid {WINE};
    border-radius: 8px;
    padding: 1rem 1.2rem;
    text-align: center;
    min-height: 80px;
  }}
  .metric-card .val {{ color: {CREAM}; font-size: 1.9rem; font-weight: 700; margin: 0; line-height: 1.1; }}
  .metric-card .lbl {{ color: {TEXT_SEC}; font-size: 0.78rem; margin: 0.2rem 0 0; }}

  /* ── Formula box ── */
  .formula-box {{
    background: {BG_MID};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
  }}
  .formula-box .eq {{
    font-family: "Courier New", monospace;
    font-size: 0.95rem;
    color: {CREAM};
    background: {BG_DARK};
    border-left: 3px solid {WINE};
    padding: 0.5rem 0.9rem;
    border-radius: 4px;
    display: block;
    margin: 0.5rem 0;
  }}
  .formula-box .label {{ color: {WINE_GLOW}; font-size: 0.78rem; font-weight: 700;
                         letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.2rem; }}
  .formula-box p {{ color: {TEXT_SEC}; font-size: 0.84rem; margin: 0.3rem 0 0; line-height: 1.5; }}

  /* ── Info callout ── */
  .callout {{
    background: {BG_PANEL};
    border-left: 4px solid {WINE};
    border-radius: 0 6px 6px 0;
    padding: 0.7rem 1rem;
    margin: 0.5rem 0;
    color: {TEXT_SEC};
    font-size: 0.85rem;
    line-height: 1.5;
  }}
  .callout b {{ color: {CREAM}; }}

  /* ── Property tags ── */
  .prop-row {{ display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 0.4rem 0; }}
  .prop-tag {{
    background: {BG_PANEL}; border: 1px solid {BORDER};
    color: {TEXT_SEC}; font-size: 0.76rem; padding: 3px 10px;
    border-radius: 20px; white-space: nowrap;
  }}
  .prop-tag.good {{ border-color: {WINE}; color: {WINE_GLOW}; }}

  /* ── Team row ── */
  .team-row {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.3rem 0.75rem; margin: 2px 0; border-radius: 4px; font-size: 0.84rem;
  }}
  .team-row-a {{ background: {BG_PANEL}; border-left: 3px solid {WINE}; }}
  .team-row-b {{ background: #121c2e; border-left: 3px solid #2F4472; }}
  .team-row .pname {{ color: {TEXT_PRI}; }}
  .team-row .pskill {{ color: {CREAM}; font-weight: 700; }}
  .skill-track {{ display:inline-block; width:80px; height:6px; background:{BORDER};
                  border-radius:3px; vertical-align:middle; margin: 0 6px; }}
  .skill-fill-a {{ display:inline-block; height:6px; background:{WINE}; border-radius:3px; }}
  .skill-fill-b {{ display:inline-block; height:6px; background:#2F4472; border-radius:3px; }}

  /* ── ETS badge ── */
  .ets-badge {{
    display: inline-block; font-size: 0.74rem; font-weight: 700;
    padding: 2px 8px; border-radius: 20px;
    background: {WINE_DARK}; color: {WINE_GLOW};
    border: 1px solid {WINE}; vertical-align: middle;
  }}
  .ets-badge-b {{
    background: #0d1a30; color: #7090CC; border-color: #2F4472;
  }}

  /* ── Tab styling ── */
  button[data-baseweb="tab"] {{ color: {TEXT_SEC} !important; }}
  button[data-baseweb="tab"][aria-selected="true"] {{
    color: {CREAM} !important;
    border-bottom-color: {WINE} !important;
  }}

  /* ── Divider ── */
  hr {{ border-color: {BORDER} !important; }}
</style>

<div class="title-banner">
  <h1>⚔️ 6v6 Skill-Based Matchmaking</h1>
  <p>Variance-adjusted win formula &nbsp;·&nbsp; Carry penalty &nbsp;·&nbsp; Optimal team composition &nbsp;·&nbsp; Skill frequency input</p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
#  SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown(f"<div style='color:{WINE_GLOW};font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.8rem'>How The Formula Works</div>", unsafe_allow_html=True)

    st.markdown(f"""
<div class="formula-box">
  <div class="label">Stage 1 — Effective Team Skill</div>
  <span class="eq">ETS = mean(skills) − 0.35 × std(skills)</span>
  <p>
    Every team gets a score that reflects not just <i>how skilled</i> they are on average,
    but <i>how evenly distributed</i> that skill is.<br><br>
    <b>std(skills)</b> measures how spread out the player skills are.
    A team of six skill-5 players has std = 0 — no penalty.
    A carry team of [10, 2, 2, 2, 2, 2] has std = 2.98 — a meaningful deduction.<br><br>
    The coefficient <b>0.35</b> sets how harshly imbalance is punished.
  </p>
</div>

<div class="formula-box">
  <div class="label">Stage 2 — Win Probability</div>
  <span class="eq">P(A) = 1 / (1 + 10^(−(ETS_A − ETS_B) / 12))</span>
  <p>
    Maps the ETS gap between teams onto a probability between 0% and 100%
    using an S-shaped curve. When ETS_A = ETS_B, the result is exactly 50%.<br><br>
    The divisor <b>12</b> controls the curve's steepness and was chosen so that
    the absolute worst-case matchup — a team of all-10s vs a team of all-1s — 
    produces 84.9%, keeping every outcome below the 85% ceiling.
  </p>
</div>

<div class="callout">
  <b>Why this matters:</b> A team with one great player and five weak ones
  has the same raw average as a team of all-average players — but the five weak 
  players each lose their individual matchups simultaneously. The carry cannot 
  be everywhere at once. This formula correctly penalises that imbalance.
</div>
""", unsafe_allow_html=True)

    st.markdown(f"<div style='color:{WINE_GLOW};font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;margin: 1rem 0 0.5rem'>Formula Properties</div>", unsafe_allow_html=True)

    props = [
        ("50% when ETS_A = ETS_B", True),
        ("Symmetric: P(A)+P(B) = 100%", True),
        ("Carry teams penalised fairly", True),
        ("Hard cap: max 84.9%", True),
        ("Uniform teams: zero penalty", True),
    ]
    tags = "".join(
        f'<span class="prop-tag{"  good" if g else ""}">{p}</span>' for p, g in props
    )
    st.markdown(f'<div class="prop-row">{tags}</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:{WINE_GLOW};font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem'>Matchmaking Method</div>", unsafe_allow_html=True)
    st.markdown(f"""<div style='color:{TEXT_SEC};font-size:0.83rem;line-height:1.6'>
All <b style='color:{CREAM}'>924 possible 6v6 splits</b> are evaluated for each group of 12 players.
The split whose win probability is closest to 50% — under the variance-adjusted formula — is selected.
Players are grouped by skill before splitting, so you only compete against similarly-rated opponents.
</div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.download_button(
        "⬇ Download Sample Excel",
        data=create_sample_excel(),
        file_name="sample_skill_frequency.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    st.markdown(f"<div style='color:{TEXT_SEC};font-size:0.75rem;margin-top:0.3rem'>Needs <b>Skill</b> (1–10) and <b>Frequency</b> columns. Min 12 players per match.</div>", unsafe_allow_html=True)


# =============================================================================
#  INPUT TABS
# =============================================================================

tab_upload, tab_generate, tab_formula = st.tabs([
    "  📂  Upload Excel  ",
    "  🎲  Generate Players  ",
    "  📐  Formula Explorer  ",
])

players_df = None

# ── Upload ─────────────────────────────────────────────────────────────────
with tab_upload:
    st.markdown(f"<div style='color:{TEXT_SEC};font-size:0.88rem;margin-bottom:0.8rem'>Upload an Excel file with <b style='color:{CREAM}'>Skill</b> and <b style='color:{CREAM}'>Frequency</b> columns. Each row is a skill tier and the number of players at that level.</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Choose .xlsx file", type=["xlsx", "xls"], label_visibility="collapsed")
    if uploaded:
        try:
            players_df = read_excel_frequency(uploaded)
            fp         = frequency_table(players_df)
            total      = len(players_df)
            st.success(f"Loaded **{total} players** — **{total // 12}** match(es) possible, {total % 12} benched.")
            st.dataframe(fp, use_container_width=True, hide_index=True, height=260)
        except Exception as e:
            st.error(f"Error: {e}")

# ── Generate ────────────────────────────────────────────────────────────────
with tab_generate:
    gc1, gc2 = st.columns([3, 1])
    with gc1:
        n_players = st.slider("Number of players", 12, 300, 48, step=12,
                              help="Multiples of 12 → zero benched players.")
    with gc2:
        seed = st.number_input("Seed", value=42, step=1)

    if st.button("Generate Players", use_container_width=True, type="primary"):
        random.seed(int(seed))
        skills = [generate_skill_level() for _ in range(n_players)]
        gen_df = pd.DataFrame({
            "Player": [f"Player_{i+1:03d}" for i in range(n_players)],
            "Skill":  skills,
        })
        st.session_state["generated_df"] = gen_df
        freq_gen = frequency_table(gen_df)
        col_tbl, col_chart = st.columns([1, 1])
        with col_tbl:
            st.dataframe(freq_gen, use_container_width=True, hide_index=True)
        with col_chart:
            st.pyplot(chart_skill_distribution(gen_df), use_container_width=True)

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            freq_gen.to_excel(w, index=False, sheet_name="Players")
        st.download_button("⬇ Download Skill/Frequency Excel", data=buf.getvalue(),
                           file_name="generated_players.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if "generated_df" in st.session_state and players_df is None:
        players_df = st.session_state["generated_df"]

# ── Formula Explorer ─────────────────────────────────────────────────────────
with tab_formula:
    st.markdown(f"<div style='color:{TEXT_SEC};font-size:0.88rem;margin-bottom:1rem'>Visualisations of how the two formula stages behave. Adjust teams manually to see the carry penalty in action.</div>", unsafe_allow_html=True)

    fe1, fe2 = st.columns(2)
    with fe1:
        st.markdown("**Stage 1 — Variance penalty curve**")
        st.caption("Shows how ETS drops below raw mean as a team's skill spread increases.")
        st.pyplot(chart_ets_curve(), use_container_width=True)
    with fe2:
        st.markdown("**Stage 2 — Win probability S-curve**")
        st.caption("Maps any ETS difference to a win probability. Always 50% when gap is zero.")
        st.pyplot(chart_win_curve(), use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**Interactive Team Builder — see the carry penalty live**")
    fi1, fi2 = st.columns(2)
    with fi1:
        st.markdown(f"<div style='color:{WINE_GLOW};font-size:0.8rem;font-weight:700;margin-bottom:0.4rem'>TEAM A</div>", unsafe_allow_html=True)
        ta_vals = [st.slider(f"Player A{i+1}", 1, 10, [10,2,2,2,2,2][i], key=f"ta{i}") for i in range(6)]
    with fi2:
        st.markdown(f"<div style='color:#5577CC;font-size:0.8rem;font-weight:700;margin-bottom:0.4rem'>TEAM B</div>", unsafe_allow_html=True)
        tb_vals = [st.slider(f"Player B{i+1}", 1, 10, [3,3,4,4,3,3][i], key=f"tb{i}") for i in range(6)]

    ets_a_live = effective_team_skill(ta_vals)
    ets_b_live = effective_team_skill(tb_vals)
    wc_live    = win_chance_teams(ta_vals, tb_vals)
    bs_live    = balance_score_from_wc(wc_live)

    fm1, fm2, fm3, fm4, fm5 = st.columns(5)
    for col, val, lbl in [
        (fm1, f"{np.mean(ta_vals):.2f}", "Team A Raw Avg"),
        (fm2, f"{np.mean(tb_vals):.2f}", "Team B Raw Avg"),
        (fm3, f"{ets_a_live:.2f} vs {ets_b_live:.2f}", "ETS A vs B"),
        (fm4, f"{wc_live*100:.1f}%", "Team A Win %"),
        (fm5, f"{bs_live*100:.1f}%", "Balance Score"),
    ]:
        col.markdown(f'<div class="metric-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown(f"""<div class="callout">
<b>Team A</b>: mean = {np.mean(ta_vals):.2f}, std = {np.std(ta_vals):.2f} → penalty = {ALPHA * np.std(ta_vals):.2f} → ETS = <b>{ets_a_live:.2f}</b><br>
<b>Team B</b>: mean = {np.mean(tb_vals):.2f}, std = {np.std(tb_vals):.2f} → penalty = {ALPHA * np.std(tb_vals):.2f} → ETS = <b>{ets_b_live:.2f}</b>
</div>""", unsafe_allow_html=True)


# =============================================================================
#  MAIN DASHBOARD — only shown when players are loaded
# =============================================================================

if players_df is not None:
    st.markdown("<hr>", unsafe_allow_html=True)

    avg_s = players_df["Skill"].mean()
    min_s = int(players_df["Skill"].min())
    max_s = int(players_df["Skill"].max())
    n_pos = len(players_df) // 12

    d1, d2, d3, d4, d5 = st.columns(5)
    for col, val, lbl in [
        (d1, len(players_df),               "Total Players"),
        (d2, f"{avg_s:.1f}",                "Average Skill"),
        (d3, f"{min_s} – {max_s}",          "Skill Range"),
        (d4, n_pos,                          "Matches Possible"),
        (d5, len(players_df) % 12,           "Will Be Benched"),
    ]:
        col.markdown(f'<div class="metric-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("")
    ov1, ov2 = st.columns([1, 1])
    with ov1:
        st.markdown("**Skill Frequency Table**")
        fd = frequency_table(players_df).copy()
        fd["ETS (uniform team)"] = fd["Skill"].apply(
            lambda s: f"{effective_team_skill([int(s)]*6):.2f}"
        )
        fd["Win % vs all-5s"]    = fd["Skill"].apply(
            lambda s: f"{win_chance_teams([int(s)]*6, [5]*6)*100:.1f}%"
        )
        st.dataframe(fd, use_container_width=True, height=320, hide_index=True)
    with ov2:
        st.markdown("**Skill Distribution**")
        st.pyplot(chart_skill_distribution(players_df), use_container_width=True)

    # ── Matchmaking controls ────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:{CREAM};font-size:1.1rem;font-weight:700;margin-bottom:0.5rem'>Run 6v6 Matchmaking</div>", unsafe_allow_html=True)

    ac1, ac2 = st.columns([3, 1])
    with ac1:
        alpha_mm = st.slider(
            "Variance penalty (α)",
            0.0, 0.80, ALPHA, step=0.05,
            help="0 = only raw average matters. Higher = harsher carry penalty. 0.35 is the calibrated default.",
        )
    with ac2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        run_btn = st.button("⚔ Generate Matches", use_container_width=True, type="primary")

    if len(players_df) < 12:
        st.warning("Need at least 12 players to form one 6v6 match.")
    elif run_btn:
        for k in ["matches", "rosters", "benched", "match_src"]:
            st.session_state.pop(k, None)
        with st.spinner("Evaluating all 924 splits per match group…"):
            matches, rosters, benched = matchmake_6v6(players_df, alpha=alpha_mm)
        st.session_state.update({
            "matches": matches, "rosters": rosters,
            "benched": benched, "match_src": players_df.copy()
        })

    # ── Results ─────────────────────────────────────────────────────────────
    if "matches" in st.session_state:
        matches   = st.session_state["matches"]
        rosters   = st.session_state["rosters"]
        benched   = st.session_state["benched"]
        src_df    = st.session_state.get("match_src", players_df)
        match_df  = pd.DataFrame(matches)
        roster_df = pd.DataFrame(rosters)

        if not matches:
            st.warning("No complete 6v6 matches could be formed.")
        else:
            avg_bs  = match_df["Balance Score"].mean()
            perfect = int((match_df["Balance Score"] >= 99.0).sum())

            r1, r2, r3, r4 = st.columns(4)
            for col, val, lbl in [
                (r1, len(matches),          "Matches Created"),
                (r2, f"{avg_bs:.1f}%",      "Avg Balance Score"),
                (r3, perfect,               "Near-Perfect (≥99%)"),
                (r4, len(benched),          "Benched Players"),
            ]:
                col.markdown(f'<div class="metric-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

            st.markdown("")

            # ── Overview charts ─────────────────────────────────────────────
            st.markdown(f"<div style='color:{CREAM};font-size:1rem;font-weight:700;margin:0.8rem 0 0.4rem'>Match Overview</div>", unsafe_allow_html=True)

            oc1, oc2, oc3 = st.columns(3)
            with oc1:
                st.markdown("**Balance Scores by Match**")
                st.caption("How close each match is to a 50/50 outcome.")
                st.pyplot(chart_balance_overview(match_df), use_container_width=True)
            with oc2:
                st.markdown("**ETS vs Raw Average**")
                st.caption("Points below the diagonal line have been penalised for uneven skill spread.")
                st.pyplot(chart_ets_vs_avg(match_df), use_container_width=True)
            with oc3:
                st.markdown("**Skill Spread (Std Dev) per Team**")
                st.caption("Lower std = more uniform team. The dashed line is a low-variance target.")
                st.pyplot(chart_std_comparison(match_df), use_container_width=True)

            # ── Summary table ───────────────────────────────────────────────
            st.markdown(f"<div style='color:{CREAM};font-size:1rem;font-weight:700;margin:1rem 0 0.3rem'>Match Summary Table</div>", unsafe_allow_html=True)
            st.caption("ETS = Effective Team Skill. Lower std → ETS closer to raw avg. Higher std → ETS is discounted.")

            def _cb(v):
                if v >= 95:   return f"background-color:#1a3a1a;color:#6aab6a"
                elif v >= 80: return f"background-color:#1e2e1e;color:#9ab59a"
                elif v >= 60: return f"background-color:#2e2410;color:{GOLD}"
                else:         return f"background-color:{WINE_DARK};color:{WINE_GLOW}"

            def _cw(v):
                if 45 <= v <= 55:   return f"color:{CREAM};font-weight:700"
                elif 35 <= v <= 65: return f"color:{GOLD}"
                else:               return f"color:{WINE_GLOW}"

            def _cs(v):
                if v <= 1.0:   return f"color:#6aab6a"
                elif v <= 2.0: return f"color:{GOLD}"
                else:          return f"color:{WINE_GLOW}"

            style = match_df.style
            if "Balance Score"  in match_df.columns: style = style.applymap(_cb, subset=["Balance Score"])
            if "Win % (A)"      in match_df.columns: style = style.applymap(_cw, subset=["Win % (A)", "Win % (B)"])
            if "Team A Std"     in match_df.columns: style = style.applymap(_cs, subset=["Team A Std", "Team B Std"])
            st.dataframe(style, use_container_width=True, height=min(420, len(matches)*42+60))

            # ── Per-match rosters ────────────────────────────────────────────
            st.markdown(f"<div style='color:{CREAM};font-size:1rem;font-weight:700;margin:1rem 0 0.3rem'>Team Rosters</div>", unsafe_allow_html=True)

            for m_row in matches:
                mn    = m_row["Match"]
                bs    = m_row["Balance Score"]
                badge = "🟢" if bs >= 95 else "🟡" if bs >= 80 else "🔴"

                with st.expander(
                    f"{badge}  Match {mn}  ·  "
                    f"ETS  A={m_row['ETS A']:.2f}  B={m_row['ETS B']:.2f}  ·  "
                    f"Win%  A={m_row['Win % (A)']:.1f}%  ·  "
                    f"Balance {bs:.1f}%"
                ):
                    # team comparison chart
                    st.pyplot(chart_team_comparison(m_row, roster_df), use_container_width=True)

                    ta_rows = roster_df[(roster_df["Match"] == mn) & (roster_df["Team"] == "A")].sort_values("Skill", ascending=False)
                    tb_rows = roster_df[(roster_df["Match"] == mn) & (roster_df["Team"] == "B")].sort_values("Skill", ascending=False)

                    rc1, rc2 = st.columns(2)
                    with rc1:
                        st.markdown(
                            f"<div style='margin-bottom:0.4rem'>"
                            f"<b style='color:{WINE_GLOW}'>Team A</b>&nbsp;&nbsp;"
                            f"avg <b>{m_row['Team A Avg']:.2f}</b>&nbsp; "
                            f"std <b>{m_row['Team A Std']:.2f}</b>&nbsp; "
                            f'<span class="ets-badge">ETS {m_row["ETS A"]:.2f}</span>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        for _, r in ta_rows.iterrows():
                            w = int(r["Skill"] * 8)
                            st.markdown(
                                f'<div class="team-row team-row-a">'
                                f'<span class="pname">{r["Player"]}</span>'
                                f'<span>'
                                f'<span class="skill-track"><span class="skill-fill-a" style="width:{w}px"></span></span>'
                                f'<span class="pskill">{int(r["Skill"])}</span>'
                                f'</span></div>',
                                unsafe_allow_html=True,
                            )
                    with rc2:
                        st.markdown(
                            f"<div style='margin-bottom:0.4rem'>"
                            f"<b style='color:#5577CC'>Team B</b>&nbsp;&nbsp;"
                            f"avg <b>{m_row['Team B Avg']:.2f}</b>&nbsp; "
                            f"std <b>{m_row['Team B Std']:.2f}</b>&nbsp; "
                            f'<span class="ets-badge ets-badge-b">ETS {m_row["ETS B"]:.2f}</span>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        for _, r in tb_rows.iterrows():
                            w = int(r["Skill"] * 8)
                            st.markdown(
                                f'<div class="team-row team-row-b">'
                                f'<span class="pname">{r["Player"]}</span>'
                                f'<span>'
                                f'<span class="skill-track"><span class="skill-fill-b" style="width:{w}px"></span></span>'
                                f'<span class="pskill">{int(r["Skill"])}</span>'
                                f'</span></div>',
                                unsafe_allow_html=True,
                            )

            if benched:
                st.warning(f"**{len(benched)} player(s) benched** (no complete group of 12): {', '.join(benched)}")

            # ── Export ───────────────────────────────────────────────────────
            st.markdown("<hr>", unsafe_allow_html=True)
            excel_out = results_to_excel(matches, rosters, benched, src_df)
            st.download_button(
                "⬇ Download Full Results (.xlsx)",
                data=excel_out,
                file_name="6v6_match_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
            st.caption("Sheets: Match Summaries · Team Rosters · Skill Frequency · Benched (if any)")

else:
    st.markdown(
        f'<div class="callout" style="margin-top:1rem;text-align:center;padding:2rem">'
        f'Upload a Skill/Frequency Excel file or generate random players to get started.'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    f'<div style="color:{TEXT_SEC};font-size:0.75rem;text-align:center;margin-top:2rem;padding-top:1rem;border-top:1px solid {BORDER}">'
    f'ETS = mean(skills) − 0.35 × std(skills) &nbsp;·&nbsp; '
    f'P(A) = 1 / (1 + 10^(−(ETS_A − ETS_B) / 12)) &nbsp;·&nbsp; '
    f'Max win% = 84.9% &nbsp;·&nbsp; Team splits: C(12,6) = 924 combinations evaluated per match'
    f'</div>',
    unsafe_allow_html=True,
)
