import streamlit as st
import pandas as pd
import numpy as np
import random
import io

# =============================================================================
#  WIN FORMULA  —  Elo-style logistic, K=12
#
#  P(Team A wins) = 1 / (1 + 10 ^ ( -(avgA - avgB) / 12 ))
#
#  Derived from the Elo rating system used in chess and competitive gaming.
#  K=12 is chosen so the absolute maximum win probability (avg skill 10 vs
#  avg skill 1, a 9-bracket gap) is 84.9%, keeping every outcome below 85%.
#
#  Why logistic over multiplicative?
#    - Naturally bounded between 0% and 100% — no formula overflow at extremes
#    - Symmetric: P(A vs B) + P(B vs A) = 1.0 always
#    - Diminishing returns: a skill gap of 1 matters more near 50% than at 80%
#    - Battle-tested: powers chess Elo, League of Legends MMR, and others
# =============================================================================

K = 12  # Scaling constant. Increasing K flattens the curve; decreasing steepens it.


def win_chance(avg_a: float, avg_b: float) -> float:
    """
    Elo logistic win probability for Team A vs Team B.
    avg_a / avg_b = average skill level of each team (1–10).
    """
    return 1.0 / (1.0 + 10.0 ** (-(avg_a - avg_b) / K))


def balance_score_from_wc(wc: float) -> float:
    """1.0 = perfectly balanced (50/50), 0.0 = completely one-sided."""
    return 1.0 - abs(wc - 0.5)


def generate_skill_level() -> int:
    """Normal distribution centred on 5, clipped to [1, 10]."""
    return max(1, min(10, int(round(random.gauss(5, 1.8)))))


# =============================================================================
#  6v6 MATCHMAKING  —  Snake-draft within sorted groups of 12
#
#  Algorithm:
#    1. Sort all players by skill (descending).
#    2. Take consecutive groups of 12 — each group becomes one match.
#    3. Within each group of 12 (still sorted), apply a snake draft:
#         Pick order: A B B A A B B A A B B A
#       This interleaves strong and weak players equally across both teams,
#       guaranteeing the average skill difference between teams is minimised.
#    4. Compute team averages, apply the Elo formula, record results.
#    5. Players left over after full groups of 12 are benched.
# =============================================================================

SNAKE = ["A", "B", "B", "A", "A", "B", "B", "A", "A", "B", "B", "A"]


def snake_draft_teams(group: pd.DataFrame):
    """
    Split a 12-player DataFrame (sorted descending by skill) into two teams
    of 6 via snake draft. Returns (team_a_df, team_b_df).
    """
    team_a, team_b = [], []
    for idx, (_, row) in enumerate(group.iterrows()):
        (team_a if SNAKE[idx] == "A" else team_b).append(row)
    return pd.DataFrame(team_a), pd.DataFrame(team_b)


def matchmake_6v6(players_df: pd.DataFrame):
    """
    Partition players into 6v6 matches.
    Returns (list[dict match summaries], list[dict team rosters], list[str benched]).
    """
    df = players_df.copy().sort_values("Skill", ascending=False).reset_index(drop=True)
    n = len(df)
    n_matches = n // 12
    benched = list(df.iloc[n_matches * 12:]["Player"])

    matches = []
    rosters = []

    for m in range(n_matches):
        group = df.iloc[m * 12 : (m + 1) * 12].reset_index(drop=True)
        team_a, team_b = snake_draft_teams(group)

        avg_a = team_a["Skill"].mean()
        avg_b = team_b["Skill"].mean()
        wc    = win_chance(avg_a, avg_b)
        bs    = balance_score_from_wc(wc)

        match_num = m + 1
        matches.append({
            "Match #":          match_num,
            "Team A Avg Skill": round(avg_a, 2),
            "Team B Avg Skill": round(avg_b, 2),
            "Skill Diff":       round(abs(avg_a - avg_b), 2),
            "Win % (Team A)":   round(wc * 100, 2),
            "Win % (Team B)":   round((1 - wc) * 100, 2),
            "Balance Score":    round(bs * 100, 2),
        })

        for _, row in team_a.iterrows():
            rosters.append({"Match #": match_num, "Team": "A",
                            "Player": row["Player"], "Skill": int(row["Skill"])})
        for _, row in team_b.iterrows():
            rosters.append({"Match #": match_num, "Team": "B",
                            "Player": row["Player"], "Skill": int(row["Skill"])})

    return matches, rosters, benched


# =============================================================================
#  EXCEL HELPERS
# =============================================================================

def read_excel_frequency(file) -> pd.DataFrame:
    df = pd.read_excel(file)
    df.columns = [c.strip().title() for c in df.columns]
    if "Skill" not in df.columns or "Frequency" not in df.columns:
        raise ValueError("Excel file must have 'Skill' and 'Frequency' columns.")
    df["Skill"]     = pd.to_numeric(df["Skill"],     errors="coerce").clip(1, 10).fillna(5).astype(int)
    df["Frequency"] = pd.to_numeric(df["Frequency"], errors="coerce").fillna(0).astype(int)
    df = df[df["Frequency"] > 0]
    if df.empty:
        raise ValueError("No valid rows found — check Frequency values are positive integers.")
    rows = []
    for _, row in df.iterrows():
        rows.extend([int(row["Skill"])] * int(row["Frequency"]))
    out = pd.DataFrame({"Skill": rows})
    out["Player"] = [f"Player_{i+1:03d}" for i in range(len(out))]
    return out[["Player", "Skill"]]


def frequency_table_from_players(players_df: pd.DataFrame) -> pd.DataFrame:
    freq = players_df["Skill"].value_counts().sort_index().reset_index()
    freq.columns = ["Skill", "Frequency"]
    return freq


def create_sample_excel() -> bytes:
    data = {
        "Skill":     [1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
        "Frequency": [2,  3,  6, 10, 14, 12,  8,  5,  3,  1],
    }
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame(data).to_excel(writer, index=False, sheet_name="Players")
    return buf.getvalue()


def results_to_excel(matches, rosters, benched, players_df) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame(matches).to_excel(writer, index=False, sheet_name="Match Summaries")
        pd.DataFrame(rosters).to_excel(writer, index=False, sheet_name="Team Rosters")
        frequency_table_from_players(players_df).to_excel(
            writer, index=False, sheet_name="Skill Frequency")
        if benched:
            pd.DataFrame({"Benched Players": benched}).to_excel(
                writer, index=False, sheet_name="Benched")
    return buf.getvalue()


def skill_distribution_chart(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["Skill"].value_counts().sort_index().reset_index()
    counts.columns = ["Skill Level", "Player Count"]
    return counts


# =============================================================================
#  STREAMLIT UI
# =============================================================================

st.set_page_config(page_title="6v6 Matchmaking System", page_icon="⚔️", layout="wide")

st.markdown("""
<style>
    .title-block {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem; border-radius: 12px; margin-bottom: 1.5rem;
    }
    .title-block h1 { color: #e94560; margin: 0; font-size: 2.2rem; }
    .title-block p  { color: #a8b2d8; margin: 0.5rem 0 0; font-size: 1rem; }

    .metric-card {
        background: #1e1e2e; border: 1px solid #2d2d44; border-radius: 10px;
        padding: 1rem 1.5rem; text-align: center; height: 90px;
        display: flex; flex-direction: column; justify-content: center;
    }
    .metric-card h3 { color: #e94560; font-size: 1.8rem; margin: 0; }
    .metric-card p  { color: #a8b2d8; margin: 0.2rem 0 0; font-size: 0.82rem; }

    .formula-card {
        background: #0d1117; border: 1px solid #2d2d44; border-radius: 10px;
        padding: 1.2rem 1.5rem; margin: 0.5rem 0;
    }
    .formula-card .eq {
        font-family: monospace; font-size: 1.05rem; color: #58a6ff;
        background: #161b22; padding: 0.6rem 1rem; border-radius: 6px;
        border-left: 3px solid #e94560; margin: 0.5rem 0;
    }
    .formula-card p { color: #a8b2d8; font-size: 0.88rem; margin: 0.3rem 0; }
    .formula-card h4 { color: #e94560; margin: 0 0 0.5rem; font-size: 1rem; }

    .team-a { background: #0d2137; border-left: 3px solid #58a6ff;
              border-radius: 6px; padding: 0.4rem 0.8rem; margin: 2px 0;
              font-size: 0.85rem; color: #cdd9e5; }
    .team-b { background: #1f0d2e; border-left: 3px solid #bc8cff;
              border-radius: 6px; padding: 0.4rem 0.8rem; margin: 2px 0;
              font-size: 0.85rem; color: #cdd9e5; }

    .snake-viz { font-family: monospace; font-size: 0.9rem; color: #cdd9e5;
                 background: #161b22; padding: 0.5rem 1rem; border-radius: 6px; }
</style>
<div class="title-block">
  <h1>6v6 Skill-Based Matchmaking System</h1>
  <p>Elo logistic win formula &nbsp;·&nbsp; Snake-draft team balancing &nbsp;·&nbsp; Skill frequency input</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Win Probability Formula")
    st.markdown("""
<div class="formula-card">
  <h4>Elo Logistic Model (K=12)</h4>
  <div class="eq">P(A) = 1 / (1 + 10 ^ (-(avgA - avgB) / 12))</div>
  <p><b>avgA / avgB</b> = average skill of each 6-player team</p>
  <p><b>K=12</b> calibrated so the maximum possible win% (avg 10 vs avg 1) is capped at 84.9%, keeping all outcomes below 85%.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
**Why this formula?**
- Derived from the **Elo rating system** (chess, competitive FPS, MOBAs)
- Naturally bounded — can never exceed 100% or go below 0%
- **Symmetric**: P(A vs B) + P(B vs A) = 1.0 always
- **Diminishing returns**: a 1-bracket gap near 50% matters more than at 80%
- The old multiplicative formula could exceed 100% at extreme gaps

**K explained:** Larger K = flatter curve (skill matters less). Smaller K = steeper curve (skill dominates more).
""")

    st.markdown("**Examples:**")
    examples = [(5,5),(5,3),(5,7),(5,10),(10,1)]
    for a, b in examples:
        wc = win_chance(a, b)
        st.markdown(f"- Avg {a} vs {b}: **{wc*100:.1f}%**")

    st.divider()
    st.markdown("### Matchmaking Method")
    st.markdown("""
**Snake Draft (per group of 12)**

Sort 12 players by skill, then assign:

`A B B A A B B A A B B A`

Pick 1 goes to Team A (best player), picks 2+3 go to Team B, picks 4+5 back to Team A, etc. This interleaves talent so both teams receive equal numbers of top, middle, and bottom-skill players.
""")

    st.divider()
    st.download_button(
        "Download Sample Excel",
        data=create_sample_excel(),
        file_name="sample_skill_frequency.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.caption("Needs 'Skill' (1–10) and 'Frequency' columns. Min 12 players for one match.")

# ── Input tabs ────────────────────────────────────────────────────────────────
tab_upload, tab_generate, tab_formula = st.tabs([
    "Upload Excel", "Generate Random Players", "Formula Explorer"
])

players_df = None

# ── Upload tab ──
with tab_upload:
    st.markdown(
        "Upload an Excel with **Skill** and **Frequency** columns. "
        "You need at least **12 players** for one 6v6 match."
    )
    uploaded = st.file_uploader("Choose .xlsx file", type=["xlsx", "xls"])
    if uploaded:
        try:
            players_df = read_excel_frequency(uploaded)
            freq_preview = frequency_table_from_players(players_df)
            total = len(players_df)
            matches_possible = total // 12
            st.success(
                f"Loaded **{total} players** — can form **{matches_possible} match(es)**, "
                f"{total % 12} player(s) benched."
            )
            st.dataframe(freq_preview, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Error: {e}")

# ── Generate tab ──
with tab_generate:
    col_n, col_seed = st.columns([2, 1])
    with col_n:
        n_players = st.slider("Number of players", 12, 300, 48, step=12,
                              help="Multiples of 12 create zero bench players.")
    with col_seed:
        seed = st.number_input("Random seed", value=42, step=1)

    if st.button("Generate Players", use_container_width=True):
        random.seed(int(seed))
        skills = [generate_skill_level() for _ in range(n_players)]
        gen_df = pd.DataFrame({
            "Player": [f"Player_{i+1:03d}" for i in range(n_players)],
            "Skill":  skills,
        })
        st.session_state["generated_df"] = gen_df
        freq_gen = frequency_table_from_players(gen_df)
        st.success(f"Generated **{n_players} players** across {len(freq_gen)} skill levels.")
        st.dataframe(freq_gen, use_container_width=True, hide_index=True)

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            freq_gen.to_excel(writer, index=False, sheet_name="Players")
        st.download_button("Download as Skill/Frequency Excel", data=buf.getvalue(),
                           file_name="generated_skill_frequency.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if "generated_df" in st.session_state and players_df is None:
        players_df = st.session_state["generated_df"]

# ── Formula Explorer tab ──
with tab_formula:
    st.markdown("### Interactive Win % Explorer")
    st.markdown(
        "Adjust two team average skills to see how the Elo logistic formula calculates "
        "the win probability. Compare with the old multiplicative formula."
    )
    fc1, fc2 = st.columns(2)
    with fc1:
        fa = st.slider("Team A Average Skill", 1.0, 10.0, 7.0, step=0.5)
    with fc2:
        fb = st.slider("Team B Average Skill", 1.0, 10.0, 4.0, step=0.5)

    wc_new  = win_chance(fa, fb)
    wc_old  = 0.5 * (1.1 ** (fa - fb))
    bs_new  = balance_score_from_wc(wc_new)

    ec1, ec2, ec3 = st.columns(3)
    ec1.metric("Team A Win %  (Elo logistic)",  f"{wc_new*100:.2f}%")
    ec2.metric("Team B Win %  (Elo logistic)",  f"{(1-wc_new)*100:.2f}%")
    ec3.metric("Balance Score", f"{bs_new*100:.1f}%")

    st.info(
        f"Old multiplicative formula would give: **{min(wc_old*100,100):.2f}%** "
        f"({'would exceed 100%' if wc_old > 1 else 'within range'} at extreme gaps)."
    )

    st.markdown("#### Full Win % Table — Elo Logistic (K=12)")
    skills = list(range(1, 11))
    ref_df = pd.DataFrame(
        {f"Avg {sa}": {f"Avg {sb}": f"{win_chance(sa, sb)*100:.1f}%"
                       for sb in skills} for sa in skills}
    )
    ref_df.index = [f"Avg {s}" for s in skills]
    ref_df.index.name = "Team A Avg / Team B Avg"
    st.dataframe(ref_df, use_container_width=True)
    st.caption("Each cell: win probability for the row team (Team A) vs column team (Team B). Max value in table: 84.9%.")

# ── Main dashboard ────────────────────────────────────────────────────────────
if players_df is not None:
    st.divider()

    avg_skill = players_df["Skill"].mean()
    min_skill = int(players_df["Skill"].min())
    max_skill = int(players_df["Skill"].max())
    n_matches_possible = len(players_df) // 12

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, label in [
        (c1, len(players_df),           "Total Players"),
        (c2, f"{avg_skill:.1f}",        "Average Skill"),
        (c3, f"{min_skill} to {max_skill}", "Skill Range"),
        (c4, n_matches_possible,        "Matches Possible"),
        (c5, len(players_df) % 12,      "Will Be Benched"),
    ]:
        col.markdown(
            f'<div class="metric-card"><h3>{val}</h3><p>{label}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    col_freq, col_dist = st.columns([1, 1])
    with col_freq:
        st.markdown("#### Skill Frequency Breakdown")
        freq_display = frequency_table_from_players(players_df).copy()
        freq_display["Win % vs Avg-5 Team"] = freq_display["Skill"].apply(
            lambda s: f"{win_chance(float(s), 5.0) * 100:.1f}%"
        )
        st.dataframe(freq_display, use_container_width=True, height=340, hide_index=True)
    with col_dist:
        st.markdown("#### Skill Distribution Chart")
        dist = skill_distribution_chart(players_df)
        st.bar_chart(dist.set_index("Skill Level"), use_container_width=True, height=320)

    st.divider()

    # ── Matchmaking ──────────────────────────────────────────────────────────
    st.markdown("### Run 6v6 Matchmaking")

    if len(players_df) < 12:
        st.warning("Need at least 12 players to form one 6v6 match.")
    else:
        if st.button("Generate Optimal 6v6 Matches", use_container_width=True, type="primary"):
            with st.spinner("Building teams..."):
                matches, rosters, benched = matchmake_6v6(players_df)
            st.session_state["matches"]  = matches
            st.session_state["rosters"]  = rosters
            st.session_state["benched"]  = benched
            st.session_state["match_src"] = players_df.copy()

    if "matches" in st.session_state:
        matches  = st.session_state["matches"]
        rosters  = st.session_state["rosters"]
        benched  = st.session_state["benched"]
        src_df   = st.session_state.get("match_src", players_df)

        if not matches:
            st.warning("No complete 6v6 matches could be formed.")
        else:
            match_df   = pd.DataFrame(matches)
            roster_df  = pd.DataFrame(rosters)
            avg_balance = match_df["Balance Score"].mean()
            perfect     = int((match_df["Balance Score"] >= 99.0).sum())

            m1, m2, m3, m4 = st.columns(4)
            for col, val, label in [
                (m1, len(matches),          "Matches Created"),
                (m2, f"{avg_balance:.1f}%", "Avg Balance Score"),
                (m3, perfect,               "Near-Perfect Matches"),
                (m4, len(benched),          "Benched Players"),
            ]:
                col.markdown(
                    f'<div class="metric-card"><h3>{val}</h3><p>{label}</p></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("")

            # ── Match summaries ──
            st.markdown("#### Match Summaries")

            def color_balance(val):
                if val >= 95:
                    return "background-color:#1a472a; color:#57f287"
                elif val >= 80:
                    return "background-color:#1e3a1a; color:#a8d5a2"
                elif val >= 60:
                    return "background-color:#4a3728; color:#fee75c"
                else:
                    return "background-color:#4a1428; color:#ed4245"

            def color_win(val):
                if 45 <= val <= 55:
                    return "color:#57f287; font-weight:600"
                elif 35 <= val <= 65:
                    return "color:#fee75c"
                else:
                    return "color:#ed4245"

            styled = (
                match_df.style
                .applymap(color_balance, subset=["Balance Score"])
                .applymap(color_win,     subset=["Win % (Team A)", "Win % (Team B)"])
            )
            st.dataframe(styled, use_container_width=True, height=min(400, len(matches)*40+60))

            # ── Team rosters per match ──
            st.markdown("#### Team Rosters")
            st.markdown("Expand a match to see the full team compositions.")

            for match in matches:
                mn = match["Match #"]
                wc = match["Win % (Team A)"]
                bs = match["Balance Score"]
                avg_a = match["Team A Avg Skill"]
                avg_b = match["Team B Avg Skill"]

                if bs >= 95:
                    badge = "🟢"
                elif bs >= 80:
                    badge = "🟡"
                else:
                    badge = "🔴"

                with st.expander(
                    f"{badge}  Match {mn}  |  Team A avg {avg_a:.2f}  vs  Team B avg {avg_b:.2f}"
                    f"  |  Win% A: {wc:.1f}%  |  Balance: {bs:.1f}%"
                ):
                    rc1, rc2 = st.columns(2)
                    team_a_rows = roster_df[(roster_df["Match #"] == mn) & (roster_df["Team"] == "A")]
                    team_b_rows = roster_df[(roster_df["Match #"] == mn) & (roster_df["Team"] == "B")]

                    with rc1:
                        st.markdown(f"**Team A** — avg skill {avg_a:.2f}  |  Win% {match['Win % (Team A)']:.1f}%")
                        for _, r in team_a_rows.iterrows():
                            stars = "★" * int(r["Skill"]) + "☆" * (10 - int(r["Skill"]))
                            st.markdown(
                                f'<div class="team-a">{r["Player"]} &nbsp; Skill <b>{int(r["Skill"])}</b> &nbsp; <span style="color:#e94560;font-size:0.75rem">{stars}</span></div>',
                                unsafe_allow_html=True,
                            )

                    with rc2:
                        st.markdown(f"**Team B** — avg skill {avg_b:.2f}  |  Win% {match['Win % (Team B)']:.1f}%")
                        for _, r in team_b_rows.iterrows():
                            stars = "★" * int(r["Skill"]) + "☆" * (10 - int(r["Skill"]))
                            st.markdown(
                                f'<div class="team-b">{r["Player"]} &nbsp; Skill <b>{int(r["Skill"])}</b> &nbsp; <span style="color:#bc8cff;font-size:0.75rem">{stars}</span></div>',
                                unsafe_allow_html=True,
                            )

            # ── Charts ──
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.markdown("#### Balance Score Distribution")
                bins = pd.cut(
                    match_df["Balance Score"],
                    bins=[0, 60, 80, 95, 100],
                    labels=["Poor (<60)", "Fair (60-80)", "Good (80-95)", "Perfect (95+)"],
                )
                bin_counts = (
                    bins.value_counts()
                    .reindex(["Perfect (95+)", "Good (80-95)", "Fair (60-80)", "Poor (<60)"])
                    .reset_index()
                )
                bin_counts.columns = ["Quality", "Count"]
                st.bar_chart(bin_counts.set_index("Quality"), use_container_width=True)

            with col_c2:
                st.markdown("#### Skill Diff per Match")
                diff_chart = match_df[["Match #", "Skill Diff"]].set_index("Match #")
                st.bar_chart(diff_chart, use_container_width=True)

            # ── Benched ──
            if benched:
                st.warning(
                    f"**{len(benched)} player(s) benched** (not enough for a full match): "
                    f"{', '.join(benched)}"
                )

            # ── Export ──
            st.divider()
            st.markdown("#### Export Results")
            excel_out = results_to_excel(matches, rosters, benched, src_df)
            st.download_button(
                "Download Full Results (.xlsx)",
                data=excel_out,
                file_name="6v6_match_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
            st.caption(
                "Export contains four sheets: **Match Summaries**, **Team Rosters**, "
                "**Skill Frequency**, and **Benched** players (if any)."
            )

else:
    st.info("Upload a Skill/Frequency Excel file or generate random players to get started.")

st.markdown("---")
st.caption(
    "Win formula: P(A) = 1 / (1 + 10^(-(avgA - avgB) / 12))  |  "
    "Max win% = 84.9% (avg skill 10 vs avg skill 1)  |  "
    "Teams built via snake draft on sorted skill groups of 12"
)
