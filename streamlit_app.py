import streamlit as st
import pandas as pd
import numpy as np
import random
import io

# =============================================================================
#  CORE LOGIC
# =============================================================================

def win_chance(skill_a: int, skill_b: int) -> float:
    """
    Win probability for Player A vs Player B.
    Base = 50%.  Each bracket A is above B multiplies by 1.1;
    each bracket A is below B multiplies by 0.9.
      Skill 5 vs 3  -> 0.5 * 1.1^2  = 0.6050
      Skill 5 vs 10 -> 0.5 * 0.9^5  = 0.2952
    """
    return 0.5 * (1.1 ** (skill_a - skill_b))


def balance_score(skill_a: int, skill_b: int) -> float:
    """1 = perfect 50/50, 0 = completely one-sided."""
    return 1.0 - abs(win_chance(skill_a, skill_b) - 0.5)


def generate_skill_level() -> int:
    """Normal distribution centred on 5, clipped to [1, 10]."""
    return max(1, min(10, int(round(random.gauss(5, 1.8)))))


def greedy_matchmake(players_df: pd.DataFrame):
    """
    Greedily pair players by best balance score (closest to 50/50).
    Returns (list[dict], list[str]) -> matches, unmatched player names.
    """
    df = players_df.copy().reset_index(drop=True)
    n = len(df)

    pair_scores = []
    for i in range(n):
        for j in range(i + 1, n):
            s = balance_score(int(df.at[i, "Skill"]), int(df.at[j, "Skill"]))
            pair_scores.append((s, i, j))
    pair_scores.sort(key=lambda x: -x[0])

    matched = set()
    matches = []
    for score, i, j in pair_scores:
        if i in matched or j in matched:
            continue
        a, b = df.iloc[i], df.iloc[j]
        wc = win_chance(int(a["Skill"]), int(b["Skill"]))
        matches.append({
            "Match #":       len(matches) + 1,
            "Player A":      a["Player"],
            "Skill A":       int(a["Skill"]),
            "Player B":      b["Player"],
            "Skill B":       int(b["Skill"]),
            "Win % (A)":     round(wc * 100, 2),
            "Win % (B)":     round((1 - wc) * 100, 2),
            "Balance Score": round(score * 100, 2),
        })
        matched.add(i)
        matched.add(j)

    unmatched = [df.iloc[k]["Player"] for k in range(n) if k not in matched]
    return matches, unmatched


# =============================================================================
#  EXCEL HELPERS
# =============================================================================

def read_excel_frequency(file) -> pd.DataFrame:
    """
    Read a Skill / Frequency Excel sheet and expand into one row per player.
    Auto-names players Player_001, Player_002, ...
    """
    df = pd.read_excel(file)
    df.columns = [c.strip().title() for c in df.columns]
    if "Skill" not in df.columns or "Frequency" not in df.columns:
        raise ValueError("Excel file must have 'Skill' and 'Frequency' columns.")

    df["Skill"]     = pd.to_numeric(df["Skill"],     errors="coerce").clip(1, 10).fillna(5).astype(int)
    df["Frequency"] = pd.to_numeric(df["Frequency"], errors="coerce").fillna(0).astype(int)
    df = df[df["Frequency"] > 0]

    if df.empty:
        raise ValueError("No valid rows found. Check that Frequency values are positive integers.")

    rows = []
    for _, row in df.iterrows():
        for _ in range(int(row["Frequency"])):
            rows.append(int(row["Skill"]))

    players_df = pd.DataFrame({"Skill": rows})
    players_df["Player"] = [f"Player_{i+1:03d}" for i in range(len(players_df))]
    return players_df[["Player", "Skill"]]


def frequency_table_from_players(players_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a player list back into Skill / Frequency."""
    freq = players_df["Skill"].value_counts().sort_index().reset_index()
    freq.columns = ["Skill", "Frequency"]
    return freq


def create_sample_excel() -> bytes:
    """Sample Excel in Skill / Frequency format (bell curve around 5)."""
    data = {
        "Skill":     [1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
        "Frequency": [1,  2,  4,  7, 10,  8,  5,  3,  2,  1],
    }
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame(data).to_excel(writer, index=False, sheet_name="Players")
    return buf.getvalue()


def matches_to_excel(matches, unmatched, players_df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame(matches).to_excel(writer, index=False, sheet_name="Matches")
        frequency_table_from_players(players_df).to_excel(
            writer, index=False, sheet_name="Skill Frequency"
        )
        if unmatched:
            pd.DataFrame({"Unmatched Players": unmatched}).to_excel(
                writer, index=False, sheet_name="Unmatched"
            )
    return buf.getvalue()


def skill_distribution_chart(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["Skill"].value_counts().sort_index().reset_index()
    counts.columns = ["Skill Level", "Player Count"]
    return counts


# =============================================================================
#  STREAMLIT UI
# =============================================================================

st.set_page_config(page_title="Skill-Based Matchmaking", page_icon="⚔️", layout="wide")

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
        padding: 1rem 1.5rem; text-align: center;
    }
    .metric-card h3 { color: #e94560; font-size: 2rem; margin: 0; }
    .metric-card p  { color: #a8b2d8; margin: 0.25rem 0 0; font-size: 0.85rem; }
    .formula-box {
        background: #0d1117; border-left: 4px solid #e94560;
        border-radius: 6px; padding: 0.8rem 1.2rem;
        font-family: monospace; color: #58a6ff; font-size: 0.9rem; margin: 0.5rem 0;
    }
</style>
<div class="title-block">
  <h1>Skill-Based Matchmaking System</h1>
  <p>Fair 1v1 pairings from skill frequency data &nbsp;·&nbsp; Win probabilities &nbsp;·&nbsp; Optimal balance scoring</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### How Win % is Calculated")
    st.markdown(
        "**Base:** Same skill level = **50% win chance**\n\n"
        "Each bracket *above* opponent multiplies by **x1.1**  \n"
        "Each bracket *below* opponent multiplies by **x0.9**"
    )
    st.markdown(
        '<div class="formula-box">P(A wins) = 0.5 x 1.1^(skillA - skillB)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "**Examples:**\n"
        "- Skill 5 vs 5 = **50.00%**\n"
        "- Skill 5 vs 3 = **60.50%**\n"
        "- Skill 5 vs 10 = **29.52%**\n\n"
        "**Matchmaking goal:** Pair players so win % is as close to 50% as possible."
    )
    st.divider()
    st.markdown("### Sample Excel File")
    st.markdown("Two columns: **Skill** (1-10) and **Frequency** (player count at that level).")
    st.download_button(
        "Download Sample Excel",
        data=create_sample_excel(),
        file_name="sample_skill_frequency.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# ── Input tabs ────────────────────────────────────────────────────────────────
tab_upload, tab_generate = st.tabs(["Upload Excel", "Generate Random Players"])

players_df = None

with tab_upload:
    st.markdown(
        "Upload an Excel file with **Skill** and **Frequency** columns. "
        "Each row defines a skill level and how many players have that skill."
    )
    uploaded = st.file_uploader("Choose .xlsx file", type=["xlsx", "xls"])
    if uploaded:
        try:
            players_df = read_excel_frequency(uploaded)
            freq_preview = frequency_table_from_players(players_df)
            total = freq_preview["Frequency"].sum()
            st.success(f"Loaded **{total} players** across **{len(freq_preview)} skill levels**.")
            st.dataframe(freq_preview, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Error: {e}")

with tab_generate:
    col_n, col_seed = st.columns([2, 1])
    with col_n:
        n_players = st.slider("Number of players to generate", 4, 200, 20, step=2)
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
        st.success(f"Generated **{n_players} players** - skill distribution below.")
        st.dataframe(freq_gen, use_container_width=True, hide_index=True)

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            freq_gen.to_excel(writer, index=False, sheet_name="Players")
        st.download_button(
            "Download as Skill/Frequency Excel",
            data=buf.getvalue(),
            file_name="generated_skill_frequency.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    if "generated_df" in st.session_state and players_df is None:
        players_df = st.session_state["generated_df"]

# ── Main dashboard ────────────────────────────────────────────────────────────
if players_df is not None:
    st.divider()

    avg_skill = players_df["Skill"].mean()
    min_skill = int(players_df["Skill"].min())
    max_skill = int(players_df["Skill"].max())

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, len(players_df),                  "Total Players"),
        (c2, f"{avg_skill:.1f}",               "Average Skill"),
        (c3, f"{min_skill} to {max_skill}",    "Skill Range"),
        (c4, len(players_df) % 2,              "Will Sit Out"),
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
        freq_display["Win % vs Avg (Skill 5)"] = freq_display["Skill"].apply(
            lambda s: f"{win_chance(int(s), 5) * 100:.1f}%"
        )
        st.dataframe(freq_display, use_container_width=True, height=360, hide_index=True)

    with col_dist:
        st.markdown("#### Skill Distribution Chart")
        dist = skill_distribution_chart(players_df)
        st.bar_chart(dist.set_index("Skill Level"), use_container_width=True, height=340)

    with st.expander("Win Probability Reference Table (all skill combos 1-10)"):
        skills = list(range(1, 11))
        ref_df = pd.DataFrame(
            {sa: {sb: f"{win_chance(sa, sb)*100:.1f}%" for sb in skills} for sa in skills}
        ).T
        ref_df.index.name = "Skill A / Skill B"
        st.dataframe(ref_df, use_container_width=True)

    st.divider()

    # ── Matchmaking ──────────────────────────────────────────────────────────
    st.markdown("### Run Matchmaking")

    if st.button("Generate Optimal Matches", use_container_width=True, type="primary"):
        with st.spinner("Finding optimal pairings..."):
            matches, unmatched = greedy_matchmake(players_df)
        st.session_state["matches"]   = matches
        st.session_state["unmatched"] = unmatched
        st.session_state["match_src"] = players_df.copy()

    if "matches" in st.session_state:
        matches   = st.session_state["matches"]
        unmatched = st.session_state["unmatched"]
        src_df    = st.session_state.get("match_src", players_df)

        if not matches:
            st.warning("Not enough players to form a match.")
        else:
            match_df    = pd.DataFrame(matches)
            avg_balance = match_df["Balance Score"].mean()
            perfect     = int((match_df["Balance Score"] >= 99.99).sum())

            m1, m2, m3, m4 = st.columns(4)
            for col, val, label in [
                (m1, len(matches),          "Matches Created"),
                (m2, f"{avg_balance:.1f}%", "Avg Balance Score"),
                (m3, perfect,               "Perfect 50/50 Matches"),
                (m4, len(unmatched),        "Unmatched Players"),
            ]:
                col.markdown(
                    f'<div class="metric-card"><h3>{val}</h3><p>{label}</p></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("")
            st.markdown("#### Match Results")

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
                .applymap(color_win,     subset=["Win % (A)", "Win % (B)"])
            )
            st.dataframe(styled, use_container_width=True, height=420)

            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.markdown("#### Balance Score Distribution")
                bins = pd.cut(
                    match_df["Balance Score"],
                    bins=[0, 60, 80, 95, 100],
                    labels=["Poor", "Fair", "Good", "Perfect"],
                )
                bin_counts = (
                    bins.value_counts()
                    .reindex(["Perfect", "Good", "Fair", "Poor"])
                    .reset_index()
                )
                bin_counts.columns = ["Quality", "Count"]
                st.bar_chart(bin_counts.set_index("Quality"), use_container_width=True)

            with col_c2:
                st.markdown("#### Win % Distribution (Player A)")
                win_hist = (
                    match_df["Win % (A)"].round(0)
                    .value_counts().sort_index().reset_index()
                )
                win_hist.columns = ["Win %", "Matches"]
                st.bar_chart(win_hist.set_index("Win %"), use_container_width=True)

            if unmatched:
                st.warning(
                    f"**{len(unmatched)} player(s) could not be matched** "
                    f"(odd total): {', '.join(unmatched)}"
                )

            st.divider()
            st.markdown("#### Export Results")
            excel_out = matches_to_excel(matches, unmatched, src_df)
            st.download_button(
                "Download Match Results (.xlsx)",
                data=excel_out,
                file_name="match_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
            st.caption(
                "Export includes three sheets: **Matches**, **Skill Frequency** summary, "
                "and **Unmatched** players (if any)."
            )

else:
    st.info("Upload a Skill/Frequency Excel file or generate random players to get started.")

st.markdown("---")
st.caption(
    "Win probability: P(A) = 0.5 x 1.1^(skillA - skillB)  "
    "| Balance Score: 100 x (1 - |P(A) - 0.5|)"
)
