import streamlit as st
import pandas as pd
import numpy as np
import random
import io
from itertools import combinations

# ─────────────────────────────────────────────
#  CORE LOGIC
# ─────────────────────────────────────────────

def win_chance(skill_a: int, skill_b: int) -> float:
    """
    Returns the win probability for Player A vs Player B.
    Base = 50%.  Each bracket A is above B multiplies by 1.1;
    each bracket A is below B multiplies by 0.9.
    e.g. skill 5 vs 3 → 0.5 * 1.1^2 = 0.605  ✓
         skill 5 vs 10 → 0.5 * 0.9^5 = 0.2952 ✓
    """
    diff = skill_a - skill_b
    return 0.5 * (1.1 ** diff)


def balance_score(skill_a: int, skill_b: int) -> float:
    """
    How close to a 50/50 match this pairing is.
    Score = 1 − |win_chance − 0.5|  (1.0 = perfect balance)
    """
    wc = win_chance(skill_a, skill_b)
    return 1.0 - abs(wc - 0.5)


def generate_skill_level() -> int:
    """
    Normal distribution centred on 5, clipped to [1, 10].
    """
    val = int(round(random.gauss(5, 1.8)))
    return max(1, min(10, val))


def generate_players(n: int) -> pd.DataFrame:
    random.seed(42)
    players = []
    for i in range(1, n + 1):
        players.append({"Player": f"Player_{i:03d}", "Skill": generate_skill_level()})
    return pd.DataFrame(players)


def greedy_matchmake(players_df: pd.DataFrame) -> tuple[list[dict], list[str]]:
    """
    Optimal-ish matchmaking using a greedy approach on a sorted list.

    1. Sort players by skill.
    2. Use a round-robin min-cost pairing on the sorted array:
       try every consecutive pair in a sliding window and always
       pick the best remaining unmatched pair.

    Returns: (matches, unmatched_players)
    """
    df = players_df.copy().reset_index(drop=True)
    n = len(df)

    # Build all possible pairs with their balance score
    pair_scores = []
    for i in range(n):
        for j in range(i + 1, n):
            a = df.iloc[i]
            b = df.iloc[j]
            score = balance_score(int(a["Skill"]), int(b["Skill"]))
            pair_scores.append((score, i, j))

    # Sort by best balance score descending
    pair_scores.sort(key=lambda x: -x[0])

    matched = set()
    matches = []
    for score, i, j in pair_scores:
        if i in matched or j in matched:
            continue
        a = df.iloc[i]
        b = df.iloc[j]
        wc = win_chance(int(a["Skill"]), int(b["Skill"]))
        matches.append({
            "Match #": len(matches) + 1,
            "Player A": a["Player"],
            "Skill A": int(a["Skill"]),
            "Player B": b["Player"],
            "Skill B": int(b["Skill"]),
            "Win % (A)": round(wc * 100, 2),
            "Win % (B)": round((1 - wc) * 100, 2),
            "Balance Score": round(score * 100, 2),
        })
        matched.add(i)
        matched.add(j)

    unmatched = [df.iloc[k]["Player"] for k in range(n) if k not in matched]
    return matches, unmatched


def skill_distribution_chart(df: pd.DataFrame):
    counts = df["Skill"].value_counts().sort_index().reset_index()
    counts.columns = ["Skill Level", "Player Count"]
    return counts


# ─────────────────────────────────────────────
#  EXCEL HELPERS
# ─────────────────────────────────────────────

def read_excel_players(file) -> pd.DataFrame:
    df = pd.read_excel(file)
    # Normalise column names
    df.columns = [c.strip().title() for c in df.columns]
    if "Player" not in df.columns or "Skill" not in df.columns:
        raise ValueError("Excel file must have 'Player' and 'Skill' columns.")
    df["Skill"] = pd.to_numeric(df["Skill"], errors="coerce").clip(1, 10).fillna(5).astype(int)
    df = df[["Player", "Skill"]].dropna()
    return df


def create_sample_excel() -> bytes:
    df = generate_players(20)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Players")
    return buf.getvalue()


def matches_to_excel(matches: list[dict], unmatched: list[str]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame(matches).to_excel(writer, index=False, sheet_name="Matches")
        if unmatched:
            pd.DataFrame({"Unmatched Players": unmatched}).to_excel(
                writer, index=False, sheet_name="Unmatched"
            )
    return buf.getvalue()


# ─────────────────────────────────────────────
#  STREAMLIT UI
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Skill-Based Matchmaking",
    page_icon="⚔️",
    layout="wide",
)

# ── Header ──────────────────────────────────
st.markdown("""
<style>
    .title-block { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                   padding: 2rem 2.5rem; border-radius: 12px; margin-bottom: 1.5rem; }
    .title-block h1 { color: #e94560; margin: 0; font-size: 2.2rem; }
    .title-block p  { color: #a8b2d8; margin: 0.5rem 0 0; font-size: 1rem; }
    .metric-card    { background:#1e1e2e; border:1px solid #2d2d44; border-radius:10px;
                      padding:1rem 1.5rem; text-align:center; }
    .metric-card h3 { color:#e94560; font-size:2rem; margin:0; }
    .metric-card p  { color:#a8b2d8; margin:0.25rem 0 0; font-size:0.85rem; }
    .formula-box    { background:#0d1117; border-left:4px solid #e94560;
                      border-radius:6px; padding:0.8rem 1.2rem; font-family:monospace;
                      color:#58a6ff; font-size:0.9rem; margin:0.5rem 0; }
    .badge-green  { background:#1a472a; color:#57f287; padding:2px 10px;
                    border-radius:99px; font-size:0.78rem; font-weight:600; }
    .badge-yellow { background:#4a3728; color:#fee75c; padding:2px 10px;
                    border-radius:99px; font-size:0.78rem; font-weight:600; }
    .badge-red    { background:#4a1428; color:#ed4245; padding:2px 10px;
                    border-radius:99px; font-size:0.78rem; font-weight:600; }
</style>
<div class="title-block">
  <h1>⚔️ Skill-Based Matchmaking System</h1>
  <p>Fair 1v1 pairings · Win probabilities · Optimal balance scoring</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar: How it works ────────────────────
with st.sidebar:
    st.markdown("### 📐 How Win % is Calculated")
    st.markdown("""
**Base:** Same skill → **50% win chance**

Each bracket *above* opponent → ×1.1  
Each bracket *below* opponent → ×0.9

**Formula:**
""")
    st.markdown('<div class="formula-box">win% = 0.5 × 1.1^(skillA − skillB)</div>', unsafe_allow_html=True)
    st.markdown("""
**Examples:**
- Skill 5 vs 5 → **50.00%**
- Skill 5 vs 3 → **60.50%**
- Skill 5 vs 10 → **29.52%**

**Matchmaking goal:** Pair players so that win percentages are as close to 50% as possible (Balance Score → 100%).
""")
    st.divider()
    st.markdown("### 📁 Sample File")
    sample_bytes = create_sample_excel()
    st.download_button(
        "⬇️ Download Sample Excel",
        data=sample_bytes,
        file_name="sample_players.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.caption("Excel must have **Player** and **Skill** columns. Skill range: 1–10.")

# ── Tabs ────────────────────────────────────
tab_upload, tab_generate = st.tabs(["📂 Upload Excel", "🎲 Generate Random Players"])

players_df = None

# ── Tab 1: Upload ────────────────────────────
with tab_upload:
    uploaded = st.file_uploader("Upload your player list (.xlsx)", type=["xlsx", "xls"])
    if uploaded:
        try:
            players_df = read_excel_players(uploaded)
            st.success(f"✅ Loaded **{len(players_df)} players** from file.")
        except Exception as e:
            st.error(f"❌ {e}")

# ── Tab 2: Generate ──────────────────────────
with tab_generate:
    col_n, col_seed = st.columns([2, 1])
    with col_n:
        n_players = st.slider("Number of players to generate", 4, 100, 16, step=2)
    with col_seed:
        seed = st.number_input("Random seed", value=42, step=1)

    if st.button("🎲 Generate Players", use_container_width=True):
        random.seed(int(seed))
        players_df = pd.DataFrame(
            [{"Player": f"Player_{i:03d}", "Skill": generate_skill_level()} for i in range(1, n_players + 1)]
        )
        st.session_state["generated_df"] = players_df
        st.success(f"✅ Generated **{len(players_df)} players**.")

    if "generated_df" in st.session_state and players_df is None:
        players_df = st.session_state["generated_df"]

# ── Main content: show roster + run matchmaking ──
if players_df is not None:
    st.divider()

    # ── Roster Overview ──
    col1, col2, col3, col4 = st.columns(4)
    avg_skill = players_df["Skill"].mean()
    min_skill  = players_df["Skill"].min()
    max_skill  = players_df["Skill"].max()
    n_matchable = (len(players_df) // 2) * 2

    for col, val, label in [
        (col1, len(players_df), "Total Players"),
        (col2, f"{avg_skill:.1f}", "Average Skill"),
        (col3, f"{min_skill} – {max_skill}", "Skill Range"),
        (col4, len(players_df) % 2, "Will Sit Out"),
    ]:
        col.markdown(f"""<div class="metric-card"><h3>{val}</h3><p>{label}</p></div>""",
                     unsafe_allow_html=True)

    st.markdown("")

    col_roster, col_dist = st.columns([1, 1])

    with col_roster:
        st.markdown("#### 👥 Player Roster")
        st.dataframe(
            players_df.sort_values("Skill", ascending=False).reset_index(drop=True),
            use_container_width=True,
            height=320,
        )

    with col_dist:
        st.markdown("#### 📊 Skill Distribution")
        dist = skill_distribution_chart(players_df)
        st.bar_chart(dist.set_index("Skill Level"), use_container_width=True, height=300)

    # ── Win Probability Reference Table ──────
    with st.expander("🔢 Win Probability Reference Table (all skill combos)"):
        skills = list(range(1, 11))
        ref = {}
        for sa in skills:
            ref[sa] = {sb: f"{win_chance(sa, sb)*100:.1f}%" for sb in skills}
        ref_df = pd.DataFrame(ref).T
        ref_df.index.name = "Skill A ↓ / Skill B →"
        st.dataframe(ref_df, use_container_width=True)

    st.divider()

    # ── Matchmaking ──────────────────────────
    st.markdown("### ⚔️ Run Matchmaking")

    if st.button("🚀 Generate Optimal Matches", use_container_width=True, type="primary"):
        with st.spinner("Finding optimal pairings…"):
            matches, unmatched = greedy_matchmake(players_df)

        st.session_state["matches"] = matches
        st.session_state["unmatched"] = unmatched

    if "matches" in st.session_state:
        matches  = st.session_state["matches"]
        unmatched = st.session_state["unmatched"]

        if not matches:
            st.warning("Not enough players to form a match.")
        else:
            match_df = pd.DataFrame(matches)

            avg_balance = match_df["Balance Score"].mean()
            perfect = (match_df["Balance Score"] == 100).sum()
            near_50 = match_df[(match_df["Win % (A)"] >= 45) & (match_df["Win % (A)"] <= 55)].shape[0]

            mc1, mc2, mc3, mc4 = st.columns(4)
            for col, val, label in [
                (mc1, len(matches), "Matches Created"),
                (mc2, f"{avg_balance:.1f}%", "Avg Balance Score"),
                (mc3, perfect, "Perfect 50/50 Matches"),
                (mc4, len(unmatched), "Unmatched Players"),
            ]:
                col.markdown(f"""<div class="metric-card"><h3>{val}</h3><p>{label}</p></div>""",
                             unsafe_allow_html=True)

            st.markdown("")

            # ── Colour-coded match table ──
            st.markdown("#### 🏆 Match Results")

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

            # ── Balance distribution chart ──
            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                st.markdown("#### 📈 Balance Score Distribution")
                bins = pd.cut(match_df["Balance Score"], bins=[0,60,80,95,100], labels=["Poor","Fair","Good","Perfect"])
                bin_counts = bins.value_counts().reindex(["Perfect","Good","Fair","Poor"]).reset_index()
                bin_counts.columns = ["Quality", "Count"]
                st.bar_chart(bin_counts.set_index("Quality"), use_container_width=True)

            with col_chart2:
                st.markdown("#### 📉 Win % Distribution (Player A)")
                win_hist = match_df["Win % (A)"].round(0).value_counts().sort_index().reset_index()
                win_hist.columns = ["Win %", "Matches"]
                st.bar_chart(win_hist.set_index("Win %"), use_container_width=True)

            # ── Unmatched players ──
            if unmatched:
                st.warning(f"⚠️ **{len(unmatched)} player(s) could not be matched** (odd number of players): {', '.join(unmatched)}")

            # ── Download results ──
            st.divider()
            st.markdown("#### 💾 Export Results")
            excel_out = matches_to_excel(matches, unmatched)
            st.download_button(
                "⬇️ Download Match Results (.xlsx)",
                data=excel_out,
                file_name="match_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

else:
    st.info("👆 Upload an Excel file or generate random players to get started.")

# ── Footer ────────────────────────────────────
st.markdown("---")
st.caption("Win probability formula: `P(A wins) = 0.5 × 1.1^(skillA − skillB)` · Balance Score = `100 × (1 − |P − 0.5|)`")
