# streamlit_app.py
import os
import re
import time
import html
import streamlit as st
import pandas as pd
import psycopg2
from recommender import HybridRecommender

st.set_page_config(page_title="Hybrid Recommender Demo", layout="wide")


# ---------------------------
# DB connection helper
# ---------------------------
def get_conn_params():
    if "db" in st.secrets:
        db = st.secrets["db"]
        return {
            "host": db.get("host", "localhost"),
            "port": int(db.get("port", 5432)),
            "dbname": db.get("dbname"),
            "user": db.get("user"),
            "password": db.get("password"),
        }
    return {
        "host": os.environ.get("DB_HOST", "localhost"),
        "port": int(os.environ.get("DB_PORT", 5432)),
        "dbname": os.environ.get("DB_NAME", "postgres"),
        "user": os.environ.get("DB_USER", "postgres"),
        "password": os.environ.get("DB_PASS", ""),
    }


# ---------------------------
# Helpers
# ---------------------------
def safe_rerun():
    try:
        st.rerun()
    except Exception:
        st.stop()


def highlight_match(text: str, query: str, highlight_bg: str = "#007acc", highlight_fg: str = "#ffffff") -> str:
    if not query:
        return html.escape(text)
    try:
        q = re.escape(query)
        pattern = re.compile(q, re.IGNORECASE)
        def _repl(m):
            s = m.group(0)
            return f"<span style='background:{highlight_bg}; color:{highlight_fg}; padding:2px 4px; border-radius:4px;'>{html.escape(s)}</span>"
        res = pattern.sub(_repl, text)
        return res
    except Exception:
        return html.escape(text)


# ---------------------------
# Initialize recommender & state
# ---------------------------
if "reco_obj" not in st.session_state:
    conn = get_conn_params()
    st.session_state.reco_obj = HybridRecommender(conn_params=conn, region="GLOBAL")
    st.session_state.region_selected = False
    st.session_state.basket = set()

reco: HybridRecommender = st.session_state.reco_obj


# ---------------------------
# Region selection (from DB)
# ---------------------------
st.title("Hybrid Recommender — Live Demo")
st.write("Select region first to see cold-start recommendations.")

if not st.session_state.region_selected:
    st.subheader("Step 1 — Select your region")

    try:
        conn_params = get_conn_params()
        conn = psycopg2.connect(**conn_params)
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT region FROM item_popularity ORDER BY region;")
        rows = cur.fetchall()
        cur.close(); conn.close()
        regions = [r[0] for r in rows if r[0] is not None]
    except Exception as e:
        st.warning(f"Could not load regions from DB; falling back to GLOBAL. (Error: {e})")
        regions = []

    if "GLOBAL" in regions:
        regions = ["GLOBAL"] + [r for r in regions if r != "GLOBAL"]
    elif regions:
        regions = regions
    else:
        regions = ["GLOBAL"]

    region = st.selectbox("Choose a region", regions, index=0)

    if st.button("Confirm region"):
        try:
            reco.set_region(region)
            st.session_state.region_selected = True
        except Exception as e:
            st.error("Failed to set region: " + str(e))
        safe_rerun()

    st.stop()


# ---------------------------
# UI Styles (kept same as earlier)
# ---------------------------
CARD_STYLE = """
border-radius: 10px;
border: 1px solid rgba(0,0,0,0.08);
padding: 12px;
margin-bottom: 12px;
background: linear-gradient(180deg, #0f1724, #10243a);
color: #ffffff;
"""
DESCRIPTION_STYLE = "color:#ffffff; font-size:14px; line-height:1.35;"


# ---------------------------
# Search callback: update matches on every keystroke
# ---------------------------
def update_search_matches():
    q = st.session_state.get("search_input", "") or ""
    q_str = str(q).strip()
    df = pd.DataFrame(list(reco.id_to_item.items()), columns=["item_id", "description"])
    if q_str == "":
        st.session_state.search_matches = []
        return
    mask_desc = df["description"].str.contains(q_str, case=False, na=False)
    digit_part = "".join([c for c in q_str if c.isdigit()])
    mask_id = pd.Series([False] * len(df))
    if q_str.isdigit():
        mask_id = df["item_id"].astype(str).str.contains(q_str)
    elif digit_part:
        mask_id = df["item_id"].astype(str).str.contains(digit_part)
    final_mask = mask_desc | mask_id
    matches_df = df[final_mask].head(100)
    st.session_state.search_matches = matches_df.to_dict("records")


# ---------------------------
# Sidebar: basket, search, controls
# ---------------------------
with st.sidebar:
    st.header("Your Basket")
    if st.session_state.basket:
        for i, iid in enumerate(sorted(list(st.session_state.basket))):
            st.markdown(f"**{iid}** — {reco.id_to_item.get(iid, '<unknown>')}")
            # remove button using session state
            if st.button("Remove", key=f"remove_{iid}_{i}"):
                st.session_state.basket.discard(iid)
                reco.basket.discard(iid)
                safe_rerun()
            st.markdown("---")
    else:
        st.info("Basket is empty. Add items from recommendations or search.")

    st.subheader("Search products")
    # text_input with on_change callback
    st.text_input("Search by description or item id", value="", key="search_input", on_change=update_search_matches)

    # show search results
    matches = st.session_state.get("search_matches", []) or []
    if matches:
        st.markdown(f"**Total matching: {len(matches)}**")
        for row in matches[:20]:
            # highlight the typed substring
            q_cur = st.session_state.get("search_input", "") or ""
            desc_html = highlight_match(row["description"], q_cur, highlight_bg="#007acc", highlight_fg="#ffffff")
            id_html = highlight_match(str(row["item_id"]), q_cur, highlight_bg="#5c6cff", highlight_fg="#ffffff")
            cols = st.columns([7, 2])
            with cols[0]:
                st.markdown(f"**ID {id_html}** — <span style='{DESCRIPTION_STYLE}'>{desc_html}</span>", unsafe_allow_html=True)
            with cols[1]:
                if st.button("Add", key=f"search_add_{row['item_id']}"):
                    st.session_state.basket.add(int(row['item_id']))
                    reco.basket.add(int(row['item_id']))
                    safe_rerun()

    st.markdown("---")
    if st.button("Reset basket"):
        st.session_state.basket = set()
        reco.reset_basket()
        safe_rerun()

    top_n = st.slider("Top N recommendations", 5, 50, 15)
    st.caption(f"Region chosen: {reco.region}")


# ---------------------------
# Main area: recommendations
# ---------------------------
st.subheader("Recommendations")

if st.session_state.basket != reco.basket:
    reco.basket = set(st.session_state.basket)

try:
    results = reco.recommend(top_n=top_n)
except Exception as e:
    st.error("Error fetching recommendations: " + str(e))
    st.stop()

if not results:
    st.info("No recommendations yet. Try adding items or adjust Top N.")
else:
    for r in results:
        score = r.get("final_score") or r.get("region_score") or r.get("global_score") or 0.0
        src = r.get("source", "popularity")
        if src == "region":
            reason = "Popular in selected region"
        elif src == "global":
            reason = "Popular globally"
        elif src == "rules":
            reason = "Association-rule based: users who bought items in your basket also bought this"
        else:
            reason = "Popularity-based"

        st.markdown(
            f"""
            <div style="{CARD_STYLE}">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <div style="flex:1; padding-right:12px;">
                  <h4 style="margin:0 0 6px 0; color:#ffffff;">{r['description']}</h4>
                  <div style="color:#d0d7e0; margin-bottom:6px;">Item ID: <b style="color:#ffffff;">{r['item_id']}</b> — <i>{reason}</i></div>
                  <div style="{DESCRIPTION_STYLE}">Source: <b>{src}</b></div>
                </div>
                <div style="text-align:right; min-width:140px;">
                  <div style="color:#a7e3ff; font-weight:700; font-size:16px;">Score: {score:.3f}</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(f"➕ Add item {r['item_id']}", key=f"rec_add_{r['item_id']}"):
            st.session_state.basket.add(int(r['item_id']))
            reco.basket.add(int(r['item_id']))
            safe_rerun()

st.markdown("---")
st.caption("Hint: search results update as you type.")

# # streamlit_app.py
# import os
# import re
# import time
# import html
# import streamlit as st
# import pandas as pd
# import psycopg2
# from recommender import HybridRecommender

# st.set_page_config(page_title="Hybrid Recommender Demo", layout="wide")


# # ---------------------------
# # DB connection helper
# # ---------------------------
# def get_conn_params():
#     if "db" in st.secrets:
#         db = st.secrets["db"]
#         return {
#             "host": db.get("host", "localhost"),
#             "port": int(db.get("port", 5432)),
#             "dbname": db.get("dbname"),
#             "user": db.get("user"),
#             "password": db.get("password"),
#         }
#     return {
#         "host": os.environ.get("DB_HOST", "localhost"),
#         "port": int(os.environ.get("DB_PORT", 5432)),
#         "dbname": os.environ.get("DB_NAME", "postgres"),
#         "user": os.environ.get("DB_USER", "postgres"),
#         "password": os.environ.get("DB_PASS", ""),
#     }


# # ---------------------------
# # Helpers
# # ---------------------------
# def safe_rerun():
#     try:
#         st.rerun()
#     except Exception:
#         st.stop()


# def highlight_match(text: str, query: str, highlight_bg: str = "#007acc", highlight_fg: str = "#ffffff") -> str:
#     if not query:
#         return html.escape(text)
#     try:
#         q = re.escape(query)
#         pattern = re.compile(q, re.IGNORECASE)
#         def _repl(m):
#             s = m.group(0)
#             return f"<span style='background:{highlight_bg}; color:{highlight_fg}; padding:2px 4px; border-radius:4px;'>{html.escape(s)}</span>"
#         res = pattern.sub(_repl, text)
#         return res
#     except Exception:
#         return html.escape(text)


# # ---------------------------
# # Initialize recommender & state
# # ---------------------------
# if "reco_obj" not in st.session_state:
#     conn = get_conn_params()
#     st.session_state.reco_obj = HybridRecommender(conn_params=conn, region="GLOBAL")
#     st.session_state.region_selected = False
#     st.session_state.basket = set()
#     st.session_state.search_matches = []    # list of dict rows for search results

# reco: HybridRecommender = st.session_state.reco_obj


# # ---------------------------
# # Region selection (from DB)
# # ---------------------------
# st.title("Hybrid Recommender — Live Demo")
# st.write("Select region first to see cold-start recommendations.")

# if not st.session_state.region_selected:
#     st.subheader("Step 1 — Select your region")

#     try:
#         conn_params = get_conn_params()
#         conn = psycopg2.connect(**conn_params)
#         cur = conn.cursor()
#         cur.execute("SELECT DISTINCT region FROM item_popularity ORDER BY region;")
#         rows = cur.fetchall()
#         cur.close(); conn.close()
#         regions = [r[0] for r in rows if r[0] is not None]
#     except Exception as e:
#         st.warning(f"Could not load regions from DB; falling back to GLOBAL. (Error: {e})")
#         regions = []

#     if "GLOBAL" in regions:
#         regions = ["GLOBAL"] + [r for r in regions if r != "GLOBAL"]
#     elif regions:
#         regions = regions
#     else:
#         regions = ["GLOBAL"]

#     region = st.selectbox("Choose a region", regions, index=0)

#     if st.button("Confirm region"):
#         try:
#             reco.set_region(region)
#             st.session_state.region_selected = True
#         except Exception as e:
#             st.error("Failed to set region: " + str(e))
#         safe_rerun()

#     st.stop()


# # ---------------------------
# # UI Styles (kept same as earlier)
# # ---------------------------
# CARD_STYLE = """
# border-radius: 10px;
# border: 1px solid rgba(0,0,0,0.08);
# padding: 12px;
# margin-bottom: 12px;
# background: linear-gradient(180deg, #0f1724, #10243a);
# color: #ffffff;
# """
# DESCRIPTION_STYLE = "color:#ffffff; font-size:14px; line-height:1.35;"


# # ---------------------------
# # Search callback: update matches on every keystroke
# # ---------------------------
# def update_search_matches():
#     q = st.session_state.get("search_input", "") or ""
#     q_str = str(q).strip()
#     df = pd.DataFrame(list(reco.id_to_item.items()), columns=["item_id", "description"])
#     if q_str == "":
#         st.session_state.search_matches = []
#         return
#     mask_desc = df["description"].str.contains(q_str, case=False, na=False)
#     digit_part = "".join([c for c in q_str if c.isdigit()])
#     mask_id = pd.Series([False] * len(df))
#     if q_str.isdigit():
#         mask_id = df["item_id"].astype(str).str.contains(q_str)
#     elif digit_part:
#         mask_id = df["item_id"].astype(str).str.contains(digit_part)
#     final_mask = mask_desc | mask_id
#     matches_df = df[final_mask].head(100)
#     st.session_state.search_matches = matches_df.to_dict("records")


# # ---------------------------
# # Sidebar: basket, search, controls
# # ---------------------------
# with st.sidebar:
#     st.header("Your Basket")
#     if st.session_state.basket:
#         for i, iid in enumerate(sorted(list(st.session_state.basket))):
#             st.markdown(f"**{iid}** — {reco.id_to_item.get(iid, '<unknown>')}")
#             # remove button using session state
#             if st.button("Remove", key=f"remove_{iid}_{i}"):
#                 st.session_state.basket.discard(iid)
#                 reco.basket.discard(iid)
#                 safe_rerun()
#             st.markdown("---")
#     else:
#         st.info("Basket is empty. Add items from recommendations or search.")

#     st.subheader("Search products")
#     # text_input with on_change callback
#     st.text_input("Search by description or item id", value="", key="search_input", on_change=update_search_matches)

#     # show search results
#     matches = st.session_state.get("search_matches", []) or []
#     if matches:
#         st.markdown(f"**Total matching: {len(matches)}**")
#         for row in matches[:20]:
#             # highlight the typed substring
#             q_cur = st.session_state.get("search_input", "") or ""
#             desc_html = highlight_match(row["description"], q_cur, highlight_bg="#007acc", highlight_fg="#ffffff")
#             id_html = highlight_match(str(row["item_id"]), q_cur, highlight_bg="#5c6cff", highlight_fg="#ffffff")
#             cols = st.columns([7, 2])
#             with cols[0]:
#                 st.markdown(f"**ID {id_html}** — <span style='{DESCRIPTION_STYLE}'>{desc_html}</span>", unsafe_allow_html=True)
#             with cols[1]:
#                 if st.button("Add", key=f"search_add_{row['item_id']}"):
#                     st.session_state.basket.add(int(row['item_id']))
#                     reco.basket.add(int(row['item_id']))
#                     safe_rerun()

#     st.markdown("---")
#     if st.button("Reset basket"):
#         st.session_state.basket = set()
#         reco.reset_basket()
#         safe_rerun()

#     top_n = st.slider("Top N recommendations", 5, 50, 15)
#     st.caption(f"Region chosen: {reco.region}")


# # ---------------------------
# # Main area: recommendations
# # ---------------------------
# st.subheader("Recommendations")

# if st.session_state.basket != reco.basket:
#     reco.basket = set(st.session_state.basket)

# try:
#     results = reco.recommend(top_n=top_n)
# except Exception as e:
#     st.error("Error fetching recommendations: " + str(e))
#     st.stop()

# if not results:
#     st.info("No recommendations yet. Try adding items or adjust Top N.")
# else:
#     for r in results:
#         score = r.get("final_score") or r.get("region_score") or r.get("global_score") or 0.0
#         src = r.get("source", "popularity")
#         if src == "region":
#             reason = "Popular in selected region"
#         elif src == "global":
#             reason = "Popular globally"
#         elif src == "rules":
#             reason = "Association-rule based: users who bought items in your basket also bought this"
#         else:
#             reason = "Popularity-based"

#         st.markdown(
#             f"""
#             <div style="{CARD_STYLE}">
#               <div style="display:flex; justify-content:space-between; align-items:center;">
#                 <div style="flex:1; padding-right:12px;">
#                   <h4 style="margin:0 0 6px 0; color:#ffffff;">{r['description']}</h4>
#                   <div style="color:#d0d7e0; margin-bottom:6px;">Item ID: <b style="color:#ffffff;">{r['item_id']}</b> — <i>{reason}</i></div>
#                   <div style="{DESCRIPTION_STYLE}">Source: <b>{src}</b></div>
#                 </div>
#                 <div style="text-align:right; min-width:140px;">
#                   <div style="color:#a7e3ff; font-weight:700; font-size:16px;">Score: {score:.3f}</div>
#                 </div>
#               </div>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )

#         if st.button(f"➕ Add item {r['item_id']}", key=f"rec_add_{r['item_id']}"):
#             st.session_state.basket.add(int(r['item_id']))
#             reco.basket.add(int(r['item_id']))
#             safe_rerun()

# st.markdown("---")
# st.caption("Hint: search results update as you type.")