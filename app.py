"""
app.py — RetailMind AI Agent: Streamlit entry point.

StyleCraft Product Intelligence Dashboard
Company: RetailMind Analytics
Client: StyleCraft (D2C fashion brand, 30 SKUs)

Architecture:
  - Router Pattern: LLM-based intent classification (NOT keyword/regex)
  - 6 Tool Functions: search_products, get_inventory_health, generate_restock_alert,
                      get_pricing_analysis, get_review_insights, get_category_performance
  - Multi-turn Memory: StreamlitChatMessageHistory + ConversationBufferMemory
  - Daily Briefing: Auto-generated on startup BEFORE user types anything

Run with: python run.py  OR  streamlit run app.py
"""

import os
import sys

import streamlit as st
from dotenv import load_dotenv

# Load .env before any other imports
load_dotenv()

# Add project root to path so 'core' and 'ui' imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_loader import cached_load_products, cached_load_reviews, cached_run_pipeline
from core.agent_factory import (
    PROVIDER_LABEL,
    MODEL_NAME,
    ENV_VAR,
    PLACEHOLDER,
    HELP_TEXT,
    get_llm,
    get_chat_router_agent,
    get_inventory_agent_notebook,
    get_bi_agent_notebook,
    generate_daily_briefing_text,
)
from core.tools import INVENTORY_TOOLS, BI_TOOLS
from ui.dashboard import render_dashboard
from ui.chat import render_chat
from ui.charts import render_charts
from ui.recommendations import render_recommendations


# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetailMind — StyleCraft Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Orchestrator (used by Dashboard "Run Full Analysis" button) ───────────────
class RetailMindOrchestrator:
    """
    Runs InventoryAnalystAgent → BusinessIntelligenceAgent in sequence.
    Each agent receives the accumulated output of all prior agents as context.
    Designed for the Dashboard 'Run Full Analysis' feature.
    """

    AGENT_TASKS = {
        "inventory_analyst": (
            "Perform a comprehensive inventory analysis of StyleCraft's catalog. "
            "Use generate_restock_alert to find all at-risk products. "
            "Then use get_inventory_health for the top 3 most critical items. "
            "Report: total critical count, names, days remaining, revenue at risk, "
            "and prioritised reorder recommendations."
        ),
        "business_intelligence": (
            "Perform a business intelligence review of StyleCraft's catalog. "
            "1) Use get_category_performance for each of the 5 categories. "
            "2) Identify the 3 lowest-margin products using get_pricing_analysis. "
            "3) Get review insights for the 2 worst-rated products using get_review_insights. "
            "Summarise: category health, margin issues, and customer satisfaction findings."
        ),
    }

    def __init__(self, inventory_agent, bi_agent):
        self.named_agents = {
            "inventory_analyst": inventory_agent,
            "business_intelligence": bi_agent,
        }

    def run_full_analysis(self, verbose: bool = True) -> dict:
        """Run all agents sequentially, passing accumulated context forward."""
        from datetime import datetime

        results = {"timestamp": datetime.now().isoformat(), "errors": []}
        accumulated_context = ""

        for phase_num, (agent_name, agent_exec) in enumerate(
            self.named_agents.items(), start=1
        ):
            try:
                base_task = self.AGENT_TASKS[agent_name]
                task = (
                    f"PRIOR ANALYSIS:\n{accumulated_context}\n\nYOUR TASK:\n{base_task}"
                    if accumulated_context
                    else base_task
                )

                if verbose:
                    print(f"\n{'='*60}\nPhase {phase_num}: {agent_name}\n{'='*60}")

                output = agent_exec.invoke({"input": task})["output"]
                results[agent_name] = output
                accumulated_context += (
                    f"\n\n--- PHASE {phase_num} ({agent_name.upper()}) OUTPUT ---\n{output}"
                )
            except Exception as e:
                err_msg = f"Phase {phase_num} ({agent_name}): {e}"
                results["errors"].append(err_msg)
                results[agent_name] = None
                if verbose:
                    print(f"ERROR: {err_msg}")

        return results


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    """Render the sidebar: model info, API key, catalog summary, category filter."""
    with st.sidebar:
        st.markdown("## 🛍️ RetailMind")
        st.markdown("*StyleCraft Product Intelligence*")
        st.divider()

        # ── Fixed Provider Info ───────────────────────────────────────────────
        st.markdown(f"**Model:** `{MODEL_NAME}`")
        st.caption(f"Provider: {PROVIDER_LABEL}")
        st.divider()

        # API key loaded from .env — no user input needed
        api_key_set = bool(os.environ.get(ENV_VAR, "").strip())
        st.session_state["api_key_set"] = api_key_set

        st.divider()

        # ── Category Filter ───────────────────────────────────────────────────
        st.markdown("**📂 Category Filter**")
        category_filter = st.selectbox(
            "category_selector",
            options=["All Categories", "Tops", "Dresses", "Bottoms", "Outerwear", "Accessories"],
            index=0,
            label_visibility="collapsed",
            key="category_filter",
        )

        st.divider()

        # ── Catalog Summary (always visible) ─────────────────────────────────
        st.markdown("**📊 Catalog Summary**")
        pipeline = st.session_state.get("pipeline", {})
        if pipeline:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total SKUs", pipeline.get("total_skus", 30))
                st.metric("🔴 Critical", pipeline.get("critical_stock_count", 0))
            with col2:
                st.metric("Avg Margin", f"{pipeline.get('avg_margin_pct', 0):.1f}%")
                st.metric("⭐ Avg Rating", f"{pipeline.get('avg_rating', 0):.2f}")
        else:
            st.caption("Loading catalog data...")

        st.divider()

        # ── Clear Chat Button ─────────────────────────────────────────────────
        if st.button("🔄 Clear Chat & Re-brief", use_container_width=True, type="secondary"):
            # Reset memory and agent so Daily Briefing re-triggers
            for key in ["chat_messages", "langchain_messages", "agents_initialized",
                        "chat_agent", "orchestrator", "briefing_injected"]:
                st.session_state.pop(key, None)
            st.rerun()

        st.divider()
        st.caption("RetailMind Analytics · StyleCraft Demo · v1.0")

    return category_filter


# ── Stage 1: Load Data (no API key needed) ───────────────────────────────────
def stage1_load_data():
    """Load CSVs and run analytics pipeline. Cached — runs once per process."""
    if "data_loaded" not in st.session_state:
        with st.spinner("📊 Loading StyleCraft catalog data..."):
            df_products = cached_load_products()
            df_reviews = cached_load_reviews()
            pipeline = cached_run_pipeline(df_products, df_reviews)

            st.session_state["df_products"] = df_products
            st.session_state["df_reviews"] = df_reviews
            st.session_state["pipeline"] = pipeline
            st.session_state["df_enriched"] = pipeline["df_enriched"]
            st.session_state["data_loaded"] = True


# ── Stage 2: Create Agents (requires API key) ─────────────────────────────────
def stage2_create_agents():
    """Create chat agent + orchestrator. Called only after API key is entered."""
    api_key_set = st.session_state.get("api_key_set", False)
    if not api_key_set or "agents_initialized" in st.session_state:
        return

    api_key = os.environ.get(ENV_VAR, "")

    with st.spinner("🤖 Initialising AI agents..."):
        # ChatRouterAgent for the chat tab
        chat_agent = get_chat_router_agent(api_key)
        st.session_state["chat_agent"] = chat_agent

        # Notebook-style agents for the Orchestrator (dashboard "Run Full Analysis")
        llm = get_llm(api_key)
        inv_agent = get_inventory_agent_notebook(llm)
        bi_agent = get_bi_agent_notebook(llm)
        orchestrator = RetailMindOrchestrator(inv_agent, bi_agent)
        st.session_state["orchestrator"] = orchestrator

        # Generate Daily Briefing and inject as first message
        if "briefing_injected" not in st.session_state:
            df_products = st.session_state["df_products"]
            df_reviews = st.session_state["df_reviews"]
            briefing_text = generate_daily_briefing_text(df_products, df_reviews, api_key)

            if "chat_messages" not in st.session_state:
                st.session_state["chat_messages"] = []
            st.session_state["chat_messages"].insert(
                0, {"role": "assistant", "content": briefing_text}
            )
            st.session_state["briefing_injected"] = True

        st.session_state["agents_initialized"] = True


# ── Main App ──────────────────────────────────────────────────────────────────
def main():
    # Title
    st.title("🛍️ RetailMind — StyleCraft Product Intelligence")
    st.markdown(
        "*AI-powered catalog analysis for D2C brands · "
        "Inventory · Pricing · Reviews · Category Intelligence*"
    )

    # Sidebar (renders category filter + catalog summary + API key)
    category_filter = render_sidebar()

    # Stage 1: Load data (always runs)
    stage1_load_data()

    # Stage 2: Create agents (runs when API key is available)
    stage2_create_agents()

    # Get enriched DataFrame
    df_enriched = st.session_state.get("df_enriched")
    pipeline = st.session_state.get("pipeline", {})

    if df_enriched is None:
        st.error("Failed to load catalog data. Please check that data/ CSV files exist.")
        return

    # ── 4 Main Tabs ────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard",
        "💬 Ask RetailMind",
        "📈 Visual Analytics",
        "🔔 Recommendations",
    ])

    with tab1:
        render_dashboard(pipeline, df_enriched, category_filter)

    with tab2:
        render_chat()

    with tab3:
        render_charts(df_enriched, category_filter)

    with tab4:
        render_recommendations(df_enriched, category_filter)


if __name__ == "__main__":
    main()
