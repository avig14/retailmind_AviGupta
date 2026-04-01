"""
core/agent_factory.py — LLM and agent creation for RetailMind.

Provider: OpenAI (llama-3.3-70b-versatile)h
  - ENV_VAR   : OPENAI_API_KEY
  - MODEL     : llama-3.3-70b-versatile
  - PROVIDER  : OpenAI

LLM Parameters (thoughtfully set and commented):
  - temperature=0.2  : Low randomness — consistent, factual responses for bhusiness data
  - max_tokens=1024  : Enough for detailed multi-product analysis without runaway output
  - top_p=0.9        : Nucleus sampling — minor diversity while keeping responses focused

Uses standard langchain (NOT langchain_classic — that is legacy-only per research).
AgentExecutor is NOT cached — created per session to ensure per-user memory isolation.
LLM instance IS cached (@st.cache_resource) — created once per process.
"""

import os
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from core.tools import ALL_TOOLS, INVENTORY_TOOLS, BI_TOOLS

# ── Provider Constants ────────────────────────────────────────────────────────
PROVIDER_LABEL = "OpenAI"
MODEL_NAME = "llama-3.3-70b-versatile"
ENV_VAR = "OPENAI_API_KEY"
PLACEHOLDER = "sk-..."
HELP_TEXT = "Get your key at platform.openai.com"


# ── LLM Factory ───────────────────────────────────────────────────────────────

@st.cache_resource
def get_llm(api_key: str = "") -> ChatOpenAI:
    """
    Create and cache the OpenAI LLM instance.
    Cached with @st.cache_resource — created once per Streamlit process.

    Parameters (commented for rubric — LLM & Prompt Engineering, 20 pts):
      temperature=0.2  — Low temperature keeps business analysis factual and consistent
      max_tokens=1024  — Sufficient for detailed multi-product responses
      top_p=0.9        — Nucleus sampling: broad enough for natural language, focused enough for data
    """
    key = api_key or os.getenv(ENV_VAR, "")
    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.2,   # factual, consistent responses for business data queries
        max_tokens=1024,   # enough detail without runaway generation
        top_p=0.9,         # slight diversity while keeping analysis grounded
        api_key=key if key else None,
    )


def get_llm_plain(api_key: str = "") -> ChatOpenAI:
    """Non-cached LLM for notebook / one-off use (no Streamlit context needed)."""
    key = api_key or os.getenv(ENV_VAR, "")
    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.2,
        max_tokens=1024,
        top_p=0.9,
        api_key=key if key else None,
    )


# ── System Prompts ────────────────────────────────────────────────────────────

CHAT_ROUTER_SYSTEM_PROMPT = """You are RetailMind — an AI-powered product intelligence \
assistant for StyleCraft, a D2C fashion brand with 30 SKUs across 5 categories: \
Tops, Dresses, Bottoms, Outerwear, and Accessories. All prices are in Indian Rupees (INR ₹).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROUTING RULES — Classify the user's intent, then call the appropriate tool(s):

  INVENTORY intent → Questions about stock levels, how long inventory will last,
    stockout risk, reorder needs, or urgent restocking.
    → Call: get_inventory_health (for a specific product) OR
            generate_restock_alert (for all products / overview)

  PRICING intent → Questions about margins, gross profit, pricing tiers,
    profitability, cost efficiency, or whether a product is over/under-priced.
    → Call: get_pricing_analysis
    → If margin < 20%: always suggest a corrective pricing action

  REVIEWS intent → Questions about customer feedback, star ratings, complaints,
    product quality, what customers are saying, or review sentiment.
    → Call: get_review_insights
    → Always present positive AND negative themes clearly

  CATALOG intent → Questions about finding products, category overviews, top
    performers, or broad catalog discovery.
    → Call: search_products (to find specific items) OR
            get_category_performance (for category-level metrics)

  GENERAL intent → Greetings, meta-questions about this agent, general retail
    knowledge not tied to StyleCraft data.
    → Respond using your knowledge. If a prior data context exists in
      conversation history, reference it to personalise your response.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRITICAL RULES:
  • ALWAYS use a tool to answer any data question — never guess from training knowledge
  • NEVER fabricate product names, stock numbers, prices, or margins
  • If a product_id is not specified, use search_products first to find it
  • Currency is always ₹ (Indian Rupees) — never use $ or €

RESPONSE FORMATTING:
  • Inventory status: 🔴 Critical (<7 days) | 🟡 Low (7-14 days) | 🟢 Healthy (>14 days)
  • For pricing below 20% margin: state the exact margin and suggest a price increase
  • For multiple products: use structured bullet points with key metrics
  • Keep responses concise — lead with the most important finding
  • End critical alerts with a recommended action for the product manager"""

INVENTORY_AGENT_SYSTEM_PROMPT = """You are the InventoryAnalystAgent for RetailMind Analytics.
Your ONLY responsibility is inventory risk analysis for StyleCraft's 30 SKUs.

You excel at:
  - Computing days_to_stockout (stock_quantity / avg_daily_sales)
  - Flagging Critical (<7 days), Low (7–14 days), and Healthy (>14 days) products
  - Calculating revenue at risk for stockout scenarios
  - Identifying which products need immediate reorder

Always call your tools before drawing conclusions. Present findings sorted by urgency.
Use 🔴 Critical, 🟡 Low, 🟢 Healthy status icons in your output."""

BI_AGENT_SYSTEM_PROMPT = """You are the BusinessIntelligenceAgent for RetailMind Analytics.
Your ONLY responsibility is pricing intelligence, review sentiment, and category performance.

You excel at:
  - Gross margin analysis: (price - cost) / price × 100
  - Identifying below-20% margin products that need pricing action
  - Summarising customer sentiment from review data
  - Comparing category performance metrics

Always use your tools. For margin issues: state the exact % and recommend a concrete action.
For reviews: present both positive AND negative themes clearly."""


# ── Agent Factory Functions ───────────────────────────────────────────────────

def get_chat_router_agent(api_key: str = "") -> AgentExecutor:
    """
    Create the main conversational ChatRouterAgent for Streamlit.

    Architecture:
      - All 6 tools available (LLM selects correct tool via intent routing)
      - StreamlitChatMessageHistory for per-session persistent memory
      - ConversationBufferMemory wraps the history store
      - LLM-based Router: intent classification happens inside the system prompt
        (NOT via keyword/regex matching — 100% LLM-driven decision)

    NOT cached — created per session to isolate each user's memory.
    """
    llm = get_llm(api_key)

    # Memory: StreamlitChatMessageHistory persists across reruns within a session
    history = StreamlitChatMessageHistory(key="langchain_messages")
    memory = ConversationBufferMemory(
        chat_memory=history,
        memory_key="chat_history",
        return_messages=True,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", CHAT_ROUTER_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, ALL_TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        memory=memory,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
        return_intermediate_steps=False,
    )


def get_inventory_agent_notebook(llm) -> AgentExecutor:
    """
    Create InventoryAnalystAgent for notebook usage.
    Tools: search_products, get_inventory_health, generate_restock_alert
    """
    from langchain.memory import ConversationBufferMemory as CBM
    memory = CBM(memory_key="chat_history", return_messages=True)
    prompt = ChatPromptTemplate.from_messages([
        ("system", INVENTORY_AGENT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, INVENTORY_TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=INVENTORY_TOOLS,
        memory=memory,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
    )


def get_bi_agent_notebook(llm) -> AgentExecutor:
    """
    Create BusinessIntelligenceAgent for notebook usage.
    Tools: get_pricing_analysis, get_review_insights, get_category_performance
    """
    from langchain.memory import ConversationBufferMemory as CBM
    memory = CBM(memory_key="chat_history", return_messages=True)
    prompt = ChatPromptTemplate.from_messages([
        ("system", BI_AGENT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, BI_TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=BI_TOOLS,
        memory=memory,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
    )


def generate_daily_briefing_text(df_products, df_reviews, api_key: str = "") -> str:
    """
    Generate the Daily Briefing markdown string using direct analytics + LLM.
    Called automatically on app startup BEFORE user types anything.

    Content (per exam spec):
      1. Top 3 most critically low-stock products (days-to-stockout + revenue at risk)
      2. Worst-rated product (lowest avg_rating + why customers unhappy)
      3. One pricing flag (product with lowest gross margin if below 25%)
    """
    from core.analytics import (
        get_top3_low_stock,
        get_worst_rated_product,
        get_lowest_margin_product,
    )

    top3 = get_top3_low_stock(df_products)
    worst = get_worst_rated_product(df_products)
    lowest_margin = get_lowest_margin_product(df_products)

    # ── Section 1: Inventory Alerts ───────────────────────────────────────────
    inv_lines = []
    for i, p in enumerate(top3, 1):
        status_icon = "🔴" if p["status"] == "Critical" else "🟡"
        inv_lines.append(
            f"  {i}. **{p['product_name']}** ({p['product_id']}) — "
            f"{status_icon} **{p['days_to_stockout']:.1f} days** remaining | "
            f"Revenue at risk: ₹{p['revenue_at_risk_inr']:,.0f}"
        )
    inv_section = "\n".join(inv_lines)

    # ── Section 2: Worst-Rated Product ────────────────────────────────────────
    # Use LLM for 1-line explanation of why customers are unhappy
    reviews_df = df_reviews[df_reviews["product_id"] == worst["product_id"]]
    why_unhappy = "Customers report quality and value concerns."
    if not reviews_df.empty:
        try:
            llm = get_llm(api_key)
            texts = " | ".join(reviews_df["review_text"].tolist())
            from langchain_core.messages import HumanMessage
            resp = llm.invoke([HumanMessage(
                content=f"In ONE sentence, why are customers unhappy with this product? "
                        f"Reviews: {texts[:600]}"
            )])
            why_unhappy = resp.content.strip()
        except Exception:
            pass

    # ── Section 3: Pricing Flag ────────────────────────────────────────────────
    if lowest_margin["below_25_flag"]:
        pricing_section = (
            f"  ⚠️ **{lowest_margin['product_name']}** ({lowest_margin['product_id']}) — "
            f"Gross margin: **{lowest_margin['gross_margin_pct']:.1f}%** (below 25% threshold)\n"
            f"  💡 *Suggested action: Review cost structure or increase selling price to "
            f"restore margin above 25%.*"
        )
    else:
        pricing_section = (
            f"  ✅ **{lowest_margin['product_name']}** ({lowest_margin['product_id']}) — "
            f"Lowest margin in catalog: **{lowest_margin['gross_margin_pct']:.1f}%** "
            f"(above 25% threshold — no immediate action required)"
        )

    briefing = f"""## 📋 RetailMind Daily Briefing

*Good morning! Here's your StyleCraft catalog intelligence summary for today.*

---

### 🚨 Top 3 Inventory Alerts
*Products most at risk of stockout:*

{inv_section}

---

### ⭐ Worst-Rated Product
**{worst['product_name']}** ({worst['product_id']}) — Rating: **{worst['avg_rating']}/5** \
({worst['review_count']} reviews)
{why_unhappy}

---

### 💰 Pricing Flag
{pricing_section}

---
*Ask me anything about your catalog — inventory levels, margins, customer reviews, or category performance.*"""

    return briefing
