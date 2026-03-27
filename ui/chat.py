"""
ui/chat.py — Tab 2: Conversational AI chat interface.

Features:
  - Daily Briefing displayed as first message on startup
  - 6 suggested question buttons (2-column grid)
  - StreamlitCallbackHandler shows live tool calls with expand/collapse
  - StreamlitChatMessageHistory for multi-turn memory (persists across reruns)
  - Free-form chat_input for any question
"""

import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler


# Suggested questions covering all 5 routing intents
SUGGESTED_QUESTIONS = [
    "Which products are about to run out of stock?",
    "What is the gross margin for SC004 Bohemian Printed Kurti?",
    "What are customers saying about the Floral Summer Dress?",
    "Show me the top performers in the Accessories category.",
    "Which products have critically low inventory right now?",
    "Find me all Outerwear products and their stock levels.",
]


def render_chat():
    """Render the full chat interface tab."""
    st.subheader("💬 Ask RetailMind")
    st.caption(
        "Powered by OpenAI gpt-4o-mini · LLM-based intent routing · "
        "Tools: Inventory | Pricing | Reviews | Catalog"
    )

    chat_agent = st.session_state.get("chat_agent")
    api_key_set = st.session_state.get("api_key_set", False)

    if not api_key_set:
        st.warning("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.", icon="⚠️")
        _render_chat_history()
        return

    if chat_agent is None:
        st.warning("Agent not initialised — please refresh the page.", icon="⚠️")
        return

    # ── Suggested Questions ────────────────────────────────────────────────────
    st.markdown("**💡 Suggested questions:**")
    cols = st.columns(2)
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        with cols[i % 2]:
            if st.button(q, key=f"suggest_{i}", use_container_width=True):
                _handle_user_input(q, chat_agent)

    st.divider()

    # ── Chat History ───────────────────────────────────────────────────────────
    _render_chat_history()

    # ── Chat Input ─────────────────────────────────────────────────────────────
    user_input = st.chat_input(
        "Ask anything about StyleCraft's catalog — inventory, pricing, reviews...",
        key="chat_input_main",
    )
    if user_input:
        _handle_user_input(user_input, chat_agent)


def _render_chat_history():
    """Display all messages in session state chat history."""
    messages = st.session_state.get("chat_messages", [])
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        with st.chat_message(role):
            st.markdown(content)


def _handle_user_input(user_input: str, chat_agent):
    """Process a user message: display it, invoke agent, display response."""
    # Add user message to display history
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    st.session_state["chat_messages"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Invoke agent with callback handler for live tool display
    with st.chat_message("assistant"):
        cb_container = st.container()
        cb = StreamlitCallbackHandler(
            cb_container,
            expand_new_thoughts=True,
            collapse_completed_thoughts=True,
        )

        try:
            result = chat_agent.invoke(
                {"input": user_input},
                config={"callbacks": [cb]},
            )
            response = result.get("output", "I encountered an issue — please try again.")
        except Exception as e:
            response = f"⚠️ Error: {str(e)}"

        st.markdown(response)
        st.session_state["chat_messages"].append(
            {"role": "assistant", "content": response}
        )
