# RetailMind AI Agent — StyleCraft Product Intelligence

An AI-powered product intelligence agent for **StyleCraft**, a D2C fashion brand with 30 SKUs across 5 categories. Built for **RetailMind Analytics** to replace the Product Manager's 4–5 hour weekly manual analysis with a real-time conversational interface.

---

## Quick Start

```bash
# 1. Clone / download the repo
# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenAI API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 4. Launch the app
python run.py
```

Open **http://localhost:8501** in your browser.

---

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI API key (get at platform.openai.com) |

See `.env.example` for the template.

---

## Project Structure

```
├── run.py                    ← Entry point: python run.py
├── app.py                    ← Streamlit app
├── core/
│   ├── analytics.py          ← All business formulas (pure Python)
│   ├── tools.py              ← 6 LangChain @tool functions
│   ├── data_loader.py        ← Cached CSV loaders
│   └── agent_factory.py      ← LLM + agent creation
├── ui/
│   ├── dashboard.py          ← Tab 1: KPIs + alerts + analysis
│   ├── chat.py               ← Tab 2: Conversational AI
│   ├── charts.py             ← Tab 3: 5 Plotly charts
│   └── recommendations.py   ← Tab 4: Restock table + download
├── data/
│   ├── retailmind_products.csv
│   └── retailmind_reviews.csv
├── retailmind_agent.ipynb    ← Notebook: full multi-agent pipeline
├── .env.example
└── requirements.txt
```

---

## Agentic Architecture

### 1. Tool-Calling Layer (6 Tools)

| Tool | Intent Route | Description |
|---|---|---|
| `search_products` | CATALOG | Find products by text query + optional category filter |
| `get_inventory_health` | INVENTORY | Stock, daily sales, days-to-stockout, status flag |
| `generate_restock_alert` | INVENTORY | All at-risk products sorted by urgency with revenue at risk |
| `get_pricing_analysis` | PRICING | Gross margin %, positioning vs category avg, below-20% flag |
| `get_review_insights` | REVIEWS | LLM-generated sentiment summary + positive/negative themes |
| `get_category_performance` | CATALOG | Aggregated category metrics + top 3 revenue generators |

### 2. Router Pattern (LLM-based — NOT keyword/regex)

The **ChatRouterAgent** uses a system prompt to classify user intent into 5 routes:
- **INVENTORY** → `get_inventory_health` or `generate_restock_alert`
- **PRICING** → `get_pricing_analysis`
- **REVIEWS** → `get_review_insights`
- **CATALOG** → `search_products` or `get_category_performance`
- **GENERAL** → LLM knowledge + conversation memory

Intent classification is performed entirely by the LLM — no `if/elif` keyword matching.

### 3. Multi-Agent Orchestrator (Notebook + Dashboard)

- **InventoryAnalystAgent**: Inventory risk analysis (3 tools)
- **BusinessIntelligenceAgent**: Pricing + reviews + category (3 tools)
- **RetailMindOrchestrator**: Runs both sequentially, passing context forward

### 4. Memory

- `StreamlitChatMessageHistory` stores conversation in Streamlit session state
- `ConversationBufferMemory` wraps it for `AgentExecutor`
- Full multi-turn context across the session

---

## Business Formulas

```python
days_to_stockout     = stock_quantity / avg_daily_sales
gross_margin_pct     = (price - cost) / price * 100
revenue_at_risk      = price × (stock_quantity + avg_daily_sales × threshold_days)
price_positioning    = "Premium" if price > cat_avg×1.2 else "Budget" if price < cat_avg×0.8 else "Mid-Range"
```

---

## Key Features

- **Daily Briefing**: Auto-generated on startup — top 3 low-stock, worst-rated, pricing flag
- **4 Tabs**: Dashboard · Chat · Charts · Recommendations
- **5 Interactive Charts**: Price comparison, margin health, inventory risk, velocity, priority matrix
- **CSV Export**: Download filtered recommendations table
- **24h Cache**: Data loaded once, reused across reruns

---

## LLM Parameters

```python
temperature = 0.2   # Low — keeps business analysis consistent and factual
max_tokens  = 1024  # Sufficient for multi-product analysis responses
top_p       = 0.9   # Nucleus sampling — focused but natural language output
```

---

*Built with LangChain · OpenAI gpt-4o-mini · Streamlit · Plotly*
