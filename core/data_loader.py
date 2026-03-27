"""
core/data_loader.py — Cached data loading functions for Streamlit.

Uses @st.cache_data so CSVs are loaded once per process and reused
across reruns. Pure data loading — no analytics or agent logic here.
"""

import streamlit as st
import pandas as pd

from core.analytics import load_products, load_reviews, run_full_pipeline


@st.cache_data(show_spinner=False)
def cached_load_products(path: str = "data/retailmind_products.csv") -> pd.DataFrame:
    """Load and cache the products DataFrame. Refreshes if path changes."""
    return load_products(path)


@st.cache_data(show_spinner=False)
def cached_load_reviews(path: str = "data/retailmind_reviews.csv") -> pd.DataFrame:
    """Load and cache the reviews DataFrame. Refreshes if path changes."""
    return load_reviews(path)


@st.cache_data(show_spinner=False)
def cached_run_pipeline(_df_products: pd.DataFrame, _df_reviews: pd.DataFrame) -> dict:
    """
    Run full analytics pipeline and cache results.
    Leading underscore on params tells Streamlit not to hash DataFrames.
    Returns dict with all KPIs, enriched DataFrame, and top-10 priority list.
    """
    return run_full_pipeline(_df_products, _df_reviews)
