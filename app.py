import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

MODEL_DIR = "intfloat/multilingual-e5-small"
CATALOG_PATH = Path("catalog.parquet")
EMBEDDINGS_PATH = Path("embeddings.npy")

# Очистка запроса - та же, что в ноутбуке
TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MD_RE = re.compile(r"\*{1,}|_{2,}")
ARTICLE_RE = re.compile(r"\b\d{6,}\b")
SPECIAL_RE = re.compile(r"[^\w\s.,%\-/]")
SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = MD_RE.sub(" ", text)
    text = ARTICLE_RE.sub(" ", text)
    text = SPECIAL_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text)
    return text.strip().lower()


@st.cache_resource
def load_model():
    return SentenceTransformer(str(MODEL_DIR), device="cpu")


@st.cache_data
def load_catalog():
    return pd.read_parquet(CATALOG_PATH)


@st.cache_data
def load_embeddings():
    return np.load(EMBEDDINGS_PATH)


def search(query: str, top_k: int) -> pd.DataFrame:
    model, catalog, emb = load_model(), load_catalog(), load_embeddings()
    query_emb = model.encode(
        ["query: " + clean_text(query)], normalize_embeddings=True
    )[0]
    scores = emb @ query_emb  # векторы нормированы, значит это косинусная близость
    idx = np.argsort(-scores)[:top_k]
    res = catalog.iloc[idx]
    return pd.DataFrame(
        {
            "Название": res["imt_name"].values,
            "Категория": res["subj_name"].values,
            "Бренд": res["brand_name"].values,
            "Скор": scores[idx].round(3),
            "imt_id": res["imt_id"].values,
            "Описание": res["description"].str[:150].values,
        }
    )


st.set_page_config(page_title="Семантический поиск товаров")
st.title("Семантический поиск товаров")

query = st.text_input("Запрос", placeholder="например, кожаные мокасины")
top_k = st.slider("Сколько товаров показать", 1, 20, 10)

if query.strip():
    if not clean_text(query):
        st.warning("После очистки запрос стал пустым, введите слова.")
    else:
        st.dataframe(search(query, top_k), hide_index=True, width="stretch")
