# Ячейка в конце ноутбука: сохраняет артефакты для приложения
APP_DIR = Path("artifacts/app")
APP_DIR.mkdir(parents=True, exist_ok=True)

# 1. Модель
emb_model.save(str(APP_DIR / "model"))

# 2. Каталог (тот же eval_df, на котором считали метрики)
app_catalog = eval_df[
    ["imt_id", "imt_name", "subj_name", "brand_name", "description", "product_text"]
]
app_catalog.to_parquet(APP_DIR / "catalog.parquet", index=False)

# 3. Эмбеддинги каталога в том же порядке строк
emb = load_embeddings(eval_df, EMB_PATH).astype("float32")
assert emb.shape[0] == len(app_catalog)
np.save(APP_DIR / "embeddings.npy", emb)
