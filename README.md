# Semantic search project

Проект сравнивает lexical TF-IDF, semantic e5-small и гибридный поиск товаров по короткому запросу.

## Запуск ноутбука

1. Откройте `АКК semantic_search_project_reviewed_v1_e5.ipynb` в Colab.
2. Заполните доступ к S3 и выполните ячейки сверху вниз.
3. Дождитесь загрузки каталога, разметки и модели e5-small.
4. В блоке недели 3 проверьте таблицу `val_summary`.
5. Лучшая конфигурация выбирается по `NDCG@10` на val.
6. Выполните ячейку финального теста. Она выводит `test_summary` для лучшей модели и semantic baseline.

## MLflow

Код MLflow подготовлен, но запуск учебного сервера временно не выполнен из-за внешней ошибки `503 nginx`. Это не влияет на офлайн-эксперименты и финальные метрики. После восстановления сервера можно выполнить MLflow-ячейку и проверить runs.

1. Запустите MLflow через учебную кнопку «Проверить».
2. Добавьте параметры подключения в `.env`:

```text
MLFLOW_TRACKING_URI=http://mlflow.example.com:5000
MLFLOW_TRACKING_USERNAME=student
MLFLOW_TRACKING_PASSWORD=secret
```

3. Перезапустите ячейку трекинга в ноутбуке. Она создаст эксперимент `semantic-search-project` и отдельный run для каждого варианта.
4. Проверьте записи:

```bash
python check_mlflow.py semantic-search-project --env-file .env
```

В каждом run должны быть параметры `method`, `eval_split`, `catalog_size`, `relevance_threshold` и метрики `ndcg_at_10`, `recall_at_10`, `precision_at_5`, `mrr`.

## Артефакты

- `val_summary` - сравнение экспериментов на validation;
- `test_summary` - один финальный прогон на test;
- `artifacts/mlflow/` - таблицы метрик и golden-сеты для MLflow.
