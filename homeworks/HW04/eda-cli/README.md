# EDA CLI HTTP API (HW04)

## Запуск

Убедитесь, что установлен `uv` (https://docs.astral.sh/uv/).

В корне папки `eda-cli` выполните:

```bash
uv sync
uv run uvicorn eda_cli.api:app --reload --host 0.0.0.0 --port 8000
```
API будет доступно по адресу: http://127.0.0.1:8000  
Документация (Swagger UI): http://127.0.0.1:8000/docs

## Дополнительный эндпоинт

Реализован эндпоинт:
```
POST /quality-flags-from-csv
```