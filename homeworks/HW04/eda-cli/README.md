@"
# EDA CLI HTTP API (HW04)

## Запуск
```bash
uv sync
uv run uvicorn eda_cli.api:app --reload --port 8000
POST /quality-flags-from-csv