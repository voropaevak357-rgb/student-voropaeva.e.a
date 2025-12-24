# EDA CLI HTTP API (HW04)

## Запуск

Убедитесь, что установлен `uv` (https://docs.astral.sh/uv/).

В корне папки `eda-cli` выполните:

```bash
uv sync
uv run uvicorn eda_cli.api:app --reload --host 0.0.0.0 --port 8000
POST /quality-flags-from-csv