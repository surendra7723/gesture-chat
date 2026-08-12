# AGENTS.md

## Environment

- Python virtual environment: `.venv`
- Run commands with: `.venv/bin/python manage.py ...` or `.venv/bin/python -m ruff ...`

## Validation Commands

| Check          | Command                                       |
|----------------|-----------------------------------------------|
| Django check    | `.venv/bin/python manage.py check`            |
| Tests           | `.venv/bin/python manage.py test`              |
| Lint (ruff)     | `.venv/bin/python -m ruff check`              |
| Lint fix (auto) | `.venv/bin/python -m ruff check --fix`        |
