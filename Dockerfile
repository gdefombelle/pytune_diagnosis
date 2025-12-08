# ===============================
# Étape 1 : Build avec UV
# ===============================
FROM --platform=linux/amd64 python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app

# Copier le workspace root (pyproject + lock)
COPY pyproject.toml uv.lock ./

# Copier tout le code (packages + services)
COPY src ./src

# Aller dans le service
WORKDIR /app/src/services/pytune_diagnosis

# Installer dans /app/.venv
RUN uv sync --no-dev


# ===============================
# Étape 2 : Image finale
# ===============================
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ❗ IMPORTANT : ici on doit se placer dans le dossier contenant app/main.py
WORKDIR /app/src/services/pytune_diagnosis

# Copier tout le workspace + la venv
COPY --from=builder /app /app

EXPOSE 8008

# Lancement via la venv globale créée par UV
CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8008"]