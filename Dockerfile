# Étape 1 : Image de base
FROM python:3.12.3-slim

# Étape 2 : Dossier de travail
WORKDIR /app

# Étape 3 : Dépendances système nécessaires (PostgreSQL client, etc.)
RUN apt-get update && apt-get install -y libpq-dev gcc && apt-get clean

# Étape 4 : Installer Poetry
RUN pip install --no-cache-dir poetry

# Étape 5 : Copier fichiers de dépendances
COPY pyproject.toml poetry.lock README.md ./

# Étape 6 : Installer les dépendances sans dev
RUN poetry install --without dev --no-root

# Étape 7 : Copier le code source
COPY . .

# Étape 8 : Exposer le port (FastAPI Uvicorn port)
EXPOSE 8008

# Étape 9 : Lancer l'application
CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8006"]
