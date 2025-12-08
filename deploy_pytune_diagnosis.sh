#!/bin/bash
set -e

SERVICE_NAME="pytune_diagnosis"
SERVER="gabriel@195.201.9.184"

# Position du script → dossier pytune_diagnosis
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Racine du monorepo = 3 niveaux au-dessus
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../" && pwd)"

REMOTE_DIR="/home/gabriel/deploy/$SERVICE_NAME"
LOGS_DIR="/home/gabriel/pytune_logs/$SERVICE_NAME"

echo "📁 Repo root: $REPO_ROOT"
cd "$REPO_ROOT"

echo "🚀 Deploying $SERVICE_NAME to server…"

echo "📦 Building Docker image (AMD64)..."
docker build \
    --platform linux/amd64 \
    -t $SERVICE_NAME:latest \
    -f src/services/$SERVICE_NAME/Dockerfile \
    .

echo "📦 Saving image to tarball..."
docker save $SERVICE_NAME:latest | gzip > $SERVICE_NAME.tar.gz

echo "📤 Uploading to server..."
ssh $SERVER "mkdir -p $REMOTE_DIR"
scp $SERVICE_NAME.tar.gz $SERVER:$REMOTE_DIR/

echo "🔧 Deploying on server..."
ssh $SERVER << EOF
set -e

cd $REMOTE_DIR

echo "📦 Loading Docker image…"
gunzip -f $SERVICE_NAME.tar.gz
docker load < $SERVICE_NAME.tar

echo "🛑 Stopping old container…"
docker stop $SERVICE_NAME || true
docker rm $SERVICE_NAME || true

echo "📁 Ensuring logs directory…"
mkdir -p $LOGS_DIR

echo "🚀 Starting new container…"
docker run -d \
    --name $SERVICE_NAME \
    --restart always \
    --network pytune_network \
    -v $LOGS_DIR:/var/log/pytune \
    -e LOG_DIR="/var/log/pytune" \
    --env-file /home/gabriel/pytune.env \
    -p 8008:8008 \
    $SERVICE_NAME:latest

rm -f $SERVICE_NAME.tar
EOF

echo "🎉 $SERVICE_NAME deployed successfully!"