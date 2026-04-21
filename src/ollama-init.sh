#!/bin/bash

ollama serve &

echo "waiting for the ollama to run"
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 2;
done
echo "Ollama is ready"

echo "Pulling the base mode"
ollama pull llama3:8b-instruct-q4_K_M

echo "Downloading the adapyers from huggunface"
huggunface-cli download Kuntapath/stock_analyst_adapter \
    --local-dir /app/model_adapter \
    --local-dir-use-symlinks False
echo "Base model is downloaded"


echo "Download the short model----------------------"
ollama create stock-short -f /app/Modelfile.short

echo "Download the mid model----------------------"
ollama create stock-mid -f /app/Modelfile.mid

echo "Download the long model----------------------"
ollama create stock-long -f /app/Modelfile.long

echo "Adapters are downloaded-----------------------"


wait