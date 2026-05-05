#!/bin/bash

ollama serve &
OLLAMA_PID=$!

echo "Waiting for Ollama to start..."
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 2
done
echo "Ollama is ready"

# Only pull if not already downloaded
if ! ollama list | grep -q "llama3.1"; then
    echo "Pulling the base model..."
    ollama pull llama3.1
else
    echo "Base model already exists, skipping..."
fi

# Download adapters only if they don't exist yet
if [ ! -f /app/model_adapter/quant_stock_adapter_v2_2.gguf ]; then
    echo "Downloading the adapters from huggingface"
    hf download Kuntapath/stock_analyst_adapter \
        --local-dir /app/model_adapter
else
    echo "Adapters already exist, skipping download..."
fi

echo "Dowload the adapter-stock-quantV2.2--------------------------"


ollama create stock-quant-v2-2 -f /app/Modelfile.quantv22

echo "Adapters are downloaded-----------------------"

# Keep ollama running
wait $OLLAMA_PID