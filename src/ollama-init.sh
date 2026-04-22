#!/bin/bash

ollama serve &
OLLAMA_PID=$!

echo "Waiting for Ollama to start..."
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 2
done
echo "Ollama is ready"

# Only pull if not already downloaded
if ! ollama list | grep -q "llama3"; then
    echo "Pulling the base model..."
    ollama pull llama3:8b-instruct-q4_K_M
else
    echo "Base model already exists, skipping..."
fi

# Download adapters only if they don't exist yet
if [ ! -f /app/model_adapter/short_stock_analyst_adapter.gguf ]; then
    echo "Downloading the adapters from huggingface"
    hf download Kuntapath/stock_analyst_adapter \
        --local-dir /app/model_adapter \
        --local-dir-use-symlinks False
else
    echo "Adapters already exist, skipping download..."
fi


echo "Download the short model----------------------"
ollama create stock-short -f /app/Modelfile.short

echo "Download the mid model----------------------"
ollama create stock-mid -f /app/Modelfile.mid

echo "Download the long model----------------------"
ollama create stock-long -f /app/Modelfile.long

echo "Adapters are downloaded-----------------------"

# Keep ollama running
wait $OLLAMA_PID