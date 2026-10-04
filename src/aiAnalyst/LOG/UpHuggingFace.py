from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="./src/aiAnalyst/model_adapter",      
    repo_id="Kuntapath/stock_analyst_adapter",  
    repo_type="model",
    allow_patterns=["quant_stock_adapter_v2_1.gguf", "quant_stock_adapter_v2_2.gguf"], 
)
print("✅ Upload Success!")