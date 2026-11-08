1. Download all 3 models from Kuntapath/class_model_vbeta3 in the Hugging Face
2. Put all 3 models into the class_model folder
3. On the server, run price_caches_v107.py once with RUN_MODE = "download" (already set to download). 
{v68 is the big download; if the app always uses refresh=True, you can set RUN_FOLDERS = ["price_cache_v43"] and skip it. Then run entry_api_v105.py once so the first real request isn't slow.}


>> Each month, run the price_caches_v107.py again (you can schedule it for the 1st), then restart the app.