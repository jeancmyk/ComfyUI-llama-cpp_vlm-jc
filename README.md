# ComfyUI-llama-cpp  
Run LLM/VLM models natively in ComfyUI based on llama.cpp  
**[[📃中文版](./README_zh.md)]** 

## Preview  
![](./img/preview.jpg)

## Installation  

#### Install the node:  
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lihaoyun6/ComfyUI-llama-cpp.git
python -m pip install -r ComfyUI-llama-cpp/requirements.txt
#For CUDA 13, please run:
#python -m pip install -r ComfyUI-llama-cpp/requirements_cu131.txt
```

#### Download models:  
- Place your model files in the `ComfyUI/models/LLM` folder.  

	> If you need a VLM model to process image input, don't forget to download the `mmproj` weights.

## Console telemetry & VRAM controls

The **Llama-cpp Model Loader** prints a memory plan to the ComfyUI console before
loading: checkpoint / mmproj sizes, the KV-cache estimate for the configured
`n_ctx` and cache quantizations, `FITS` / `OVER` against `vram_limit`, whether MTP
was requested and is active, and the actual model / state sizes reported by
llama.cpp after the load.

The **Llama-cpp Instruct** node prints a live line every 2 seconds while generating
(generated tokens, cumulative average tok/s, prefill progress), followed by a
per-completion prefill / decode summary.

Loader switches:
- `free_vram` (default on): before loading, ask ComfyUI to unload only as many
  models / clear as much CUDA cache as the estimated checkpoint + KV needs.
- `enforce_vram` (requires `vram_limit != -1`): keep checkpoint + KV + mmproj
  inside the cap by moving extra layers to system RAM; loading is refused if even
  one GPU layer cannot fit.
- `skip_mmproj` (default off): keep the selected projector configured but do not
  load it (text-only run). Image/audio inputs are rejected while it is on.

## Credits  
- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @JamePeng  
- [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) @kijai  
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
