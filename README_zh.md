# ComfyUI-llama-cpp
在 ComfyUI 中基于 llama.cpp 框架原生运行 LLM & VLM 模型。  
**[[📃English](./README.md)]**   

## 预览
![](./img/preview.jpg) 

## 安装步骤

#### 安装节点:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lihaoyun6/ComfyUI-llama-cpp.git
python -m pip install -r ComfyUI-llama-cpp/requirements.txt
#CUDA 13用户请执行:
#python -m pip install -r ComfyUI-llama-cpp/requirements_cu131.txt
```

### 模型路径:
- 请将下载的 `.gguf` 模型放置在 `ComfyUI/models/LLM` 目录中.  

	> 在使用VLM模型进行图像推理之前, 请确保已经下载并选择了主模型对应的`mmproj`权重文件.

## 控制台遥测与显存控制

**Llama-cpp Model Loader** 在加载前会向 ComfyUI 控制台输出一份显存规划：
checkpoint / mmproj 大小、按当前 `n_ctx` 与 KV 缓存量化方式估算的 KV 缓存大小、
相对 `vram_limit` 的 `FITS` / `OVER` 结果、MTP 是否请求并已激活，以及加载完成后
llama.cpp 报告的实际模型 / 状态大小。

**Llama-cpp Instruct** 在生成过程中每 2 秒输出一行实时数据（已生成 token 数、
累计平均 tok/s、prefill 进度），并在每次生成结束后输出 prefill / decode 汇总。

Loader 选项：
- `free_vram`（默认开启）：加载前请求 ComfyUI 仅卸载满足“checkpoint + KV 估算”
  所需的模型，并清理 CUDA 缓存。
- `enforce_vram`（需 `vram_limit != -1`）：通过把多余层放到内存来保证
  checkpoint + KV + mmproj 不超过上限；若连一个 GPU 层都放不下则拒绝加载。
- `skip_mmproj`（默认关闭）：保留已选择的 projector 配置但不加载它（纯文本运行）。
  开启期间图像/音频输入会被拒绝。

## 致谢
- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @JamePeng  
- [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) @kijai
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
