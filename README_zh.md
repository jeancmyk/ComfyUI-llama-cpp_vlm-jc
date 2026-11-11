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

## OpenAI 兼容接口（由 ComfyUI 自身提供）

ComfyUI 会在自己的端口上同时提供 OpenAI 兼容接口。它随 ComfyUI 启动/停止，
使用**与节点相同的那个常驻 llama.cpp 模型**，不需要额外进程——因此 VLM 推理和
扩散模型不会各持一份相同的权重占用显存。

```
http://127.0.0.1:8188/v1
```

| 接口 | 用途 |
| --- | --- |
| `GET /v1/models` | 已注册模型列表，含 `context_length`、`architecture.input_modalities`、`capabilities` 与实时 `status` |
| `GET /v1/models/{id}` | 单个模型的元数据 |
| `POST /v1/chat/completions` | 对话，支持 `image_url`（视觉）与 `input_audio` 内容块；`stream: true` 为 SSE |
| `POST /v1/completions` | 纯文本补全 |
| `POST /v1/embeddings` | 向量化（上下文会热切换为 embeddings 模式再切回） |
| `POST /v1/unload` | 释放常驻模型 |
| `GET/POST/DELETE /v1/sessions…` | 保存的会话 |
| `GET /llama_cpp/api/status` | 当前模型、模型数量、空闲显存 |
| `GET/PUT /llama_cpp/api/models` | 读取或整体替换注册表 |

### 在 ComfyUI 中配置模型

添加 **Llama-cpp API Model** 节点并设置其参数（checkpoint、`mmproj`、
`chat_handler`、`n_ctx`、KV 缓存类型、`vram_limit`、采样默认值）。运行该节点会把
一条配置写入本目录的 `api_models.json`，接口在下次请求时即生效，无需重启。
`/v1/models` 只列出你注册的模型，而不是目录下的全部文件。

返回的元数据是推导出来的而非猜测：带 `mmproj` 且 `chat_handler` 匹配的条目会报告
`input_modalities: [text, image]` 与 `capabilities.vision: true`；ASR 类 handler
（Qwen3-ASR）报告音频能力；纯文本条目会被当作纯文本强制执行（图像输入直接返回
400，而不是在 llama.cpp 深处失败）。

注册表为空时，`auto_register` 会把 `models/LLM` 下所有可独立服务的 `.gguf` 以保守默认值
列出（标记为 `meta.comfyui.auto: true`，且不声明视觉/音频能力），避免新装用户面对
空列表；safetensors 分片、MTP/draft 附属文件与 mmproj 会被跳过。一旦注册了真实模型，
这些条目就会消失。

`GET /llama_cpp/api/models` 有意区分两者：`models` 是文件里真正保存的条目，
`effective_models` 是接口当前实际对外提供的列表（注册表为空时包含自动条目）。`PUT`
请求体带显式 `models` 对象时**整体替换**，因此编辑器可以安全地读出、修改再写回，
而不会把自动推断的条目变成正式配置；只带 `{id: entry}` 的请求体则只更新这些 id。

没有 pooling 类型的 checkpoint（即所有因果聊天模型）会让 llama.cpp 返回逐 token 的
向量；`/v1/embeddings` 会把它们均值池化成每个输入一个向量（OpenAI 客户端的预期），
真正的向量模型（Qwen3-Embedding 等）则原样返回。

### 在节点里编辑整个注册表

**Llama-cpp API Registry** 节点把整份 `api_models.json`（`_settings` 与全部
模型）放进一个多行文本框，让你无需离开 ComfyUI 就能集中调整安全余量、思考
模式、api key 与所有模型条目：

- 添加节点时文本框会**自动从文件填充**，并有 **Load from server** 按钮在任何
  改动后（手改、HTTP 写入、或另一个标签页）刷新；
- 以 `action=save` 运行节点写回文本框，`action=load` 把文件拉进文本框，
  `validate_only` 则只校验不写盘；
- 写回路径对每个条目走与 HTTP 编辑器相同的校验，因此写错的键会**点名该条目**
  被拒绝，且不写入任何内容；
- `api_key` 显示为 `***` 且永不回显真实值——保留即沿用已存的密钥，输入新值即
  修改，清空即关闭鉴权。

这些旋钮（`vram_margin`、`vram_reserve_gb`、`auto_fit_ctx`、`reasoning` 等）
是即时的：接口在下次请求时重读文件，无需重启。

### 在触碰 llama.cpp 之前先核对加载规划

过度分配的 CUDA 显存会**中止整个 ComfyUI 进程**（`Fatal Python error`，
任何异常处理都接不住），因此请求绝不会带着放不下的规划进入 llama.cpp。
`support/vram_plan.py` 用文件大小和 GGUF 元数据（与 loader 打印的同一套输入）
估算 checkpoint、projector 与 KV 缓存，然后：

- **下调 n_ctx** 到空闲显存能容纳的最大值（预留计算缓冲并留 8% 余量），
  这样 ComfyUI 占着扩散权重时到达的请求也能得到回答而不是中止。下调会
  写进日志，`/v1/models` 报告实际服务的上下文，且常驻模型**不会**每个请求
  都重新加载；
- 连权重都放不下时**拒绝**，返回 `400 model_wont_fit` 会写明数字与对策
  （开 `free_vram` 让 ComfyUI 先卸载扩散模型，或用 `vram_limit` +
  `enforce_vram` 让 llama.cpp 把层放到内存）；
- `vram_limit` + `enforce_vram` 交给 loader 自行处理（它会干净地卸层或拒绝）。

在 `api_models.json` 里设置 `_settings.auto_fit_ctx: false` 可让请求直接
报错说明，而不是服务一个更小的上下文。

下调保留两项安全余量，都是可配置的：`vram_margin`（默认 `0.92`，即规划
最多使用报告空闲显存的 92%）与 `vram_reserve_gb`（默认 `2.0`，为 CUDA
上下文与 llama.cpp 计算缓冲预留——这部分**不在**权重+KV 之内，40 层模型
实测约 2 GB）。32 GB 显卡上默认合计约留 4.4 GB；放宽可按比例换回上下文，
但请先用节点的 `test_load` 实测验证——过度分配的 CUDA 显存会**中止整个
ComfyUI 进程**，而不是抛出异常。

**Llama-cpp API Model** 节点的 `test_load` 开关就是为此准备的：它会执行
条目自身的 `free_vram` 清理，在生成锁下真实加载模型，报告 loader 实测
数据（权重、state/KV、加载前后空闲显存），并保持常驻让首个 API 请求
即热。加载被拒绝或失败时，结果会写进节点的 `summary` 输出。

### 与扩散模型的显存协同

两个方向都已处理：

- 接口按条目的 `free_vram` 在加载前调用 ComfyUI 自己的 `free_memory`，只在确实
  需要空间时才卸载扩散模型；
- 在 `comfy.model_management.free_memory` 上安装了仲裁钩子：当**扩散**加载显存
  不足时释放常驻的 llama.cpp 模型。否则 ComfyUI 只能卸载它自己注册过的模型，
  扩散加载会因为显存不足而失败。

正在进行的生成不会被中断：接口与图节点共享同一把锁，仲裁钩子在锁被占用时跳过卸载。

### 会话状态

在 JSON body 中传入 `session_id`（或使用 `X-Session-Id` 头），接口会把已保存的对话
历史拼在你的消息之前，并把本轮追加到 `api_sessions/<id>.json`。状态在 ComfyUI
重启后依然存在，客户端可以隔几天再续聊。图片只保存一次：内联 base64 会转存为旁路
文件，重放时再还原，因此会话文件很小；只保存回答，不保存模型的思考内容。
`GET /v1/sessions` 列出会话，`DELETE` 删除，`POST /v1/sessions/{id}/clear` 清空历史。

存储上限由 `api_models.json` 中的 `_settings.max_sessions` 与
`_settings.session_ttl_days` 控制；请求里设置 `store: false` 可完全不写入会话。

### 推理（thinking）模型

默认情况下接口会识别当前模型的思考标记（与 Instruct 节点使用同一套预设），把回答
放在 `content`，把思考内容放在独立的 `reasoning_content` 字段（流式与非流式一致）。
请求中设置 `"reasoning": "keep"`（或注册表中的 `_settings.reasoning`）可原样输出，
`"strip"` 则完全丢弃思考内容；`include_reasoning: false` 在流式响应中省略它。

> 思考模型会先用掉 token 预算：`max_tokens` 太小时可能思考还没结束、回答为空。
> 给它几百个 token，或改用非 thinking 的 chat handler。

### 安全

未设置 `_settings.api_key` 时接口无鉴权；设置后所有路由都需要
`Authorization: Bearer <key>`（或 `X-Api-Key`）。ComfyUI 默认只绑定本机地址，
若对外暴露请务必设置密钥。

运行 `python tests/test_openai_api.py` 可做自检（加 `--live` 会加载一个 GGUF 并
实测生成与流式输出）。

## 致谢
- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @JamePeng  
- [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) @kijai
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
