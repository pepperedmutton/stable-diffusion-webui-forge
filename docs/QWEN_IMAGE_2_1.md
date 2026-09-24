# Qwen Image 2.1 本地使用说明

在 Forge 顶部的 **Model preset** 选择 **Qwen Image 2.1** 即可。模型、文本编码器、VAE 和内存管理会一起切换，无需手动选择附加模块。原 Forge 环境保持不变，新模型使用独立运行环境。

**手动精度选择**

Qwen 的精度选项提供三档；主模型和文本编码器同时切换，VAE 始终以 BF16 运行。精度选择会保存，刷新页面或切换其他模型后再返回也会保留。模型下拉框中的精度名称与该选项同步。

| 精度 | 用途 | 模型名称 |
| --- | --- | --- |
| NF4，4 位量化（首次迁移默认） | 权重占用最小，适合节省显存和内存 | `Qwen-Image-2.1-NF4` |
| INT8，8 位量化 | 比 4 位保留更高的权重精度，占用也更高 | `Qwen-Image-2.1-INT8` |
| BF16，原始精度 | 官方未量化权重，用于原始效果对照 | `Qwen-Image-2.1-BF16` |

量化只覆盖适用的线性层，嵌入、归一化等层保留浮点精度，因此不能把整个模型大小简单除以 2 或 4。NF4 使用双重量化和 BF16 计算；INT8 使用 bitsandbytes 的 LLM.int8 内核，部分激活会在内核中转换为 FP16。两种量化均在本机从已校验的官方权重转换并保存，切换时直接加载，不会每次重新量化。

完整文件占用约为 **BF16 33.1 GB、INT8 18.5 GB、NF4 11.4 GB**（十进制；量化目录还包含校验清单）。这是磁盘占用，不是推理时的显存需求。量化版的主模型有 232 个量化线性层，文本编码器有 368 个。

INT8 运行时额外处理两项本机实测兼容问题：迁移 CPU/GPU 时同步量化缓存；在线性层计算前规范非连续输入，避免上游 INT8 CUDA 内核误读步长产生异常色块。这些修复只作用于独立 Qwen 运行进程，不改其他模型使用的库。

**生成与编辑**

- 文生图：在 `txt2img` 输入描述并生成。默认 **40 步、CFG 1**；界面固定显示 **Euler / Simple**，实际使用模型自带的 `FlowMatchEulerDiscreteScheduler`。
- 图片编辑：在 `img2img` 上传图片，在提示词中写明修改要求，例如“把背景改成雪山，保留人物的姿势和衣服”。该流程采用文字指令编辑，不使用传统重绘强度。
- 宽高范围为 **256–2048**，需为 **32 的倍数**。兼容尺寸保持不变，其余尺寸自动向下对齐；步数仍可调整。
- 本机 24 GB 显卡建议日常使用 **1024 × 1024**。2K 已验证能够输出，但完整解码接近显存容量，耗时明显更长。保留官方默认的完整 VAE 解码，避免分块产生的细微接缝。
- 结果保存为 **PNG**，保留模型输出的 **RGBA 透明通道**和生成参数。RGBA 不代表每张图片都会自动生成透明背景，仍需在提示词中说明需要的背景效果。
- 当前集成不支持旧 LoRA、文本嵌入、Hypernetwork、ControlNet/扩展脚本、Hires fix、Refiner、负面提示词及传统重绘参数；相应界面会自动隐藏或禁用。切换时会移除提示词中的旧 LoRA 等标签，切回其他模型时恢复原提示词；若使用 Qwen 时已修改正文，则保留新正文。
- 切回 Krea、XL 或 Anima 会恢复其配置和可用控件，并释放空闲的 Qwen 运行进程。

**本地文件与维护**

以下路径相对于 `D:\Forge\stable-diffusion-webui-forge`：

| 内容 | 路径 |
| --- | --- |
| 完整模型、编码器、VAE、处理器 | `models/diffusers/Qwen-Image-2.1/` |
| NF4 量化版 | `models/diffusers/Qwen-Image-2.1-NF4/` |
| INT8 量化版 | `models/diffusers/Qwen-Image-2.1-INT8/` |
| 独立 Python 环境 | `runtimes/qwen-image-2.1/` |
| Forge 接入与运行进程 | `modules_forge/qwen21.py`、`modules_forge/qwen21_worker.py` |
| 环境安装脚本 | `scripts/setup_qwen21_runtime.py` |
| 权重下载、断点续传和校验脚本 | `scripts/download_qwen21.py` |
| 本地量化转换脚本 | `scripts/quantize_qwen21.py` |
| 固定版本的官方文件清单 | `tmp/qwen21-install/hub-manifest.json` |
| 已完成的逐文件校验报告 | `tmp/qwen21-install/verified-model.json` |
| 下载和校验日志 | `tmp/qwen21-install/range-download.jsonl` |

三档精度都采用 **CPU 卸载**，为完整 VAE 解码留出显存。CPU 卸载本身不是量化。运行时复用 Forge 已安装的 CUDA PyTorch；新版 Diffusers、Transformers 和 bitsandbytes 0.50.2 等依赖只安装在独立环境中。新运行时使用固定 Diffusers 提交 `8b3c707ebd3ec4881f4190cf42931da07eaf3b65`。

如需重建环境或补齐下载，在 Forge 根目录的 PowerShell 中运行；环境安装脚本需要 `uv`：

```powershell
.\venv\Scripts\python.exe .\scripts\setup_qwen21_runtime.py
.\venv\Scripts\python.exe .\scripts\download_qwen21.py
.\runtimes\qwen-image-2.1\Scripts\python.exe .\scripts\quantize_qwen21.py --precision nf4
.\runtimes\qwen-image-2.1\Scripts\python.exe .\scripts\quantize_qwen21.py --precision int8
```

模型来自官方 [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1/tree/790c92633540aa0cb11d9abf19eb46d861714758)，固定版本为 `790c92633540aa0cb11d9abf19eb46d861714758`。本次下载共 **27 个文件、33,131,616,412 字节**，不包含模型卡展示素材。下载器在完成整文件校验后才发布文件：权重核对官方 SHA-256，小型 Git 文件核对 Git blob SHA-1；校验报告同时记录文件大小和 SHA-256。重复运行下载脚本可核对已有文件并续传缺失部分。

**旧模型备份**

旧主权重 `qwen_image_fp8_e4m3fn.safetensors` 和旧编码器 `qwen_2.5_vl_7b_fp8_scaled.safetensors` 已移至 `models/legacy-qwen-image/`，不再出现在正常模型列表中。`models/VAE/qwen_image_vae.safetensors` 仍由 Krea 使用，保留在原位。

**原 BF16 版本已验证范围（2026-09-22）**

测试硬件为 RTX 5090 D v2（约 24 GB 显存）、约 32 GB 系统内存，使用本机现有 CUDA PyTorch。

| 实际操作 | 结果 |
| --- | --- |
| 前端直接点击生成，1024 × 1024、40 步 | 成功，界面记录 **71.6 秒**；已逐图检查，完整解码没有此前分块色条 |
| 参考图编辑，1024 × 1024、40 步 | 成功，**100.50 秒**；红色茶壶改为蓝色，输出 RGBA PNG |
| 2048 × 2048、2 步兼容性测试 | 成功，**260.56 秒**；此项只验证高分辨率流程和显存使用，不能作为 40 步速度或画质评价 |
| Qwen 切回原有 XL 并生成 256 × 256、2 步 | 成功，**15.19 秒**；随后再次切回 Qwen，前端生成成功 |

最终文生图样例：`outputs/qwen21-validation/final-1024.png`。验证汇总：`outputs/qwen21-validation/validation-summary.json`。**86 项自动检查通过**，并完成真实浏览器的模型切换、控件恢复和刷新保留选择检查。名称为 `full-1024` 和 `edit-1024` 的较早文件使用了已取消的分块解码，仅保留为排查记录。

PNG 正常参数保留；Qwen 输出跳过 Forge 的像素隐写，避免该功能将透明通道改为不透明。Forge 原生显存统计的 A/R 数值来自主进程，不包含独立 Qwen 推理进程，不能用它判断 Qwen 实际显存峰值；验证时另行监测了整张显卡。

旧 PNG 明确记录 `Qwen-Image-2.1` 与 `Diffusers BF16 / CPU offload` 时，恢复参数会自动选择 BF16 专用检查点；新图片会记录完整精度名称。

**量化版本验证（2026-09-23）**

| 实际操作 | 结果 |
| --- | --- |
| NF4，1024 × 1024、40 步文生图 | 成功，46.06 秒，整卡采样显存峰值约 12.4 GiB |
| INT8，1024 × 1024、40 步文生图 | 成功，48.31 秒，整卡采样显存峰值约 19.1 GiB |
| 同一 INT8 进程再次编辑参考图，1024 × 1024、40 步 | 成功，31.69 秒，整卡采样显存峰值约 16.0 GiB |
| INT8 切回 BF16，256 × 256、2 步 | 成功，62.86 秒；仅验证切换与原精度运行流程 |
| BF16 再切到 NF4 编辑参考图，512 × 512、40 步 | 成功，35.12 秒 |

上述耗时是单次实测；两项文生图包含首次加载权重，不是纯采样速度。图片均检查实际画面和 PNG 精度信息。INT8 编辑样例能完成改色，但也增强了纹理，此前 BF16 编辑也出现过该现象。量化更新本轮验证到 1024；2K 仅有前述 BF16 安装时的测试记录。

主环境自动检查 106 项，其中 105 项通过，1 项 GPU 检查按环境跳过；该项已在独立 Qwen 环境单独通过，覆盖 INT8 缓存卸载、连续调用及非连续输入。首次迁移和精度持久化也有自动回归检查。验证完成后本机选择保存为 **INT8**。

最终样例：`outputs/qwen21-validation/nf4-1024-0.png`、`outputs/qwen21-validation/int8-fixed-1024-0.png`、`outputs/qwen21-validation/nf4-edit-512-0.png`。完整记录：`outputs/qwen21-validation/quantization-validation-summary.json`。旧的 `int8-1024` 与 `diagnostic-int8*` 文件为修复前排查样例，不属于最终验证结果。

> Validation artifact paths above refer to local files. Generated images and private validation outputs are not published in this repository.
