# Stable Diffusion WebUI Forge（本 Fork 中文说明）

英文主文档请见 [README.md](README.md)。

本文件只覆盖这个 fork 的改动重点：新增模型家族支持（`qwen / anima / lumina / xl`）以及首次安装所需模型。

## 这个 Fork 的特别之处

- 左上角 `UI` 预设支持：`qwen`、`anima`、`lumina`、`xl`
- 预设会联动切换推荐参数（采样器、步数、CFG 等）和附加模块加载策略
- 针对 Lumina 增加了专用设置：`Maximum sequence length`、`Sampler shift`、`Normalize CFG output`

## 首次使用必须准备的模型

请把模型放到这些目录：

- 主模型（checkpoint）：`models/Stable-diffusion/`
- 文本编码器：`models/text_encoder/`
- VAE：`models/VAE/`

下表中的文件名是这个 fork 代码里的默认名：

| UI 预设 | 必需 checkpoint（`models/Stable-diffusion`） | 必需附加模块 | 下载来源 | 备注 |
|---|---|---|---|---|
| `qwen` | `qwen_image_fp8_e4m3fn.safetensors` | `models/text_encoder/qwen_2.5_vl_7b_fp8_scaled.safetensors` 和 `models/VAE/qwen_image_vae.safetensors` | [Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main)<br>[checkpoint](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors)<br>[text encoder](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors)<br>[VAE](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors) | checkpoint 名含 `qwen` 会自动识别为 qwen 家族 |
| `anima` | `anima-base-v1.0.safetensors` | `models/text_encoder/anima_text_encoder.safetensors` 和 `models/VAE/anima_vae.safetensors` | [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima/tree/main/split_files)<br>[diffusion model 目录](https://huggingface.co/circlestone-labs/Anima/tree/main/split_files/diffusion_models)<br>[text encoder 目录](https://huggingface.co/circlestone-labs/Anima/tree/main/split_files/text_encoders)<br>[VAE 目录](https://huggingface.co/circlestone-labs/Anima/tree/main/split_files/vae) | checkpoint 名含 `anima` 会自动识别；上游文件名可能与本 fork 默认名不同，见下方重命名说明 |
| `lumina` | 任意兼容 Lumina 的 checkpoint | 默认不需要附加模块 | [Alpha-VLLM/Lumina-Image-2.0](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0)<br>[Comfy repackaged all-in-one](https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/blob/main/all_in_one/lumina_2.safetensors)<br>[Neta-Lumina](https://huggingface.co/neta-art/Neta-Lumina) | 启用 lumina 预设后会使用 Lumina 专用参数逻辑 |
| `xl` | `novaAnimeXL_ilV170.safetensors` | `models/VAE/pppanimixVAE_il.safetensors` | [Civitai 搜索（checkpoint）](https://civitai.com/search/models?query=novaAnimeXL)<br>[Civitai 搜索（VAE）](https://civitai.com/search/models?query=pppanimix%20vae) | 社区模型文件名可能因发布者或版本略有变化 |

## Anima 文件重命名对照

如果你从 `circlestone-labs/Anima` 直接下载，请按下面映射重命名为本 fork 默认名：

- `anima-base-v1.0.safetensors` -> `anima-base-v1.0.safetensors`
- `qwen_3_06b_base.safetensors` -> `anima_text_encoder.safetensors`
- `qwen_image_vae.safetensors` -> `anima_vae.safetensors`

## 首次启动建议流程

1. 按上表下载并放置模型文件。
2. 启动 WebUI 后，在左上角选择对应 `UI` 预设。
3. 检查顶栏中的 `Checkpoint` 与附加模块是否匹配你安装的文件。
4. 如你使用了自定义文件名，手动选择一次后，`qwen/anima` 的选择会按预设保存。
