# Stable Diffusion WebUI Forge — 个人分支

本分支保留本机使用的 Qwen Image 2.1、Krea 2、Anima、Illustrious 模型预设、原生 Outpaint，以及适合手机浏览器的英文触控界面。基于 [Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) 和 AUTOMATIC1111 WebUI。

完整安装入口、模型清单、兼容性限制与验证范围见[英文主页](README.md)。

## 在 Colab 安装并启动

先选择支持原生 BF16 的 GPU（例如 L4 或 A100），再在该运行时中挂载 Google Drive。Forge 源码、完整 Python 解释器、主环境和 Qwen 环境都保存在 `MyDrive/ForgeColab/forge`；模型保存在同级的 `models`，文本嵌入保存在 `embeddings`。模型权重不在 GitHub 中，下面的安装命令也不会上传本机模型。

在 Colab 的一个代码单元中运行：

```python
from pathlib import Path
import os
import subprocess

storage = Path("/content/drive/MyDrive/ForgeColab")
if not os.path.ismount("/content/drive"):
    raise RuntimeError("Mount Google Drive before installing Forge.")
storage.mkdir(parents=True, exist_ok=True)
(storage / "models").mkdir(exist_ok=True)
repo = storage / "forge"
url = "https://github.com/pepperedmutton/stable-diffusion-webui-forge.git"
if repo.is_symlink():
    raise RuntimeError("Use a real forge directory inside your Drive storage.")
if not repo.exists():
    subprocess.run(["git", "clone", "--branch", "main", "--single-branch", url, str(repo)], check=True)
if not (repo / ".git").is_dir():
    raise RuntimeError("This directory is not a Forge Git clone.")
origin = subprocess.check_output(["git", "-C", str(repo), "remote", "get-url", "origin"], text=True).strip()
if origin.rstrip("/").removesuffix(".git") != url.removesuffix(".git"):
    raise RuntimeError("This checkout belongs to a different repository.")

%run /content/drive/MyDrive/ForgeColab/forge/colab/start.py --drive-root /content/drive/MyDrive/ForgeColab --install-only
```

首次安装不要求模型已齐全，也不会开启公开链接。

先在另一个 Colab 单元中下载模型。使用隐藏输入填写 Civitai API Key；它只通过内存传给下载器，不写进下载链接或笔记本源码：

```python
import getpass
import os

os.environ["CIVITAI_API_KEY"] = getpass.getpass("Civitai API key: ").strip()
try:
    %run /content/drive/MyDrive/ForgeColab/forge/colab/download_models.py --drive-root /content/drive/MyDrive/ForgeColab --download
finally:
    os.environ.pop("CIVITAI_API_KEY", None)
```

去掉 `--download` 可先查看清单和空间需求。下载器优先使用精确匹配的 Civitai 文件；该站缺少的公开模型（包括 Qwen Image 2.1）使用固定版本的官方来源。WAI Illustrious SDXL v150 替代本机 Novsw 合并模型，跳过三个私有 LoRA（`csky_sora_nova_il_r32_a16`、`galkan`、`shadow_beast`）。权重和文本嵌入直接保存到 Drive。中断后重跑相同命令即可续传；已完成文件会核验校验值。下载记录位于 `ForgeColab/download-state/download-report.json`，下载成功不代表已完成出图验收。

BF16 权重与 Qwen 独立环境准备好后，在合适的 GPU 上运行两条量化命令。输入和输出都在 Drive，保留原 BF16 文件：

```python
!/content/drive/MyDrive/ForgeColab/forge/runtimes/qwen-image-2.1/bin/python /content/drive/MyDrive/ForgeColab/forge/scripts/quantize_qwen21.py --precision nf4 --source /content/drive/MyDrive/ForgeColab/models/diffusers/Qwen-Image-2.1 --output /content/drive/MyDrive/ForgeColab/models/diffusers/Qwen-Image-2.1-NF4
!/content/drive/MyDrive/ForgeColab/forge/runtimes/qwen-image-2.1/bin/python /content/drive/MyDrive/ForgeColab/forge/scripts/quantize_qwen21.py --precision int8 --source /content/drive/MyDrive/ForgeColab/models/diffusers/Qwen-Image-2.1 --output /content/drive/MyDrive/ForgeColab/models/diffusers/Qwen-Image-2.1-INT8
```

每个输出目录的 `quantization-manifest.json` 记录校验值和重新加载检查。再次执行会复用已验证的完整结果；遇到不完整或不匹配的旧输出会停止，不直接覆盖。旧版分体 Qwen 文件另存于 `models/legacy-qwen-image`，不作为 Qwen Image 2.1 档位。

完成后启动 Forge：

```python
%run /content/drive/MyDrive/ForgeColab/forge/colab/start.py --drive-root /content/drive/MyDrive/ForgeColab
```

按提示输入本次使用的密码；留空会自动生成密码并显示在笔记本输出中，请勿公开该输出。启动后，在手机浏览器打开输出的 `https://…gradio.live` 链接，用户名为 `forge`。使用过程中保持 Colab 单元运行。

安装器会恢复本机扩展源码并应用手机版网页布局。完整 Python 3.10 保存在 `forge/.colab/python`，主环境保存在 `forge/venv`，Qwen 环境保存在 `forge/runtimes/qwen-image-2.1`。首次安装可能临时下载引导解释器，再将其完整解释器和标准库保存到 Drive；全部模型依赖只安装在 Drive，启动时不复制整套环境到 `/content`。Colab 自带解释器、必要的系统库及临时安装文件仍属于当前虚拟机。

重连后只需挂载同一个 Drive，再运行启动命令，即可直接调用保存的 Python 和依赖。安装器会检查路径和导入结果；新虚拟机缺少必要系统库时会补装。设置写入 `MyDrive/ForgeColab/state`，生成结果写入 `MyDrive/ForgeColab/outputs`。启动器不会自动下载缺少的模型、重新量化已有权重或更新 Git 源码。Drive 读取大量小文件可能较慢；挂载点报告的剩余空间也不等于账号存储配额。

启动时会列出缺少的可选扩展依赖。FaceID 另需兼容的 `insightface`，Linux 安装器不会自动安装它。安装中断或失败后，可在 `%run` 行末添加 `--repair-environment` 重试环境准备；这不会补齐模型权重。

该完整模型组合不支持 T4，安装器会提前提示。Colab 对免费托管环境中的内容生成网页界面有限制，请按 [Colab 官方说明](https://research.google.com/colaboratory/faq.html) 使用符合要求且有剩余计算单元的付费环境。GPU 类型与持续运行时间不保证；结束后停止单元并断开、删除运行时。

## 手机操作

参数默认锁定。点击 **Edit parameters** 解锁，再点 **Edit value** 修改数值；**Apply value** 提交该数值，**Cancel** 丢弃该数值的草稿。下拉框和复选框解锁后立即生效，**Finish editing** 只重新锁定，不撤销已经应用的修改。

滚动经过锁定滑块不会改变数值，生成操作有确认与防重复触发。Outpaint 使用 **Edit outpaint** 开始调整，再选择 **Apply and lock** 应用或 **Discard and lock** 丢弃草稿。新增手机控件均为英文。

浏览器触控仿真已覆盖防误触逻辑，但尚不能代替 Pixel 8 Pro 真机验收。原有画布不具备完整的双指缩放和平移，请使用数值控件精确调整 Outpaint。

## 源码与模型

仓库保留本机新增代码、Krea 等扩展及修改过的模型识别代码。权重、生成图片、个人配置、虚拟环境和缓存不发布。扩展源码及其来源、基准提交和校验值见 [colab/vendor](colab/vendor/README.md)。

[Qwen 2.1 详细说明](docs/QWEN_IMAGE_2_1.md) · [Outpaint 说明](extensions-builtin/forge_outpaint/README.md)

本地测试与临时目录测试不等于真实 Colab 验证。首次使用仍需在所选 Colab GPU 上完成启动和出图检查；本地 Qwen 历史性能数据不代表 Colab 性能。

Forge 使用 [AGPL-3.0](LICENSE.txt)，打包的组件和模型各自保留其许可证。
