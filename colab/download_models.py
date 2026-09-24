"""Verified, resumable downloads for this Forge model collection.

Run only after Drive has been mounted:
  %run /content/drive/MyDrive/ForgeColab/qa/scripts/download_models.py --drive-root /content/drive/MyDrive/ForgeColab --download

Without --download, prints a plan and does not create directories. This script
prefers exact Civitai files and uses fixed official sources for missing public
files. CIVITAI_API_KEY is read in memory and sent only to https://civitai.com.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
import threading
import time
import zipfile
from urllib.parse import urlsplit, urljoin

import requests

# Filled from live publisher metadata, retaining pinned revisions and hashes.
CATALOG = {'schema': 3, 'verified_utc': '2026-09-24', 'source_policy': 'Prefer exact Civitai matches; missing public models use pinned official sources as authorized; no local uploads', 'default_checkpoint': 'waiIllustriousSDXL_v150.safetensors', 'files': [{'storage': 'models', 'path': 'Stable-diffusion/waiIllustriousSDXL_v150.safetensors', 'bytes': 6938040682, 'sha256': 'befc694a296f75e996488ebf9f9db8a1493bd059b6e704b975829e87d5aeb4fa', 'provider': 'civitai', 'group': 'checkpoint', 'model_id': 827184, 'version_id': 2167369, 'file_id': 2062324, 'url': 'https://civitai.com/api/download/models/2167369?fileId=2062324', 'source': 'https://civitai.com/models/827184?modelVersionId=2167369'}, {'storage': 'models', 'path': 'ControlNet/ip-adapter-faceid-plusv2_sdxl_lora.safetensors', 'bytes': 371842896, 'sha256': 'f24b4bb2dad6638a09c00f151cde84991baf374409385bcbab53c1871a30cb7b', 'provider': 'civitai', 'group': 'auxiliary', 'model_id': 301797, 'version_id': 338913, 'file_id': 269150, 'url': 'https://civitai.com/api/download/models/338913?fileId=269150', 'source': 'https://civitai.com/models/301797?modelVersionId=338913'}, {'storage': 'models', 'path': 'ControlNet/openpose.safetensors', 'bytes': 2502140008, 'sha256': '0d8bacf24534dc6f2716f5d0ffa6085571928776f8687565ef290a17d9f3615c', 'provider': 'civitai', 'group': 'auxiliary', 'model_id': 136070, 'version_id': 1274546, 'file_id': 1179489, 'url': 'https://civitai.com/api/download/models/1274546?fileId=1179489', 'source': 'https://civitai.com/models/136070?modelVersionId=1274546'}, {'storage': 'models', 'path': 'ControlNetPreprocessor/CLIP-ViT-H-14.safetensors', 'bytes': 2528373448, 'sha256': '6ca9667da1ca9e0b0f75e46bb030f7e011f44f86cbfb8d5a36590fcd7507b030', 'provider': 'civitai', 'group': 'auxiliary', 'model_id': 2883234, 'version_id': 3258981, 'file_id': 3142466, 'url': 'https://civitai.com/api/download/models/3258981?fileId=3142466', 'source': 'https://civitai.com/models/2883234?modelVersionId=3258981'}, {'storage': 'models', 'path': 'RealESRGAN/RealESRGAN_x4plus_anime_6B.pth', 'bytes': 17938799, 'sha256': 'f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da', 'provider': 'civitai', 'group': 'auxiliary', 'model_id': 147821, 'version_id': 164904, 'file_id': 124735, 'url': 'https://civitai.com/api/download/models/164904?fileId=124735', 'source': 'https://civitai.com/models/147821?modelVersionId=164904'}, {'storage': 'models', 'path': 'VAE/ae.safetensors', 'bytes': 335304388, 'sha256': 'afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38', 'provider': 'civitai', 'group': 'auxiliary', 'model_id': 636193, 'version_id': 711305, 'file_id': 625974, 'url': 'https://civitai.com/api/download/models/711305?fileId=625974', 'source': 'https://civitai.com/models/636193?modelVersionId=711305'}, {'storage': 'models', 'path': 'VAE/anima_vae.safetensors', 'bytes': 253806246, 'sha256': 'a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f', 'provider': 'civitai', 'group': 'auxiliary', 'model_id': 1842123, 'version_id': 2089517, 'file_id': 1985231, 'url': 'https://civitai.com/api/download/models/2089517?fileId=1985231', 'source': 'https://civitai.com/models/1842123?modelVersionId=2089517'}, {'storage': 'models', 'path': 'VAE/pppanimixVAE_il.safetensors', 'bytes': 334640988, 'sha256': '5fc6a4f0501db32e9a8ec23bb3f39c376b96b011edb11ef72bfe503b798e3d1d', 'provider': 'civitai', 'group': 'auxiliary', 'model_id': 285852, 'version_id': 2250794, 'file_id': 2143397, 'url': 'https://civitai.com/api/download/models/2250794?fileId=2143397', 'source': 'https://civitai.com/models/285852?modelVersionId=2250794'}, {'storage': 'models', 'path': 'VAE/qwen_image_vae.safetensors', 'bytes': 253806246, 'sha256': 'a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f', 'provider': 'civitai', 'group': 'auxiliary', 'model_id': 1842123, 'version_id': 2089517, 'file_id': 1985231, 'url': 'https://civitai.com/api/download/models/2089517?fileId=1985231', 'source': 'https://civitai.com/models/1842123?modelVersionId=2089517'}, {'storage': 'models', 'path': 'text_encoder/anima_text_encoder.safetensors', 'bytes': 1192135096, 'sha256': 'cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba', 'provider': 'civitai', 'group': 'auxiliary', 'model_id': 2400206, 'version_id': 2698710, 'file_id': 2584763, 'url': 'https://civitai.com/api/download/models/2698710?fileId=2584763', 'source': 'https://civitai.com/models/2400206?modelVersionId=2698710'}, {'storage': 'models', 'path': 'text_encoder/gemma_3_4b_it_bf16.safetensors', 'bytes': 7765267250, 'sha256': '9ca0ed0c8b0093357c044fc528f439d369fb1029200dcd8011328771e8af8d6e', 'provider': 'civitai', 'group': 'auxiliary', 'model_id': 2197517, 'version_id': 2520101, 'file_id': 2407980, 'url': 'https://civitai.com/api/download/models/2520101?fileId=2407980', 'source': 'https://civitai.com/models/2197517?modelVersionId=2520101'}, {'storage': 'models', 'path': 'text_encoder/qwen3vl_4b_fp8_scaled.safetensors', 'bytes': 5242467968, 'sha256': '54bd5144df0bbc25dd6ccadfcb826b521445a1b06ae5a42570bdd2974ca87094', 'provider': 'civitai', 'group': 'auxiliary', 'model_id': 2753756, 'version_id': 3098281, 'file_id': 2977940, 'url': 'https://civitai.com/api/download/models/3098281?fileId=2977940', 'source': 'https://civitai.com/models/2753756?modelVersionId=3098281'}, {'storage': 'models', 'path': 'Stable-diffusion/JANKUTrainedNoobaiRouwei_v69.safetensors', 'bytes': 6938040674, 'sha256': '5d255f746e67a45b9255a3e7cff5e6cb69f83164acf77bc42fdb161be4385926', 'provider': 'civitai', 'group': 'checkpoint', 'model_id': 1277670, 'version_id': 2578565, 'file_id': 2465829, 'url': 'https://civitai.com/api/download/models/2578565?fileId=2465829', 'source': 'https://civitai.com/models/1277670?modelVersionId=2578565'}, {'storage': 'models', 'path': 'Stable-diffusion/anima_aestheticV11.safetensors', 'bytes': 4182230656, 'sha256': '3c1868387a3a1ff504bbb87c33678321965ead381fcf87afbd0264daa600c082', 'provider': 'civitai', 'group': 'checkpoint', 'model_id': 2458426, 'version_id': 3126581, 'file_id': 3007030, 'url': 'https://civitai.com/api/download/models/3126581?fileId=3007030', 'source': 'https://civitai.com/models/2458426?modelVersionId=3126581'}, {'storage': 'models', 'path': 'Stable-diffusion/krea2Cocoamixzero_v10.safetensors', 'bytes': 13141817728, 'sha256': '223515d5b39638e4487641f44424889ab42829fe3e820d3f83896e9bd63a61ab', 'provider': 'civitai', 'group': 'checkpoint', 'model_id': 2844761, 'version_id': 3211739, 'file_id': 3093484, 'url': 'https://civitai.com/api/download/models/3211739?fileId=3093484', 'source': 'https://civitai.com/models/2844761?modelVersionId=3211739'}, {'storage': 'models', 'path': 'Stable-diffusion/miaomiaoHarem_anima15.safetensors', 'bytes': 4182218328, 'sha256': 'eebcd007a03b6e456841ef976821001fb998284e074040fdb091932dad21a529', 'provider': 'civitai', 'group': 'checkpoint', 'model_id': 934764, 'version_id': 3153747, 'file_id': 3034524, 'url': 'https://civitai.com/api/download/models/3153747?fileId=3034524', 'source': 'https://civitai.com/models/934764?modelVersionId=3153747'}, {'storage': 'models', 'path': 'Stable-diffusion/miaomiaoHarem_v20.safetensors', 'bytes': 6938040400, 'sha256': 'ffc89da29bdc291997094265ef347927ab6a8db0bd7d225bbc60938afc625cdc', 'provider': 'civitai', 'group': 'checkpoint', 'model_id': 934764, 'version_id': 2851583, 'file_id': 2737749, 'url': 'https://civitai.com/api/download/models/2851583?fileId=2737749', 'source': 'https://civitai.com/models/934764?modelVersionId=2851583'}, {'storage': 'models', 'path': 'Stable-diffusion/novaAnimeXL_ilV170.safetensors', 'bytes': 6938512340, 'sha256': 'a83e8664e4bf349f31542791149de9361767d1e9dbfebb4ba05bfbd5f5a9a28b', 'provider': 'civitai', 'group': 'checkpoint', 'model_id': 376130, 'version_id': 2741698, 'file_id': 2628050, 'url': 'https://civitai.com/api/download/models/2741698?fileId=2628050', 'source': 'https://civitai.com/models/376130?modelVersionId=2741698'}, {'storage': 'models', 'path': 'Stable-diffusion/specustrious_v40.safetensors', 'bytes': 7105348366, 'sha256': '7eca31007cd49291055b6a0d61a9b5f34e8a109adff12864c34543770ae43004', 'provider': 'civitai', 'group': 'checkpoint', 'model_id': 2267817, 'version_id': 2833380, 'file_id': 2719438, 'url': 'https://civitai.com/api/download/models/2833380?fileId=2719438', 'source': 'https://civitai.com/models/2267817?modelVersionId=2833380'}, {'storage': 'embeddings', 'path': 'lazyhand.safetensors', 'bytes': 123032, 'sha256': 'e12c4bd3364944a006597e4fabfb60ac8a8efca518d7d8d14166c018d8423805', 'provider': 'civitai', 'group': 'embeddings', 'model_id': 1302719, 'version_id': 2268235, 'file_id': 2160398, 'url': 'https://civitai.com/api/download/models/2268235?fileId=2160398', 'source': 'https://civitai.com/models/1302719?modelVersionId=2268235'}, {'storage': 'embeddings', 'path': 'lazyneg.safetensors', 'bytes': 344216, 'sha256': 'ba21023c7054d4488f77dcacf0cd8e0bb3f4f64e9a7810b3287d4f99c06cb16a', 'provider': 'civitai', 'group': 'embeddings', 'model_id': 1302719, 'version_id': 1860747, 'file_id': 1760455, 'url': 'https://civitai.com/api/download/models/1860747?fileId=1760455', 'source': 'https://civitai.com/models/1302719?modelVersionId=1860747'}, {'storage': 'embeddings', 'path': 'lazypos.safetensors', 'bytes': 180376, 'sha256': '30866692653cb0063484dec240bc7971adc5767753b0cbbe072e2ddd7ff16b81', 'provider': 'civitai', 'group': 'embeddings', 'model_id': 1302719, 'version_id': 1833157, 'file_id': 1733353, 'url': 'https://civitai.com/api/download/models/1833157?fileId=1733353', 'source': 'https://civitai.com/models/1302719?modelVersionId=1833157'}, {'storage': 'embeddings', 'path': 'lazywet.safetensors', 'bytes': 417944, 'sha256': 'ab6e3863d319f43af751ebecbdbd009344fa8216d02f71903b702496b3bd6f0e', 'provider': 'civitai', 'group': 'embeddings', 'model_id': 1302719, 'version_id': 2512494, 'file_id': 2400358, 'url': 'https://civitai.com/api/download/models/2512494?fileId=2400358', 'source': 'https://civitai.com/models/1302719?modelVersionId=2512494'}, {'storage': 'models', 'path': 'legacy-qwen-image/qwen_2.5_vl_7b_fp8_scaled.safetensors', 'bytes': 9384670680, 'sha256': 'cb5636d852a0ea6a9075ab1bef496c0db7aef13c02350571e388aea959c5c0b4', 'provider': 'civitai', 'group': 'legacy-qwen', 'model_id': 1842123, 'version_id': 2089462, 'file_id': 1985224, 'url': 'https://civitai.com/api/download/models/2089462?fileId=1985224', 'source': 'https://civitai.com/models/1842123?modelVersionId=2089462'}, {'storage': 'models', 'path': 'legacy-qwen-image/qwen_image_fp8_e4m3fn.safetensors', 'bytes': 20430635136, 'sha256': '98763a127701eb6fb59096f7742cb3aa7d64ed510b9f4e882d8351f8176e3ce3', 'provider': 'civitai', 'group': 'legacy-qwen', 'model_id': 1843568, 'version_id': 2086298, 'file_id': 1982308, 'url': 'https://civitai.com/api/download/models/2086298?fileId=1982308', 'source': 'https://civitai.com/models/1843568?modelVersionId=2086298'}, {'path': 'diffusers/Qwen-Image-2.1/.gitattributes', 'bytes': 1630, 'git_sha1': '7236eee3e26bfd60df2cd50255284a1c1716274a', 'url': 'https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/790c92633540aa0cb11d9abf19eb46d861714758/.gitattributes?download=true', 'source': 'https://huggingface.co/Qwen/Qwen-Image-2.1/blob/790c92633540aa0cb11d9abf19eb46d861714758/.gitattributes', 'provider': 'huggingface', 'group': 'qwen-bf16', 'revision': '790c92633540aa0cb11d9abf19eb46d861714758', 'storage': 'models'}, {'path': 'diffusers/Qwen-Image-2.1/LICENSE', 'bytes': 7831, 'git_sha1': '13ae08d5a5828f508cf0e852b1b316c0c0bbb9b3', 'url': 'https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/790c92633540aa0cb11d9abf19eb46d861714758/LICENSE?download=true', 'source': 'https://huggingface.co/Qwen/Qwen-Image-2.1/blob/790c92633540aa0cb11d9abf19eb46d861714758/LICENSE', 'provider': 'huggingface', 'group': 'qwen-bf16', 'revision': '790c92633540aa0cb11d9abf19eb46d861714758', 'storage': 'models'}, {'path': 'diffusers/Qwen-Image-2.1/README.md', 'bytes': 5358, 'git_sha1': 'df7f635aeb0358a45427d74610535dd1dc0b992d', 'url': 'https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/790c92633540aa0cb11d9abf19eb46d861714758/README.md?download=true', 'source': 'https://huggingface.co/Qwen/Qwen-Image-2.1/blob/790c92633540aa0cb11d9abf19eb46d861714758/README.md', 'provider': 'huggingface', 'group': 'qwen-bf16', 'revision': '790c92633540aa0cb11d9abf19eb46d861714758', 'storage': 'models'}, {'path': 'diffusers/Qwen-Image-2.1/model_index.json', 'bytes': 447, 'git_sha1': '85d02d05f150cbe24a8fd3542062b565767bbe7b', 'url': 'https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/790c92633540aa0cb11d9abf19eb46d861714758/model_index.json?download=true', 'source': 'https://huggingface.co/Qwen/Qwen-Image-2.1/blob/790c92633540aa0cb11d9abf19eb46d861714758/model_index.json', 'provider': 'huggingface', 'group': 'qwen-bf16', 'revision': '790c92633540aa0cb11d9abf19eb46d861714758', 'storage': 'models'}, {'path': 'diffusers/Qwen-Image-2.1/processor/added_tokens.json', 'bytes': 707, 'git_sha1': 'b54f9135e44c1e81047e8d05cb027af8bc039eed', 'url': 'https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/790c92633540aa0cb11d9abf19eb46d861714758/processor/added_tokens.json?download=true', 'source': 'https://huggingface.co/Qwen/Qwen-Image-2.1/blob/790c92633540aa0cb11d9abf19eb46d861714758/processor/added_tokens.json', 'provider': 'huggingface', 'group': 'qwen-bf16', 'revision': '790c92633540aa0cb11d9abf19eb46d861714758', 'storage': 'models'}, {'path': 'diffusers/Qwen-Image-2.1/processor/chat_template.jinja', 'bytes': 5292, 'git_sha1': '124386803f142761528f710e77ae483f5f8c4fc4', 'url': 'https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/790c92633540aa0cb11d9abf19eb46d861714758/processor/chat_template.jinja?download=true', 'source': 'https://huggingface.co/Qwen/Qwen-Image-2.1/blob/790c92633540aa0cb11d9abf19eb46d861714758/processor/chat_template.jinja', 'provider': 'huggingface', 'group': 'qwen-bf16', 'revision': '790c92633540aa0cb11d9abf19eb46d861714758', 'storage': 'models'}, {'path': 'diffusers/Qwen-Image-2.1/processor/merges.txt', 'bytes': 1671853, 'git_sha1': '31349551d90c7606f325fe0f11bbb8bd5fa0d7c7', 'url': 'https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/790c92633540aa0cb11d9abf19eb46d861714758/processor/merges.txt?download=true', 'source': 'https://huggingface.co/Qwen/Qwen-Image-2.1/blob/790c92633540aa0cb11d9abf19eb46d861714758/processor/merges.txt', 'provider': 'huggingface', 'group': 'qwen-bf16', 'revision': '790c92633540aa0cb11d9abf19eb46d861714758', 'storage': 'models'}, {'path': 'diffusers/Qwen-Image-2.1/processor/preprocessor_config.json', 'bytes': 782, 'git_sha1': '2fa6553554f69b4e3a6b742a75264ff9dff90ad0', 'url': 'https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/790c92633540aa0cb11d9abf19eb46d861714758/processor/preprocessor_config.json?download=true', 'source': 'https://huggingface.co/Qwen/Qwen-Image-2.1/blob/790c92633540aa0cb11d9abf19eb46d861714758/processor/preprocessor_config.json', 'provider': 'huggingface', 'group': 'qwen-bf16', 'revision': '790c92633540aa0cb11d9abf19eb46d861714758', 'storage': 'models'}, {'path': 'diffusers/Qwen-Image-2.1/processor/special_tokens_map.json', 'bytes': 613, 'git_sha1': 'ac23c0aaa2434523c494330aeb79c58395378103', 'url': 'https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/790c92633540aa0cb11d9abf19eb46d861714758/processor/special_tokens_map.json?download=true', 'source': 'https://huggingface.co/Qwen/Qwen-Image-2.1/blob/790c92633540aa0cb11d9abf19eb46d861714758/processor/special_tokens_map.json', 'provider': 'huggingface', 'group': 'qwen-bf16', 'revision': '790c92633540aa0cb11d9abf19eb46d861714758', 'storage': 'models'}, {'path': 'diffusers/Qwen-Image-2.1/processor/tokenizer.json', 'bytes': 11422654, 'sha256': 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4', 'url': 'https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/790c92633540aa0cb11d9abf19eb46d861714758/processor/tokenizer.json?download=true', 'source': 'https://huggingface.co/Qwen/Qwen-Image-2.1/blob/790c92633540aa0cb11d9abf19eb46d861714758/processor/tokenizer.json', 'provider': 'huggingface', 'group': 'qwen-bf16', 'revision': '790c92633540aa0cb11d9abf19eb46d861714758', 'storage': 'models'}, {'path': 'diffusers/Qwen-Image-2.1/processor/tokenizer_config.json', 'bytes': 5445, 'git_sha1': 'fec7f182135c384a08ea2fc9635566ca79c45afc', 'url': 'https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/790c92633540aa0cb11d9abf19eb46d861714758/processor/tokenizer_config.json?download=true', 'source': 'https://huggingface.co/Qwen/Qwen-Image-2.1/blob/790c92633540aa0cb11d9abf19eb46d861714758/processor/tokenizer_config.json', 'provider': 'huggingface', 'group': 'qwen-bf16', 'revision': '790c92633540aa0cb11d9abf19eb46d861714758', 'storage': 'models'}, {'path': 'diffusers/Qwen-Image-2.1/processor/video_preprocessor_config.json', 'bytes': 817, 'git_sha1': 'e32b1d90356f74457edbfca27aa1e121dab7cdc7', 'url': 'https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/790c92633540aa0cb11d9abf19eb46d861714758/processor/video_preprocessor_config.json?download=true', 'source': 'https://huggingface.co/Qwen/Qwen-Image-2.1/blob/790c92633540aa0cb11d9abf19eb46d861714758/processor/video_preprocessor_config.json', 'provider': 'huggingface', 'group': 'qwen-bf16', 'revision': '790c92633540aa0cb11d9abf19eb46d861714758', 'storage': 'models'}, {'path': 'diffusers/Qwen-Image-2.1/processor/vocab.json', 'bytes': 2776833, 'git_sha1': '4783fe10ac3adce15ac8f358ef5462739852c569', 'url': 'https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/790c92633540aa0cb11d9abf19eb46d861714758/processor/vocab.json?download=true', 'source': 'https://huggingface.co/Qwen/Qwen-Image-2.1/blob/790c92633540aa0cb11d9abf19eb46d861714758/processor/vocab.json', 'provider': 'huggingface', 'group': 'qwen-bf16', 'revision': '790c92633540aa0cb11d9abf19eb46d861714758', 'storage': 'models'}, {'path': 'diffusers/Qwen-Image-2.1/scheduler/scheduler_config.json', 'bytes': 485, 'git_sha1': '45cd05b021892b1cafbd16d61ce692637e88bedd', 'url': 'https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/790c92633540aa0cb11d9abf19eb46d861714758/scheduler/scheduler_config.json?download=true', 'source': 'https://huggingface.co/Qwen/Qwen-Image-2.1/blob/790c92633540aa0cb11d9abf19eb46d861714758/scheduler/scheduler_config.json', 'provider': 'huggingface', 'group': 'qwen-bf16', 'revision': '790c92633540aa0cb11d9abf19eb46d861714758', 'storage': 'models'}, {'path': 'diffusers/Qwen-Image-2.1/text_encoder/config.json', 'bytes': 1517, 'git_sha1': '581ce4655dde5272e577065dd08b1fda7e97d6e8', 'url': 'https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/790c92633540aa0cb11d9abf19eb46d861714758/text_encoder/config.json?download=true', 'source': 'https://huggingface.co/Qwen/Qwen-Image-2.1/blob/790c92633540aa0cb11d9abf19eb46d861714758/text_encoder/config.json', 'provider': 'huggingface', 'group': 'qwen-bf16', 'revision': '790c92633540aa0cb11d9abf19eb46d861714758', 'storage': 'models'}, {'path': 'diffusers/Qwen-Image-2.1/text_encoder/generation_config.json', 'bytes': 213, 'git_sha1': 'bdb4e0372b04220c80df250f2b3354d1c81c97cb', 'url': 'https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/790c92633540aa0cb11d9abf19eb46d861714758/text_encoder/generation_config.json?download=true', 'source': 'https://huggingface.co/Qwen/Qwen-Image-2.1/blob/790c92633540aa0cb11d9abf19eb46d861714758/text_encoder/generation_config.json', 'provider': 'huggingface', 'group': 'qwen-bf16', 'revision': '790c92633540aa0cb11d9abf19eb46d861714758', 'storage': 'models'}, {'path': 'diffusers/Qwen-Image-2.1/text_encoder/model-00001-of-00004.safetensors', 'bytes': 4998056552, 'sha256': 'dde00291b5f7fb92013895310a3da0ddba78674df9f10d505d375243dc01fc6f', 'url': 'https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/790c92633540aa0cb11d9abf19eb46d861714758/text_encoder/model-00001-of-00004.safetensors?download=true', 'source': 'https://huggingface.co/Qwen/Qwen-Image-2.1/blob/790c92633540aa0cb11d9abf19eb46d861714758/text_encoder/model-00001-of-00004.safetensors', 'provider': 'huggingface', 'group': 'qwen-bf16', 'revision': '790c92633540aa0cb11d9abf19eb46d861714758', 'storage': 'models'}, {'path': 'diffusers/Qwen-Image-2.1/text_encoder/model-00002-of-00004.safetensors', 'bytes': 4915962464, 'sha256': '9047faccc0a6d98496a52d55f27be1c94a9c259d1e283fbea0128d054a948d42', 'url': 'https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/790c92633540aa0cb11d9abf19eb46d861714758/text_encoder/model-00002-of-00004.safetensors?download=true', 'source': 'https://huggingface.co/Qwen/Qwen-Image-2.1/blob/790c92633540aa0cb11d9abf19eb46d861714758/text_encoder/model-00002-of-00004.safetensors', 'provider': 'huggingface', 'group': 'qwen-bf16', 'revision': '790c92633540aa0cb11d9abf19eb46d861714758', 'storage': 'models'}, {'path': 'diffusers/Qwen-Image-2.1/text_encoder/model-00003-of-00004.safetensors', 'bytes': 4915962496, 'sha256': '8c54187654c0176b73ae73785bf791dc9a14c9df7fb4310083a09d42048cb57e', 'url': 'https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/790c92633540aa0cb11d9abf19eb46d861714758/text_encoder/model-00003-of-00004.safetensors?download=true', 'source': 'https://huggingface.co/Qwen/Qwen-Image-2.1/blob/790c92633540aa0cb11d9abf19eb46d861714758/text_encoder/model-00003-of-00004.safetensors', 'provider': 'huggingface', 'group': 'qwen-bf16', 'revision': '790c92633540aa0cb11d9abf19eb46d861714758', 'storage': 'models'}, {'path': 'diffusers/Qwen-Image-2.1/text_encoder/model-00004-of-00004.safetensors', 'bytes': 2704357976, 'sha256': '5311532aaaeae3259eb6a7b2c600636be1159adf7ded35f53579f7d0e7d43cdd', 'url': 'https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/790c92633540aa0cb11d9abf19eb46d861714758/text_encoder/model-00004-of-00004.safetensors?download=true', 'source': 'https://huggingface.co/Qwen/Qwen-Image-2.1/blob/790c92633540aa0cb11d9abf19eb46d861714758/text_encoder/model-00004-of-00004.safetensors', 'provider': 'huggingface', 'group': 'qwen-bf16', 'revision': '790c92633540aa0cb11d9abf19eb46d861714758', 'storage': 'models'}, {'path': 'diffusers/Qwen-Image-2.1/text_encoder/model.safetensors.index.json', 'bytes': 67795, 'git_sha1': '6d4d74c934b05b8addb0744305eef3674650cd8b', 'url': 'https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/790c92633540aa0cb11d9abf19eb46d861714758/text_encoder/model.safetensors.index.json?download=true', 'source': 'https://huggingface.co/Qwen/Qwen-Image-2.1/blob/790c92633540aa0cb11d9abf19eb46d861714758/text_encoder/model.safetensors.index.json', 'provider': 'huggingface', 'group': 'qwen-bf16', 'revision': '790c92633540aa0cb11d9abf19eb46d861714758', 'storage': 'models'}, {'path': 'diffusers/Qwen-Image-2.1/transformer/config.json', 'bytes': 370, 'git_sha1': '706df60672976086089e99f6595037f917d7602e', 'url': 'https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/790c92633540aa0cb11d9abf19eb46d861714758/transformer/config.json?download=true', 'source': 'https://huggingface.co/Qwen/Qwen-Image-2.1/blob/790c92633540aa0cb11d9abf19eb46d861714758/transformer/config.json', 'provider': 'huggingface', 'group': 'qwen-bf16', 'revision': '790c92633540aa0cb11d9abf19eb46d861714758', 'storage': 'models'}, {'path': 'diffusers/Qwen-Image-2.1/transformer/diffusion_pytorch_model-00001-of-00002.safetensors', 'bytes': 9968332504, 'sha256': '9e6bc2d641e67bf277895ea8777141044a38f3edb7101bc469b2961dd7c36b4b', 'url': 'https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/790c92633540aa0cb11d9abf19eb46d861714758/transformer/diffusion_pytorch_model-00001-of-00002.safetensors?download=true', 'source': 'https://huggingface.co/Qwen/Qwen-Image-2.1/blob/790c92633540aa0cb11d9abf19eb46d861714758/transformer/diffusion_pytorch_model-00001-of-00002.safetensors', 'provider': 'huggingface', 'group': 'qwen-bf16', 'revision': '790c92633540aa0cb11d9abf19eb46d861714758', 'storage': 'models'}, {'path': 'diffusers/Qwen-Image-2.1/transformer/diffusion_pytorch_model-00002-of-00002.safetensors', 'bytes': 4261951904, 'sha256': '3aaf234dcbe128530479735854a346b5e3e66283b7c11db56f836bbd1c13ebaa', 'url': 'https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/790c92633540aa0cb11d9abf19eb46d861714758/transformer/diffusion_pytorch_model-00002-of-00002.safetensors?download=true', 'source': 'https://huggingface.co/Qwen/Qwen-Image-2.1/blob/790c92633540aa0cb11d9abf19eb46d861714758/transformer/diffusion_pytorch_model-00002-of-00002.safetensors', 'provider': 'huggingface', 'group': 'qwen-bf16', 'revision': '790c92633540aa0cb11d9abf19eb46d861714758', 'storage': 'models'}, {'path': 'diffusers/Qwen-Image-2.1/transformer/diffusion_pytorch_model.safetensors.index.json', 'bytes': 30283, 'git_sha1': 'bd64579cfd093a6a4ea8fb492217790018c53c0d', 'url': 'https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/790c92633540aa0cb11d9abf19eb46d861714758/transformer/diffusion_pytorch_model.safetensors.index.json?download=true', 'source': 'https://huggingface.co/Qwen/Qwen-Image-2.1/blob/790c92633540aa0cb11d9abf19eb46d861714758/transformer/diffusion_pytorch_model.safetensors.index.json', 'provider': 'huggingface', 'group': 'qwen-bf16', 'revision': '790c92633540aa0cb11d9abf19eb46d861714758', 'storage': 'models'}, {'path': 'diffusers/Qwen-Image-2.1/vae/config.json', 'bytes': 2079, 'git_sha1': 'ac6fe795d898a5eff16e4527256c2d039f5b0d4b', 'url': 'https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/790c92633540aa0cb11d9abf19eb46d861714758/vae/config.json?download=true', 'source': 'https://huggingface.co/Qwen/Qwen-Image-2.1/blob/790c92633540aa0cb11d9abf19eb46d861714758/vae/config.json', 'provider': 'huggingface', 'group': 'qwen-bf16', 'revision': '790c92633540aa0cb11d9abf19eb46d861714758', 'storage': 'models'}, {'path': 'diffusers/Qwen-Image-2.1/vae/diffusion_pytorch_model.safetensors', 'bytes': 1350989512, 'sha256': 'a07a1b7c4ee2966a1b3bdc37de9b4f983d56937e46619f709a80b6e490675417', 'url': 'https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/790c92633540aa0cb11d9abf19eb46d861714758/vae/diffusion_pytorch_model.safetensors?download=true', 'source': 'https://huggingface.co/Qwen/Qwen-Image-2.1/blob/790c92633540aa0cb11d9abf19eb46d861714758/vae/diffusion_pytorch_model.safetensors', 'provider': 'huggingface', 'group': 'qwen-bf16', 'revision': '790c92633540aa0cb11d9abf19eb46d861714758', 'storage': 'models'}, {'storage': 'models', 'path': 'Codeformer/codeformer-v0.1.0.pth', 'bytes': 376637898, 'sha256': '1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7', 'url': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth', 'provider': 'official-github', 'source': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth', 'group': 'auxiliary', 'checksum_basis': 'exact local file SHA256; official release asset size matched'}, {'storage': 'models', 'path': 'GFPGAN/detection_Resnet50_Final.pth', 'bytes': 109497761, 'sha256': '6d1de9c2944f2ccddca5f5e010ea5ae64a39845a86311af6fdf30841b0a5a16d', 'url': 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth', 'provider': 'official-github', 'source': 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth', 'group': 'auxiliary', 'checksum_basis': 'exact local file SHA256; official release asset size matched'}, {'storage': 'models', 'path': 'GFPGAN/parsing_parsenet.pth', 'bytes': 85331193, 'sha256': '3d558d8d0e42c20224f13cf5a29c79eba2d59913419f945545d8cf7b72920de2', 'url': 'https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth', 'provider': 'official-github', 'source': 'https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth', 'group': 'auxiliary', 'checksum_basis': 'exact local file SHA256; official release asset size matched'}, {'storage': 'models', 'path': 'torch_deepdanbooru/model-resnet_custom_v3.pt', 'bytes': 644144887, 'sha256': '3841542cda4dd037da12a565e854b3347bb2eec8fbcd95ea3941b2c68990a355', 'url': 'https://github.com/AUTOMATIC1111/TorchDeepDanbooru/releases/download/v1/model-resnet_custom_v3.pt', 'provider': 'official-github', 'source': 'https://github.com/AUTOMATIC1111/TorchDeepDanbooru/releases/download/v1/model-resnet_custom_v3.pt', 'group': 'auxiliary', 'checksum_basis': 'exact local file SHA256; official release asset size matched'}, {'storage': 'models', 'path': 'VAE-approx/model.pt', 'bytes': 213777, 'sha256': '4f88c9078bb2238cdd0d8864671dd33e3f42e091e41f08903f3c15e4a54a9b39', 'url': 'https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/download/v1.0.0-pre/model.pt', 'provider': 'official-github', 'source': 'https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/download/v1.0.0-pre/model.pt', 'group': 'auxiliary', 'checksum_basis': 'exact local file SHA256; official release asset size matched'}, {'storage': 'models', 'path': 'VAE-approx/vaeapprox-sdxl.pt', 'bytes': 213777, 'sha256': '6c8db511f910184436a613807ce06a729f4cf96bbf7977c7c6c40d5b3e4a8333', 'url': 'https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/download/v1.0.0-pre/vaeapprox-sdxl.pt', 'provider': 'official-github', 'source': 'https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/download/v1.0.0-pre/vaeapprox-sdxl.pt', 'group': 'auxiliary', 'checksum_basis': 'exact local file SHA256; official release asset size matched'}, {'storage': 'models', 'path': 'ControlNetPreprocessor/openpose/body_pose_model.pth', 'bytes': 209267595, 'sha256': '25a948c16078b0f08e236bda51a385d855ef4c153598947c28c0d47ed94bb746', 'url': 'https://huggingface.co/lllyasviel/Annotators/resolve/982e7edaec38759d914a963c48c4726685de7d96/body_pose_model.pth?download=true', 'provider': 'huggingface', 'source': 'https://huggingface.co/lllyasviel/Annotators/resolve/982e7edaec38759d914a963c48c4726685de7d96/body_pose_model.pth?download=true', 'group': 'auxiliary', 'revision': '982e7edaec38759d914a963c48c4726685de7d96'}, {'storage': 'models', 'path': 'ControlNetPreprocessor/openpose/facenet.pth', 'bytes': 153718792, 'sha256': '8beb52e548624ffcc4aed12af7aee7dcbfaeea420c75609fee999fe7add79d43', 'url': 'https://huggingface.co/lllyasviel/Annotators/resolve/982e7edaec38759d914a963c48c4726685de7d96/facenet.pth?download=true', 'provider': 'huggingface', 'source': 'https://huggingface.co/lllyasviel/Annotators/resolve/982e7edaec38759d914a963c48c4726685de7d96/facenet.pth?download=true', 'group': 'auxiliary', 'revision': '982e7edaec38759d914a963c48c4726685de7d96'}, {'storage': 'models', 'path': 'ControlNetPreprocessor/openpose/hand_pose_model.pth', 'bytes': 147341049, 'sha256': 'b76b00d1750901abd07b9f9d8c98cc3385b8fe834a26d4b4f0aad439e75fc600', 'url': 'https://huggingface.co/lllyasviel/Annotators/resolve/982e7edaec38759d914a963c48c4726685de7d96/hand_pose_model.pth?download=true', 'provider': 'huggingface', 'source': 'https://huggingface.co/lllyasviel/Annotators/resolve/982e7edaec38759d914a963c48c4726685de7d96/hand_pose_model.pth?download=true', 'group': 'auxiliary', 'revision': '982e7edaec38759d914a963c48c4726685de7d96'}, {'storage': 'models', 'path': 'BLIP/model_base_caption_capfilt_large.pth', 'bytes': 896081425, 'sha256': '96ac8749bd0a568c274ebe302b3a3748ab9be614c737f3d8c529697139174086', 'url': 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth?generation=1646265144744181', 'provider': 'official-google-storage', 'source': 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth?generation=1646265144744181', 'group': 'auxiliary', 'checksum_basis': 'official object MD5 matches original; final SHA256 must match original'}, {'storage': 'models', 'path': 'CLIP/ViT-L-14.pt', 'bytes': 932768134, 'sha256': 'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836', 'url': 'https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt', 'provider': 'official-openai', 'source': 'https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt', 'group': 'auxiliary', 'required_dependency_added': True}, {'storage': 'models', 'path': 'karlo/ViT-L-14_stats.th', 'bytes': 7079, 'sha256': '37113fd962c56652f7052c3182b922b958b7efc4be5072d3049cbbe1bdb0af79', 'url': 'https://arena.kakaocdn.net/brainrepo/models/karlo-public/v1.0.0.alpha/0b62380a75e56f073e2844ab5199153d/ViT-L-14_stats.th', 'provider': 'official-kakaobrain', 'source': 'https://arena.kakaocdn.net/brainrepo/models/karlo-public/v1.0.0.alpha/0b62380a75e56f073e2844ab5199153d/ViT-L-14_stats.th', 'group': 'auxiliary', 'required_dependency_added': False}, {'storage': 'models', 'path': 'ControlNet/ip-adapter-faceid-plusv2_sdxl.bin', 'bytes': 1487555181, 'sha256': 'c6945d82b543700cc3ccbb98d363b837e9c596281607857c74b713a876daf5fb', 'url': 'https://civitai.com/api/download/models/338917?fileId=269158', 'provider': 'civitai', 'source': 'https://civitai.com/models/301776?modelVersionId=338917', 'group': 'auxiliary', 'required_dependency_added': True, 'official_sha256_reference': 'https://huggingface.co/h94/IP-Adapter-FaceID/blob/43907e6f44d079bf1a9102d9a6e56aef7a219bae/ip-adapter-faceid-plusv2_sdxl.bin'}, {'storage': 'models', 'path': 'Lora/ip-adapter-faceid-plusv2_sdxl_lora.safetensors', 'bytes': 371842896, 'sha256': 'f24b4bb2dad6638a09c00f151cde84991baf374409385bcbab53c1871a30cb7b', 'provider': 'civitai', 'group': 'auxiliary', 'model_id': 301797, 'version_id': 338913, 'file_id': 269150, 'url': 'https://civitai.com/api/download/models/338913?fileId=269150', 'source': 'https://civitai.com/models/301797?modelVersionId=338913', 'required_dependency_added': True, 'purpose': 'Expose the existing companion weights to the Forge LoRA registry'}], 'unavailable_profiles': [], 'local_derivations': [{'path': 'diffusers/Qwen-Image-2.1-NF4', 'precision': 'nf4'}, {'path': 'diffusers/Qwen-Image-2.1-INT8', 'precision': 'int8'}], 'limitations': ['Qwen NF4 and INT8 must be derived from the verified BF16 bundle after download.', 'Three locally trained LoRAs are excluded as requested.', 'Legacy Qwen weights are preserved separately; they are not additional current Forge presets.'], 'archives': [{'storage': 'download-state', 'path': 'archives/buffalo_l.zip', 'group': 'auxiliary', 'bytes': 288621354, 'sha256': '80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f', 'url': 'https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip', 'source': 'https://github.com/deepinsight/insightface/releases/tag/model-zoo', 'provider': 'official-github', 'members': [{'storage': 'models', 'path': 'insightface/models/buffalo_l/1k3d68.onnx', 'bytes': 143607619, 'sha256': 'df5c06b8a0c12e422b2ed8947b8869faa4105387f199c477af038aa01f9a45cc'}, {'storage': 'models', 'path': 'insightface/models/buffalo_l/2d106det.onnx', 'bytes': 5030888, 'sha256': 'f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf'}, {'storage': 'models', 'path': 'insightface/models/buffalo_l/det_10g.onnx', 'bytes': 16923827, 'sha256': '5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91'}, {'storage': 'models', 'path': 'insightface/models/buffalo_l/genderage.onnx', 'bytes': 1322532, 'sha256': '4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb'}, {'storage': 'models', 'path': 'insightface/models/buffalo_l/w600k_r50.onnx', 'bytes': 174383860, 'sha256': '4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43'}]}], 'required_dependencies_added': ['FaceID Plus v2 SDXL adapter', 'FaceID companion LoRA registry entry', 'OpenAI CLIP ViT-L/14 for image interrogation']}
PRINT_LOCK = threading.Lock()


class AccessDenied(RuntimeError):
    pass


class IntegrityError(RuntimeError):
    pass


def emit(event, **values):
    with PRINT_LOCK:
        print(json.dumps({"event": event, **values}), flush=True)


def safe_relative(value):
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value or
            any(ord(c) < 32 for c in value) or PurePosixPath(value).is_absolute() or
            any(part in ("", ".", "..") for part in value.split("/"))):
        raise ValueError("Unsafe model path")
    return value


def safe_target(base, relative):
    safe_relative(relative)
    base = Path(base).resolve()
    path = base / relative
    if not path.resolve().is_relative_to(base):
        raise ValueError("A model path escapes its destination")
    cursor = path
    while cursor != base:
        if cursor.is_symlink():
            raise ValueError("Download destinations must not contain symlinks")
        cursor = cursor.parent
    return path


def validate_item(item):
    safe_relative(item["path"])
    if item.get("storage", "models") not in ("models", "embeddings", "download-state"):
        raise ValueError("Unknown model storage area")
    if type(item.get("bytes")) is not int or item["bytes"] < 0:
        raise ValueError("Invalid expected model size")
    sha, git = item.get("sha256"), item.get("git_sha1")
    if bool(sha) == bool(git):
        raise ValueError("Each file needs exactly one publisher checksum")
    if not re.fullmatch(r"[a-f0-9]{64}" if sha else r"[a-f0-9]{40}", sha or git):
        raise ValueError("Invalid publisher checksum")
    if "url" in item:
        url = urlsplit(item["url"])
        if (url.scheme != "https" or url.username or url.password or url.port or url.fragment or
                url.hostname not in ("civitai.com", "huggingface.co", "github.com", "storage.googleapis.com", "openaipublic.azureedge.net", "arena.kakaocdn.net")):
            raise ValueError("Expected a trusted HTTPS publisher download URL")
        if url.hostname == "huggingface.co":
            if not re.fullmatch(r"/[^/]+/[^/]+/resolve/[a-f0-9]{40}/.+", url.path) or url.query not in ("", "download=true"):
                raise ValueError("Hugging Face files must use a fixed commit without credentials")
        elif url.hostname == "github.com":
            if not re.fullmatch(r"/(sczhou/CodeFormer|xinntao/facexlib|AUTOMATIC1111/TorchDeepDanbooru|AUTOMATIC1111/stable-diffusion-webui|deepinsight/insightface)/releases/download/[^/]+/[^/]+", url.path) or url.query:
                raise ValueError("Expected a fixed official GitHub model release")
        elif url.hostname == "storage.googleapis.com":
            if url.path != "/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth" or not re.fullmatch(r"generation=[0-9]+", url.query):
                raise ValueError("Expected the fixed official BLIP object generation")
        elif url.hostname == "openaipublic.azureedge.net":
            if url.path != "/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt" or url.query:
                raise ValueError("Expected the official OpenAI CLIP model")
        elif url.hostname == "arena.kakaocdn.net":
            if url.path != "/brainrepo/models/karlo-public/v1.0.0.alpha/0b62380a75e56f073e2844ab5199153d/ViT-L-14_stats.th" or url.query:
                raise ValueError("Expected the fixed official Karlo statistics file")
        elif not re.fullmatch(r"/api/download/models/[0-9]+", url.path) or not re.fullmatch(r"fileId=[0-9]+", url.query):
            raise ValueError("Civitai files must use an explicit version and file ID")


def expected_hash(item):
    return item.get("sha256") or item["git_sha1"]


def new_hasher(item):
    if item.get("sha256"):
        return hashlib.sha256()
    value = hashlib.sha1()
    value.update(f"blob {item['bytes']}\0".encode())
    return value


def verify_file(path, item):
    if not path.is_file() or path.stat().st_size != item["bytes"]:
        return False
    digest = new_hasher(item)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == expected_hash(item)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".download-json-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_storage(drive_root):
    """The mount check must happen before creating any Drive-looking path."""
    mount = Path("/content/drive")
    if sys.platform != "linux" or not os.path.ismount(mount) or not (mount / "MyDrive").is_dir():
        raise ValueError("Mount Google Drive at /content/drive first. No destination folders were created")
    drive = Path(drive_root).expanduser().resolve()
    if not drive.is_relative_to(mount / "MyDrive") or drive == mount / "MyDrive":
        raise ValueError("Use a dedicated folder under /content/drive/MyDrive, such as ForgeColab")
    safe_target(drive, "models")
    safe_target(drive, "download-state")
    return drive


@contextmanager
def civitai_response(session, url, headers, api_key):
    """Follow HTTPS redirects explicitly; a CDN never receives the API key."""
    current = url
    for _ in range(8):
        parsed = urlsplit(current)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or
                parsed.password or parsed.port or parsed.fragment or "\\" in current or
                any(ord(c) < 32 for c in current)):
            raise IntegrityError("The publisher returned an unsafe download redirect")
        outgoing = dict(headers)
        if parsed.hostname == "civitai.com" and api_key:
            outgoing["Authorization"] = "Bearer " + api_key
        response = session.get(current, headers=outgoing, stream=True, timeout=(20, 30), allow_redirects=False)
        if response.status_code in (301, 302, 303, 307, 308):
            location = response.headers.get("Location")
            response.close()
            if not location:
                raise IntegrityError("The publisher returned an empty download redirect")
            current = urljoin(current, location)
            continue
        try:
            yield response
        finally:
            response.close()
        return
    raise IntegrityError("Too many publisher download redirects")


def download_file(item, models_dir, state_dir, retries=6, session_factory=requests.Session, cancelled=None, api_key=None):
    """Resume a byte prefix, verify the complete checksum, then publish atomically."""
    validate_item(item)
    target = safe_target(models_dir, item["path"])
    if target.exists():
        if not verify_file(target, item):
            raise IntegrityError("An existing destination does not match its publisher checksum; it was not replaced")
        emit("verified_existing", path=item["path"], bytes=item["bytes"])
        return {"path": item["path"], "bytes": item["bytes"], "status": "verified", "existing": True}
    identity = hashlib.sha256((item["path"] + expected_hash(item)).encode()).hexdigest()
    part = safe_target(state_dir, "parts/" + identity + ".partial")
    part.parent.mkdir(parents=True, exist_ok=True)
    if part.exists() and (not part.is_file() or part.stat().st_size > item["bytes"]):
        raise IntegrityError("An invalid partial download exists; it was retained for inspection")
    emit("downloading", path=item["path"], bytes=item["bytes"], resumed=part.stat().st_size if part.exists() else 0)
    session = session_factory()
    # Do not read .netrc, browser cookies, tokens, or inherited authentication.
    session.trust_env = False
    try:
        for attempt in range(retries):
            try:
                if cancelled is not None and cancelled.is_set():
                    raise RuntimeError("Download cancelled; partial file retained")
                current = part.stat().st_size if part.exists() else 0
                if current == item["bytes"] and part.exists():
                    if not verify_file(part, item):
                        raise IntegrityError("A completed partial file failed its publisher checksum")
                    break
                headers = {"Range": f"bytes={current}-", "Accept-Encoding": "identity", "User-Agent": "Forge-Colab-Model-Migration/1"}
                with civitai_response(session, item["url"], headers, api_key) as response:
                    if response.status_code in (401, 403):
                        raise AccessDenied("The selected publisher denied this file; check account access. No unverified substitute was used")
                    if response.status_code not in (200, 206):
                        raise requests.RequestException(f"Publisher returned HTTP {response.status_code}")
                    if response.status_code == 206:
                        expected_range = f"bytes {current}-{item['bytes'] - 1}/{item['bytes']}"
                        if response.headers.get("Content-Range") != expected_range:
                            raise IntegrityError("The server returned a different byte range or model size")
                    elif current:
                        # A CDN may ignore Range. Never append a full response to
                        # a prefix, which would silently corrupt the checkpoint.
                        emit("server_did_not_resume", path=item["path"], action="restart_partial_only")
                        current = 0
                    length = response.headers.get("Content-Length")
                    if length is not None and int(length) != item["bytes"] - current:
                        raise IntegrityError("The download length does not match the publisher metadata")
                    digest = new_hasher(item)
                    if current:
                        with part.open("rb") as previous:
                            for block in iter(lambda: previous.read(8 * 1024 * 1024), b""):
                                digest.update(block)
                    last_progress = time.monotonic()
                    with part.open("ab" if current else "wb") as stream:
                        for block in response.iter_content(chunk_size=1024 * 1024):
                            if cancelled is not None and cancelled.is_set():
                                raise RuntimeError("Download cancelled; partial file retained")
                            if not block:
                                continue
                            if current + len(block) > item["bytes"]:
                                raise IntegrityError("The response exceeds the expected file size")
                            stream.write(block)
                            digest.update(block)
                            current += len(block)
                            if time.monotonic() - last_progress >= 20:
                                stream.flush()
                                emit("progress", path=item["path"], received=current, total=item["bytes"])
                                last_progress = time.monotonic()
                        stream.flush()
                    if current != item["bytes"]:
                        raise requests.RequestException("Transfer ended before the expected size")
                    if digest.hexdigest() != expected_hash(item):
                        raise IntegrityError("The downloaded file failed its publisher checksum")
                break
            except (requests.RequestException, OSError):
                if attempt + 1 == retries:
                    raise RuntimeError("Transfer interrupted repeatedly; the partial file was saved for the next run") from None
                emit("retry", path=item["path"], attempt=attempt + 1)
                if cancelled is None:
                    time.sleep(min(2 ** attempt, 16))
                else:
                    cancelled.wait(min(2 ** attempt, 16))
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise IntegrityError("Another process created the destination; its file was not replaced")
        part.replace(target)
        emit("verified_download", path=item["path"], bytes=item["bytes"], checksum=expected_hash(item))
        return {"path": item["path"], "bytes": item["bytes"], "status": "verified", "existing": False}
    finally:
        session.close()


def manifest_entry(item):
    value = {"path": item["path"], "bytes": item["bytes"]}
    if item.get("sha256"):
        value["sha256"] = item["sha256"]
    # stage_models accepts SHA256 only. Git blob files are converted to SHA256
    # after their official Git checksum has been verified.
    return value


def extract_verified_archive(item, drive):
    """Extract only named regular files, verified independently before publishing."""
    archive = safe_target(drive / "download-state", item["path"])
    if not verify_file(archive, item):
        raise IntegrityError("The model archive failed verification")
    verified = []
    with zipfile.ZipFile(archive) as bundle:
        for member in item["members"]:
            validate_item(member)
            target = safe_target(drive / "models", member["path"])
            if target.exists():
                if not verify_file(target, member):
                    raise IntegrityError("An existing archive model differs; it was not replaced")
                verified.append("models/" + member["path"])
                continue
            matching = [entry for entry in bundle.infolist() if PurePosixPath(entry.filename).name == PurePosixPath(member["path"]).name]
            if len(matching) != 1:
                raise IntegrityError("The model archive has missing or duplicate members")
            info = matching[0]
            safe_relative(info.filename)
            if info.is_dir() or (info.external_attr >> 16) & 0o170000 == 0o120000 or info.file_size != member["bytes"]:
                raise IntegrityError("The model archive contains an unexpected member")
            identity = hashlib.sha256((member["path"] + member["sha256"]).encode()).hexdigest()
            temporary = safe_target(drive / "download-state", "extract/" + identity + ".partial")
            temporary.parent.mkdir(parents=True, exist_ok=True)
            try:
                digest, received = hashlib.sha256(), 0
                with bundle.open(info) as source, temporary.open("wb") as output:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        received += len(block)
                        if received > member["bytes"]:
                            raise IntegrityError("An archive member exceeds its expected length")
                        output.write(block)
                        digest.update(block)
                if received != member["bytes"] or digest.hexdigest() != member["sha256"]:
                    raise IntegrityError("An archive model failed its exact-file checksum")
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise IntegrityError("Another process created the archive destination; it was not replaced")
                temporary.replace(target)
            finally:
                if temporary.exists():
                    temporary.unlink()
            verified.append("models/" + member["path"])
            emit("verified_archive_model", path=member["path"], bytes=member["bytes"])
    return verified


def merge_verified_manifest(drive, successful, storage="models"):
    path = safe_target(drive, storage + "-manifest.json")
    current = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {"files": []}
    if not isinstance(current, dict) or not isinstance(current.get("files"), list):
        raise ValueError("Existing models-manifest.json is malformed; it was not replaced")
    entries = {safe_relative(item["path"]): item for item in current["files"]}
    all_files = CATALOG["files"] + [member for archive in CATALOG.get("archives", []) for member in archive["members"]]
    source = {item["storage"] + "/" + item["path"]: item for item in all_files}
    for key in successful:
        item = source[key]
        if item["storage"] != storage:
            continue
        relative = item["path"]
        value = manifest_entry(item)
        if "sha256" not in value:
            value["sha256"] = hashlib.sha256((drive / storage / relative).read_bytes()).hexdigest()
        entries[relative] = value
    atomic_json(path, {**current, "files": [entries[key] for key in sorted(entries)]})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", type=Path, default=Path("/content/drive/MyDrive/ForgeColab"))
    parser.add_argument("--download", action="store_true", help="Download verified files to Drive; otherwise print a read-only plan")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--group", choices=("checkpoint", "auxiliary", "embeddings", "legacy-qwen", "qwen-bf16"), action="append")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be 1 through 4")
    files = [item for item in CATALOG["files"] + CATALOG.get("archives", []) if not args.group or item["group"] in args.group]
    for item in files:
        validate_item(item)
    plan = {"download_files": len(files), "bytes": sum(item["bytes"] for item in files),
            "unavailable_profiles": CATALOG["unavailable_profiles"], "destination": str(args.drive_root),
            "default_checkpoint": CATALOG["default_checkpoint"], "limitations": CATALOG["limitations"],
            "local_derivations": CATALOG.get("local_derivations", [])}
    emit("plan", **plan)
    if not args.download:
        return 0
    drive = validate_storage(args.drive_root)
    api_key = os.environ.get("CIVITAI_API_KEY", "").strip()
    if any(item["provider"] == "civitai" for item in files) and (not api_key or any(ord(c) < 33 or ord(c) > 126 for c in api_key)):
        raise ValueError("Set CIVITAI_API_KEY using a hidden Colab input before downloading; never put it in a URL or command")
    state = safe_target(drive, "download-state")
    destinations = {storage: safe_target(drive, storage) for storage in ("models", "embeddings", "download-state")}
    for destination in destinations.values():
        destination.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    # Drive's FUSE quota report may not be exact, but a definite shortage should
    # stop before any weight transfer. Existing files are accounted separately.
    needed = 0
    for item in files:
        needed += sum(member["bytes"] for member in item.get("members", []) if not (drive / "models" / member["path"]).is_file())
        if (destinations[item["storage"]] / item["path"]).is_file():
            continue
        identity = hashlib.sha256((item["path"] + expected_hash(item)).encode()).hexdigest()
        part = safe_target(state, "parts/" + identity + ".partial")
        resumed = part.stat().st_size if part.is_file() else 0
        needed += max(0, item["bytes"] - resumed)
    if shutil.disk_usage(drive).free < needed + 1024**3:
        raise ValueError("The mounted Drive reports insufficient space for the selected models")
    report = {"verified": [], "failed": [], "unavailable_profiles": CATALOG["unavailable_profiles"],
              "limitations": CATALOG["limitations"], "default_checkpoint": CATALOG["default_checkpoint"],
              "quantization_pending": CATALOG.get("local_derivations", [])}
    report_path = safe_target(state, "download-report.json")
    import fcntl
    cancelled = threading.Event()
    with safe_target(state, "download.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("A model download is already running for this Drive folder") from None
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(download_file, item, destinations[item["storage"]], state, cancelled=cancelled, api_key=api_key): item for item in files}
            try:
                for future in as_completed(futures):
                    item = futures[future]
                    try:
                        future.result()
                        if item.get("members"):
                            report["verified"].extend(extract_verified_archive(item, drive))
                            merge_verified_manifest(drive, report["verified"], "models")
                        else:
                            report["verified"].append(item["storage"] + "/" + item["path"])
                            merge_verified_manifest(drive, report["verified"], item["storage"])
                    except Exception as error:
                        # Do not persist redirect URLs, cookies, or exception payloads.
                        reason = str(error) if isinstance(error, (IntegrityError, AccessDenied)) else "Transfer did not complete; retry to resume"
                        report["failed"].append({"path": item["path"], "kind": type(error).__name__, "reason": reason})
                        emit("file_failed", path=item["path"], kind=type(error).__name__, reason=reason)
                    atomic_json(report_path, report)
            except KeyboardInterrupt:
                cancelled.set()
                for future in futures:
                    future.cancel()
                emit("cancelled", note="Partial files remain available for the next run")
                raise
        report["selected_downloads_complete"] = not report["failed"]
        report["entire_original_collection_complete"] = False
        atomic_json(report_path, report)
        emit("finished", files_verified=len(report["verified"]), failed=len(report["failed"]), unavailable_profiles=report["unavailable_profiles"],
             note="Only selected catalog files are covered. Qwen NF4 and INT8 need local quantization. Image generation is not tested by this download.")
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
