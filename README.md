# sam3-label

基于 [luohuabuxiema/LabelPaw](https://github.com/luohuabuxiema/LabelPaw) 的本地图片标注与单目标视频跟踪版本，由 [paulfengling](https://github.com/paulfengling) 整理维护。保留 LabelPaw 的界面与标注格式，补充 Mac CPU 兼容、中文提示词、鼠标拖框、结果微调和视频导出。

完整操作与提示词扩展见 [RUN_SAM3.md](RUN_SAM3.md)。[上游中文说明](README_zh-CN.md) 与 [上游英文说明](docs/README_upstream.md) 保留原作者的介绍、历史安装步骤和引用；当前版本请按本页安装。

## 当前功能与边界

- **图片**：SAM3 文本、点选和视觉拖框分割；支持矩形、多边形、旋转框及手动微调，保存 JSON / YOLO / XML。中文常用目标和胶带短语会转换为英文模型提示，标注标签保留原文。
- **视频**：SAM3 在参考帧确定一个目标，再由 **SAM2.1 Hiera Large** 向前、向后传播，固定 `track_id=1`。这是 SAM3 初始化与 SAM2 视频跟踪的混合流程。输出轮廓视频、每帧 PNG 掩码和 JSON，可选 YOLO 分割；导出视频不含音轨。
- **选定目标**：视频可在任意参考帧左键拖框。未拖框时使用文字，只有恰好找到一个实例才开始跟踪；多个或零个实例会要求重新框选。
- **模型能力**：颜色和关系短语的效果依赖画面。跟踪同一胶带不会逐帧判断它是否仍被手拿着；完全遮挡、出画或快速运动后可能漂移，可换清晰参考帧重新框选生成新结果。

本版本已在 **macOS Apple Silicon、Python 3.10、CPU** 下完成真实模型验收。程序在 CUDA 可用时选择 CUDA，否则使用 CPU；Mac 当前使用 CPU，不要求 CUDA 或 Triton。图片初始化和视频逐帧计算需要等待。

## 安装与运行

建议使用 Python **3.10–3.12**。仓库内含保留兼容修复的 `sam2/sam2/`、`sam3/sam3/` 核心源码、配置和 BPE 词表，无需另外安装这两个仓库。

macOS / Linux，在项目根目录执行（`python3` 应指向上述版本）：

```bash
git clone https://github.com/paulfengling/sam3-label.git
cd sam3-label
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py
```

macOS 也可在创建环境后双击 `start_labelpaw.command`。Windows：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

Windows 可使用 `start_labelpaw.bat`。NVIDIA 环境应先按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 安装与驱动匹配的 `torch==2.10.0`、`torchvision==0.25.0` CUDA 版本，再安装其余依赖。

## 模型权重

权重不上传到此仓库，程序不会自动下载。请自行从 Meta 官方渠道获取，并按对应许可使用：

- **SAM3**：从 [Meta SAM3 官方仓库](https://github.com/facebookresearch/sam3) 指向的 [facebook/sam3](https://huggingface.co/facebook/sam3) 获取 `sam3.pt`，按页面要求申请访问。此版本使用 SAM3 权重。
- **SAM2.1 Large**：从 [Meta SAM2 官方仓库](https://github.com/facebookresearch/sam2#download-checkpoints) 获取 [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)。

创建目录并保持文件名如下。图片 SAM3 需要第一个文件；混合视频跟踪需要两个文件：

```text
weights/
└── sam_weights/
    ├── sam3.pt
    └── sam2.1_hiera_large.pt
```

## 使用与演示

图片：打开图片目录，选择 SAM3 和标注形状，输入 `人`、`手掌`、`白色胶带` 或英文短语；也可直接在目标外拖框。选择已有标注后可调整位置、顶点、矩形边角及旋转框角度。详细操作见 [RUN_SAM3.md](RUN_SAM3.md)。

视频：点击主界面的“视频目标跟踪”，打开视频、选择参考帧、拖框指定目标，然后开始跟踪并导出。命令行也可运行：

```bash
.venv/bin/python track_video.py \
  --video examples/demo/synthetic_tape_occlusion.mp4 \
  --output outputs/video_tracking \
  --box 154 207 641 273 \
  --prompt "白色胶带"
```

小型演示：[胶带测试图](examples/demo/tape_scene.png)、[文字分割对比](examples/demo/tape_prompt_preview.jpg)、[合成测试视频](examples/demo/synthetic_tape_occlusion.mp4)、[跟踪预览](examples/demo/tracking_preview.jpg)、[轮廓结果视频](examples/demo/overlay.mp4)。

演示图片由生成模型制作；视频由生成图片和分层运动合成，包含中途部分遮挡，并在画面中标明来源。本机 CPU 对 768×512、30 帧短片的这次真实 SAM3→SAM2 推理约耗时 132 秒，仅用于说明操作和检查流程。

## 验证

当前 **138 项轻量回归测试通过**，覆盖图片标注、CPU 兼容、几何编辑、视频传播接口、取消和导出。这些测试不加载权重：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

放好本地权重后可运行真实模型检查：

```bash
.venv/bin/python smoke_sam2.py
.venv/bin/python smoke_sam3.py
```

## 来源与许可证

LabelPaw 原作者为 **luohuabuxiema**，原始项目：[LabelPaw: Intelligent image annotation system](https://github.com/luohuabuxiema/LabelPaw)。本分支保留原作者归属、引用和项目 [GPL-3.0 许可证](LICENSE)。上游完整介绍和 BibTeX 见 [原英文说明](docs/README_upstream.md#citation)。

第三方组件保留各自许可证：

- SAM2：Meta 的 [Apache-2.0 许可证](sam2/LICENSE)，以及 [LICENSE_cctorch](sam2/LICENSE_cctorch)。
- SAM3：Meta 的 [SAM License](sam3/LICENSE)，仓库同时保留 [SAM_LICENSE.txt](SAM_LICENSE.txt)。
- 上游集成的 Ultralytics：参见 [官方项目与许可](https://github.com/ultralytics/ultralytics)。

模型、第三方源码与本项目分别遵循其对应许可。SAM 研究引用保留在上游说明中：[SAM3 论文](https://arxiv.org/abs/2511.16719)、[SAM2 论文](https://arxiv.org/abs/2408.00714)。
