# 模型权重

模型文件不提交到 GitHub；程序不会自动下载。请按官方要求获取并遵循对应许可。

- SAM3：通过 [Meta 官方仓库](https://github.com/facebookresearch/sam3) 的说明访问 [facebook/sam3](https://huggingface.co/facebook/sam3)，按要求申请权限并下载 `sam3.pt`。
- SAM2.1 Hiera Large：[Meta 官方仓库](https://github.com/facebookresearch/sam2#download-checkpoints)；[官方权重下载](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)。

在项目根目录建立：

```text
weights/sam_weights/sam3.pt
weights/sam_weights/sam2.1_hiera_large.pt
```

SAM3 图片标注需要第一个文件；当前 SAM3 初始化 + SAM2.1 视频跟踪需要两个文件。下载后重启程序。

以下为本项目验收时使用的文件，可用于核对：

| 文件 | 字节数 | SHA256 |
| --- | ---: | --- |
| sam3.pt | 3450062241 | `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e` |
| sam2.1_hiera_large.pt | 898083611 | `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318` |

运行方法与模型能力边界见 [README](../README.md) 和 [操作说明](../RUN_SAM3.md)。
