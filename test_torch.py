# test_torch.py
import torch

print("="*50)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch built‑with CUDA: {torch.version.cuda}")

# 测试CPU张量
cpu_tensor = torch.randn(2, 4)
print("\n✅ CPU张量测试成功\n", cpu_tensor)

# 如果GPU可用，测试GPU张量
if torch.cuda.is_available():
    gpu_tensor = torch.randn(2,4).cuda()
    print("\n✅ GPU张量测试成功\n", gpu_tensor)

print("\n===== Torch 全部测试完成 =====")
