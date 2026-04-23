import torch
import transformers
import trl
import deepspeed
import pkg_resources

print("="*50)
print("当前环境版本信息（核心依赖）")
print("="*50)
print(f"PyTorch 版本: {torch.__version__}")
print(f"Transformers 版本: {transformers.__version__}")
print(f"TRL 版本: {trl.__version__}")
print(f"DeepSpeed 版本: {deepspeed.__version__}")
print(f"Python 版本: {pkg_resources.get_distribution('python').version}")
print("="*50)

# 自动判断问题
ds_version = deepspeed.__version__
if ds_version >= "0.14.0":
    print("❌ 问题定位：DeepSpeed 版本过高，与 GRPOTrainer 分布式不兼容！")
    print("✅ 推荐安装：deepspeed==0.12.6")
elif ds_version in ["0.12.6", "0.13.1"]:
    print("✅ DeepSpeed 版本为兼容版本，报错非版本导致")
else:
    print("⚠️  DeepSpeed 版本未验证，推荐切换到 0.12.6")
print("="*50)