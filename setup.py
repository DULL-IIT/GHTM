import torch
import subprocess

print("Checking system...")

if torch.cuda.is_available():
    cuda = torch.version.cuda
    print(f"GPU detected (CUDA {cuda})")

    if cuda.startswith("11.8"):
        url = "https://data.pyg.org/whl/torch-2.7.0+cu118.html"
    elif cuda.startswith("12.6"):
        url = "https://data.pyg.org/whl/torch-2.7.0+cu126.html"
    elif cuda.startswith("12.8"):
        url = "https://data.pyg.org/whl/torch-2.7.0+cu128.html"
    else:
        print(f"Unsupported PyTorch version for your CUDA. You can try installing another PyTorch version that is compatible with your CUDA version: {cuda} or you can use CPU")
        exit()

    subprocess.run([
        "pip", "install",
        "torch-scatter", "torch-sparse", 'torch-cluster', "torch-spline-conv", "torch-geometric",
        "-f", url
    ])
else:
    print("No GPU detected. Installing CPU version...")

    subprocess.run([
        "pip", "install",
        "torch-scatter", "torch-sparse", "torch-geometric",
        "-f", "https://data.pyg.org/whl/torch-2.7.0+cpu.html"
    ])
    