# setup script for installing deepsnap and torch_geometric

# GPU (assumes Cuda 11.6, Python 3.9, and PyTorch 1.13.0)
# https://download.pytorch.org/whl/torch_stable.html
# https://pypi.org/project/torch-scatter/

# pip install https://download.pytorch.org/whl/cu116/torch-1.13.0%2Bcu116-cp39-cp39-linux_x86_64.whl
# pip install deepsnap
# pip install transformers
# pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
# pip install torch-geometric

# CPU
# https://stackoverflow.com/questions/65860764/pytorch-torch-sparse-installation-without-cuda

pip install https://download.pytorch.org/whl/cpu/torch-1.13.0%2Bcpu-cp310-cp310-linux_x86_64.whl
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.13.0+cpu.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.13.0+cpu.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.13.0+cpu.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.13.0+cpu.html
pip install torch_geometric
