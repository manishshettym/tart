# tart init specfically for github actions
# for your own use, please follow the instructions as in tart.readthedocs.io

pip install https://download.pytorch.org/whl/cpu/torch-1.13.0%2Bcpu-cp310-cp310-linux_x86_64.whl
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.13.0+cpu.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.13.0+cpu.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.13.0+cpu.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.13.0+cpu.html
pip install torch_geometric
pip install deepsnap
pip install transformers