pip install -r requirements.txt
# https://download.pytorch.org/whl/torch_stable.html
pip install https://download.pytorch.org/whl/cu116/torch-1.13.0%2Bcu116-cp39-cp39-linux_x86_64.whl
pip install deepsnap
pip install transformers
# https://pypi.org/project/torch-scatter/
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
pip install torch-geometric

# CPU
# https://stackoverflow.com/questions/65860764/pytorch-torch-sparse-installation-without-cuda
# torch-scatter -f https://pytorch-geometric.com/whl/torch-1.13.0+cpu.html
# torch-sparse -f https://pytorch-geometric.com/whl/torch-1.13.0+cpu.html
# torch-cluster -f https://pytorch-geometric.com/whl/torch-1.13.0+cpu.html
# torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.13.0+cpu.html
# torch_geometric

pip install .
python setup.py develop