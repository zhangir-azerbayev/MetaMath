BASE_DIR=/replace/with/repo/base/dir # Replace with path this repository
TRAIN_FILE=/path/to/MetaMathQA/json # Path to MetaMathQA-395k dataset in json form

# System-specific configuration: environment, CUDA, etc.

source /home/hailey81/miniconda3/bin/activate metainstruct

which python

export LD_LIBRARY_PATH=/home/hailey81/miniconda3/envs/metainstruct/lib/
export PATH=/home/hailey81/cuda_install/bin:$PATH

ln -s /home/hailey81/miniconda3/envs/metainstruct-updated/bin/gcc/ ~/.local/bin/gcc
export PATH=$HOME/.local/bin:$PATH
