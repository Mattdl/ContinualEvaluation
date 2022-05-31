#!/bin/bash

# Copyright (c) 2022. Matthias De Lange (KU Leuven).
# Copyrights licensed under the MIT License. All rights reserved.
# See the accompanying LICENSE file for terms.
#
# Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
# publicly available at https://arxiv.org/abs/2205.13452

CONDA_ENV_NAME="continual-eval"
python="3.8"
cuda_version="11.3" # Select a CUDA version between 9.2,10.1,10.2,11.3,none
pytorch_version="1.8.1"
torchvision_version="0.9.1"

while test $# -gt 0; do
         case "$1" in
              --python)
                  shift
                  python=$1
                  shift
                  ;;
              --cuda_version)
                  shift
                  cuda_version=$1
                  shift
                  ;;
              *)
                 echo "$1 is not a recognized flag! Use --python and/or --cuda_version."
                 exit 1;
                 ;;
        esac
done  

echo "python version : $python";
echo "cuda version : $cuda_version";

if ! [[ "$python" =~ ^(3.6|3.7|3.8)$ ]]; then
    echo "Select a python version between 3.6, 3.7, 3.8"
    exit 1
fi

if ! [[ "$cuda_version" =~ ^(9.2|10.1|10.2|11.3|"none")$ ]]; then
    echo "Select a CUDA version between 9.2,10.1,10.2,11.3,none"
    exit 1
fi

conda create -n "$CONDA_ENV_NAME" python=$python -c conda-forge
conda activate "$CONDA_ENV_NAME"
if [[ "$cuda_version" = "none" ]]; then 
    conda install pytorch=$pytorch_version torchvision=$torchvision_version cpuonly -c pytorch
else 
    conda install pytorch=$pytorch_version torchvision=$torchvision_version cudatoolkit=$cuda_version -c pytorch
fi
conda env update --file environment.yml

