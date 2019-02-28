#!/usr/bin/env bash
cd ./utils/

CUDA_PATH=/usr/local/cuda/

python build.py build_ext --inplace

cd ..
