#!/bin/bash

# Activate conda environment
# Try to activate conda, fallback to direct python path if conda not available
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate libero 2>/dev/null || {
        # If conda activate fails, use the conda environment's python directly
        export PATH="/root/miniconda3/envs/libero/bin:$PATH"
    }
else
    # Use conda environment python directly
    export PATH="/root/miniconda3/envs/libero/bin:$PATH"
fi

# Set default values if not provided
SEED=${SEED:-0}
POLICY=${POLICY:-bc_rnn_policy}
ALGO=${ALGO:-base}
GPU_ID=${GPU_ID:-0}

# Set EGL environment variables for offscreen rendering
export CUDA_VISIBLE_DEVICES=$GPU_ID
export MUJOCO_EGL_DEVICE_ID=$GPU_ID
export PYOPENGL_PLATFORM=egl

# Disable cuDNN globally for RTX 5090 compatibility (sm_120 not fully supported)
# This forces PyTorch to use fallback implementations
export PYTORCH_CUDNN_V8_API_ENABLED=0

# Run the training script (force CPU since current PyTorch build doesn't support this GPU)
# Also disable DataLoader multiprocessing (num_workers=0) to avoid h5py pickling issues
python libero/lifelong/main.py seed=$SEED \
                               benchmark_name=LIBERO_SPATIAL \
                               policy=$POLICY \
                               lifelong=$ALGO \
                               device=cpu \
                               train.num_workers=0 \
                               eval.num_workers=0