LoRA-Diffusion: Parameter-Efficient Lifelong Learning for Robotic Manipulation 
A parameter-efficient lifelong learning framework for robotic manipulation, built on the LIBERO benchmark. It uses a task-level expert pool with frozen shared backbones and LoRA adapters to avoid catastrophic forgetting with minimal parameter overhead.
📋 Overview
This repository implements a novel lifelong learning framework that:
Maintains a pool of lightweight task experts instead of full-model fine-tuning.
Uses LoRA (Low-Rank Adaptation) to handle task distribution shifts with ~10k new parameters per task.
Integrates a pluggable diffusion teacher distillation interface.
Combines distillation loss and behavior cloning (BC) loss for stable training.
Is fully compatible with the native LIBERO training/evaluation pipeline.

✨ Key Features
Task-Level Expert Pool: Each task has an independent expert to avoid catastrophic forgetting.
Parameter-Efficient LoRA Adapters: Freeze the shared backbone and only update low-rank LoRA heads.
Pluggable Diffusion Teacher: Interface to distill knowledge from high-capacity diffusion models.
Hybrid Training Objective: Balances distillation loss and BC loss with configurable weights.
Non-Intrusive LIBERO Compatibility: Works seamlessly with the original LIBERO codebase.

🏗️ Framework Overview
The framework follows a three-layer design:
Algorithm Layer: Dynamic expert pool for task management and switching.
Model Layer: Frozen shared backbone + task-specific LoRA adapters.
Training Layer: Hybrid distillation + BC loss with optional routing weights.

📂 Code Structure
<pre>
```text
LoRA-Diffusion/
├── benchmark_scripts/ # LIBERO evaluation & utility scripts
├── images/ # Docs & visualization assets
├── libero/ # Core LIBERO codebase + our algorithm
│ ├── algos/ # Algorithm implementations
│ │ ├── expert_pool.py # [Core] ExpertPoolAlgo main logic
│ │ ├── expert_policy.py # Expert policy with LoRA heads
│ │ ├── lora_adapter.py # Standalone LoRA module
│ │ └── diffusion_teacher.py # Diffusion teacher interface
│ ├── configs/ # Hydra configuration files
│ └── [Other LIBERO modules]
├── notebooks/ # Jupyter notebooks for debugging
├── scripts/ # Data download & helper scripts
├── templates/ # Project templates
├── .gitignore # Git ignore rules
├── LICENSE # MIT License
├── requirements.txt # Python dependencies
├── run.sh # One-click training script
└── setup.py # Project installation script
```
</pre>

🛠️ Installation
Prerequisites
Ubuntu 20.04+
Python 3.8+
PyTorch 1.12+
MuJoCo 2.1.0 (required by LIBERO)

Steps:
1.Clone the repository:
```
git clone https://github.com/Haidong-Huang/LoRA-Diffusion.git
cd LoRA-Diffusion
```
2.Install dependencies:
```
pip install -r requirements.txt
pip install -e .
```
3.Download LIBERO datasets (if not already downloaded):
```
python scripts/download_libero_data.py
```
🚀 Quick Start
Train
Use the provided script for one-click training:
```
bash run.sh
```
Or manually launch with Hydra:
```
python main.py --config-path=libero/configs/lifelong --config-name=your_config.yaml algo=expert_pool
```
📌 Current Status
✅ Implemented
Core expert pool framework with task switching.
LoRA-based expert policy with frozen backbone.
Hybrid distillation + BC loss training.
Full compatibility with LIBERO.

🚧 In Progress
Full implementation of trainable DiffusionTeacherPolicy.
Complete Hydra configs for ExpertPoolAlgo.
Unified checkpoint naming/loading.
Systematic ablation studies.
Full benchmark results on LIBERO.

📜 Citation
If you use this code, please cite:
```
@inproceedings{liu2023libero,
  title={LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  author={Liu, Bo and Zhu, Yifeng and Feng, Yihao and Liu, Qiang and Wu, Ying Nian and Zhu, Yuke},
  booktitle={Conference on Robot Learning},
  pages={1924--1935},
  year={2023},
  organization={PMLR}
}
```
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

💬 Contact
For questions or discussions, please open an issue or contact Haidong-Huang on GitHub.

