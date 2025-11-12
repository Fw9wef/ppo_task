[![License](https://img.shields.io/github/license/giansimone/ppo-mujoco-halfcheetah)](https://github.com/giansimone/ppo-mujoco-halfcheetah/blob/main/LICENSE)

# Proximal Policy Optimisation (PPO) for MuJoCo Half Cheetah Environment

A PyTorch implementation of the Proximal Policy Optimisation (PPO) algorithm to train an agent to play with the Half Cheetah environment from MuJoCo.

## Installation

You can clone the repository and install the required dependencies using Poetry or pip. 

Using Poetry:

```bash
git clone https://github.com/giansimone/ppo-mujoco-halfcheetah.git
cd ppo-mujoco-halfcheetah
poetry install
poetry shell
```

Alternatively, you can clone the repository and install the dependencies locally using pip:

```bash
git clone https://github.com/giansimone/ppo-mujoco-halfcheetah.git
cd ppo-mujoco-halfcheetah
python -m venv venv
source venv/bin/activate
pip install -e .
```

## Project Structure

```bash
ppo-mujoco-halfcheetah/
├── ppo_mujoco_halfcheetah/
│   ├── __init__.py
│   ├── agent.py
│   ├── buffer.py
│   ├── config.yaml
│   ├── enjoy.yaml
│   ├── environment.py
│   ├── export.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── .gitignore
├── LICENSE
├── README.md
└── pyproject.toml
```

## Quick Start

### Training

Train a PPO agent with the default configuration:

```bash
python -m ppo_mujoco_halfcheetah.train
```

### Configuration

Edit ```config.yaml``` to customise training parameters such as learning rate, number of episodes, and more.

```yaml
#Environment
env_id: HalfCheetah-v5
num_envs: 8

#Network Architecture
hidden_dim: 256

#Training
total_timesteps: 1_000_000
n_steps: 1024
batch_size: 64

#PPO Agent
learning_rate: 0.0003
gamma: 0.99
gae_lambda: 0.95
clip_epsilon: 0.2
value_coef: 0.5
entropy_coef: 0.01
max_grad_norm: 0.5
ppo_epochs: 10

#Logging
log_dir: runs/

#System
seed: 42
```

### Enjoying a Trained Agent

Watch a trained agent by running the enjoy script:

```bash
python -m ppo_mujoco_halfcheetah.enjoy \
    --artifact-path runs/ppo_YYYY-MM-DD_HHhMMmSSs/final_model.pt \
    --num-episodes 1
```
### Exporting to Hugging Face Hub

Share your trained model on the Hugging Face Hub:

```bash
python -m ppo_mujoco_halfcheetah.export \
    --username YOUR_HF_USERNAME \
    --repo-name ppo-mujoco-halfcheetah \
    --artifact-path runs/ppo_YYYY-MM-DD_HHhMMmSSs/final_model.pt \
    --movie-fps 30 
```

This will:

- Create a repository on Hugging Face Hub.
- Upload the model weights, configuration, and evaluation results.
- Generate and upload a replay movie.
- Create a model card with usage instructions.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.