[![License](https://img.shields.io/github/license/giansimone/ppo-mujoco-halfcheetah)](https://github.com/giansimone/ppo-mujoco-halfcheetah/blob/main/LICENSE)

# Proximal Policy Optimisation (PPO) for MuJoCo Half Cheetah Environment

A PyTorch implementation of the Proximal Policy Optimisation (PPO) algorithm to train an agent to play with the Half Cheetah environment from MuJoCo.

## Installation

You can clone the repository and install the required dependencies using Poetry or pip. This project requires **Python 3.13**.

### Using Poetry (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/giansimone/ppo-mujoco-halfcheetah.git
cd ppo-mujoco-halfcheetah

# 2. Initialize environment and install dependencies
poetry env use python3.13
poetry install

# 3. Activate the shell
eval $(poetry env activate)
```

### Using Pip

```bash
# 1. Clone the repository
git clone https://github.com/giansimone/ppo-mujoco-halfcheetah.git
cd ppo-mujoco-halfcheetah

# 2. Create and activate virtual environment
python3.13 -m venv venv
source venv/bin/activate

# 3. Install package in editable mode
pip install -e .
```

## Project Structure

```bash
ppo-mujoco-halfcheetah/
├── ppo_mujoco_halfcheetah/
│   ├── __init__.py
│   ├── agent.py       # PPO implementation (Actor/Critic)
│   ├── buffer.py      # Rollout Buffer
│   ├── config.yaml    # Training hyperparameters
│   ├── environment.py # Gym environment wrappers
│   ├── enjoy.py       # Evaluation script
│   ├── export.py      # Hugging Face export script
│   ├── model.py       # PyTorch Network definitions
│   ├── train.py       # Main training loop
│   └── utils.py
├── .gitignore
├── LICENSE
├── README.md
└── pyproject.toml
```

## Usage

Ensure you are in the ```ppo_mujoco_halfcheetah``` source directory where ```config.yaml``` is located before running these commands.

```bash
cd ppo_mujoco_halfcheetah
```

### Training

Train a PPO agent with the default configuration.

**Note:** The Replay Buffer pre-allocates memory. Ensure your system has at least 8GB of RAM available.

```bash
python -m train
```

### Configuration

Edit ```config.yaml``` to customise training parameters.

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
python -m enjoy \
    --artifact runs/ppo_HalfCheetah-v5_YYYY-MM-DD_HHhMMmSSs/final_model.pt \
    --num-episodes 5
```
### Exporting to Hugging Face Hub

Share your trained model, config, and a replay video to the Hugging Face Hub.

```bash
python -m export \
    --username YOUR_HF_USERNAME \
    --repo-name ppo-mujoco-halfcheetah \
    --artifact-path runs/ppo_HalfCheetah-v5_YYYY-MM-DD_HHhMMmSSs/final_model.pt \
    --movie-fps 30 \
    --n-eval 10
```

This will automatically:

- Upload the model weights and config.

- Generate a model card with evaluation metrics (Mean Reward +/- Std).

- Record and upload a video of the agent.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.