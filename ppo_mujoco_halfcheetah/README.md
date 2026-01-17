[![License](https://img.shields.io/github/license/giansimone/ppo-mujoco-halfcheetah)](https://github.com/giansimone/ppo-mujoco-halfcheetah/blob/main/LICENSE)

# Proximal Policy Optimisation (PPO) for MuJoCo Half Cheetah Environment

A PyTorch implementation of the Proximal Policy Optimisation (PPO) algorithm to train an agent to play with the Half Cheetah environment from MuJoCo.

## Project Structure

```bash
ppo_mujoco_halfcheetah/
├── __init__.py
├── agent.py       # PPO implementation (Actor/Critic)
├── buffer.py      # Rollout Buffer
├── config.yaml    # Training hyperparameters
├── environment.py # Gym environment wrappers
├── enjoy.py       # Evaluation script
├── export.py      # Hugging Face export script
├── model.py       # PyTorch Network definitions
├── train.py       # Main training loop
└── utils.py

## Usage

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
