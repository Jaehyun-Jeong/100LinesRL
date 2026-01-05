# 100LinesRL

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Code Size](https://img.shields.io/github/languages/code-size/jaehyun-jeong/100LinesRL)
![Stars](https://img.shields.io/github/stars/jaehyun-jeong/100LinesRL?style=social)

Implementations of basic RL algorithms with minimal lines of codes! (PyTorch based)

Inspired by [minimalRL](https://github.com/seungeunrho/minimalRL).

* Each algorithm is complete within a single file.

* Length of each file is up to 100 lines of codes.

## Algorithms
1. [DQN](https://github.com/jaehyun-jeong/100LinesRL/blob/master/dqn.py) (98 lines, including replay memory and target network)
2. [PPO](https://github.com/jaehyun-jeong/100LinesRL/blob/master/ppo.py) (100 lines, including GAE)
3. [SAC](https://github.com/jaehyun-jeong/100LinesRL/blob/master/sac.py) (99 lines)
4. [TD3](https://github.com/jaehyun-jeong/100LinesRL/blob/master/td3.py) (100 lines)

## Dependencies
1. PyTorch
2. gymnasium

## Usage
```bash
# Works only with Python 3.
# e.g.
python3 dqn.py
python3 ppo.py
python3 sac.py
python3 td3.py
```
