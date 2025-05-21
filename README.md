# Deep Q-Learning (DQN) for GridWorld

![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-1.8%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

This project implements a Deep Q-Network (DQN) agent to solve the GridWorld problem using PyTorch. The agent learns to navigate from the start position to the goal position in a customizable grid environment.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/thealper2/pytorch-dqn.git
cd pytorch-dqn
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
python3 main.py train
python3 main.py evaluate
```

## Usage

```bash
 Usage: main.py train [OPTIONS]

 Train a DQN agent on the GridWorld environment.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --width                                INTEGER  Width of the grid [default: 5]                           │
│ --height                               INTEGER  Height of the grid [default: 5]                          │
│ --obstacle-density                     FLOAT    Density of obstacles in the grid [default: 0.2]          │
│ --episodes                             INTEGER  Number of episodes to train for [default: 1000]          │
│ --buffer-size                          INTEGER  Size of the replay buffer [default: 10000]               │
│ --batch-size                           INTEGER  Size of the training batch [default: 64]                 │
│ --gamma                                FLOAT    Discount factor [default: 0.99]                          │
│ --epsilon-start                        FLOAT    Initial epsilon value [default: 1.0]                     │
│ --epsilon-end                          FLOAT    Final epsilon value [default: 0.05]                      │
│ --epsilon-decay                        FLOAT    Decay rate for epsilon [default: 0.995]                  │
│ --learning-rate                        FLOAT    Learning rate [default: 0.001]                           │
│ --target-update                        INTEGER  Number of episodes between target network updates        │
│                                                 [default: 10]                                            │
│ --model-path                           TEXT     Path to save the model [default: models/dqn_model.pt]    │
│ --results-dir                          TEXT     Directory to save results [default: results]             │
│ --evaluate            --no-evaluate             Evaluate the model after training [default: evaluate]    │
│ --eval-episodes                        INTEGER  Number of episodes to evaluate for [default: 10]         │
│ --seed                                 INTEGER  Random seed [default: None]                              │
│ --help                                          Show this message and exit.                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (git checkout -b feature-branch)
3. Commit your changes (git commit -am 'Add new feature')
4. Push to the branch (git push origin feature-branch)
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.