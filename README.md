## Distributed IQN
- Python 3.7, PyTorch 1.0
- Buffer in GPU for fast sampling
- Multiple instances of the environment

## Results
Average returns after 10 hours of training on one T4 GPU
- `MsPacman`: 4454
- `SpaceInvaders`: 2413

## Start
```
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
tensorboard --logdir runs
python -m train cartpole
```

## Dependencies
```
git clone https://github.com/openai/baselines.git
pip install -e baselines
```
