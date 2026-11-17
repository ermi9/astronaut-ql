# Astronaut: Q-learning (Seminar topic C)

## Setup

I used Python 3 with a virtual environment. From the folder with the code:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "famnit_gym@git+https://github.com/DomenSoberlFamnit/famnit-gym"
```

This also installs gymnasium and numpy, which is everything the training code needs.

## Files

- `explore.py` plays random games and prints the height and speed ranges. I used it to choose the state ranges and to get the random baseline.
- `train_no_wrapper.py` is part 1: the plain game, without asteroids or oxygen.
- `train_asteroids.py` is part 2: the game with the Asteroids wrapper.
- `train_asteroids_oxygen.py` is part 3: the game with both the Asteroids and Oxygen wrappers.
- `graph.py` draws the loss and survival graphs from the saved logs. It is optional and needs matplotlib (`pip install matplotlib`). The training code does not use it.
- `*_log.csv` are the logs from my own runs, the ones the poster graphs were made from. Running the scripts again overwrites them with identical files.

## Running

Each part is its own script, so run them one at a time:

```bash
python train_no_wrapper.py
python train_asteroids.py
python train_asteroids_oxygen.py
```

Each script trains the agent, then tests it on 50 games it never saw during training, and compares it with a random agent on the same 50 games. At the end it prints the average number of frames survived, how many test games lasted the full 1200 frames, the training time, and for part 3 also the average number of oxygen tanks collected.

Rough running times on my laptop: part 1 about 20 seconds, part 2 about 2 minutes, part 3 about 17 to 18 minutes (60,000 training games).

Each script also saves two files:
- `<name>_log.csv` with the loss and frames survived for every training game
- `<name>_qtable.npy` with the trained Q-table

To draw a graph, set `name`, `title` and `block_size` at the top of `graph.py` and run `python graph.py`. It saves `<name>_plot.png`. I used `block_size` 20 for part 1, 200 for part 2 and 1000 for part 3.

## Extra experiments for part 3

The poster says that with 20,000 training games part 3 survived only about 700 frames, and that a bigger crash penalty did not fix this, while more training did. Both runs use `train_asteroids_oxygen.py` with small changes.

**Fewer training games.** Change `episodes=60000` to `episodes=20000` and run the script.

**Crash penalty of -10.** This is the run saved in `asteroids_oxygen_crash10_log.csv`. Starting from `train_asteroids_oxygen.py`:
1. In the reward function, make a collision return `-10` instead of `-1`. The oxygen reward stays `+1`.
2. Change `episodes=60000` to `episodes=20000`.
3. Change the output file names to `asteroids_oxygen_crash10_log.csv` and `asteroids_oxygen_crash10_qtable.npy`, so the main part 3 results are not overwritten.

Both runs take about 6 minutes. Change `episodes` back to 60000 afterwards to get the main part 3 results again.

## Reproducibility

All randomness is seeded, so running a script again gives the same results as on the poster:
- `np.random.seed(0)` fixes the agent's random choices (when it explores and which action it picks)
- `env.action_space.seed(0)` fixes the random actions taken from the game's action space
- training game number `i` is started with `env.reset(seed=i)`, so every training game has the same asteroid and oxygen positions every run
- the 50 test games use seeds 100000 to 100049, which are never used in training

The only number that can change between runs is the training time, since it depends on the computer.
