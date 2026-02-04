[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-orange?style=flat&logo=huggingface)](https://huggingface.co/mcrimi/snakeformer)
[![License: GPL3](https://img.shields.io/badge/License-GPL--3.0-blue?logo=gnu)](LICENSE)


<img src="media/logo.png" width="1200" height="400" />


> The overengineered snake game that nobody asked for.

  
So I trained a decoder-only transformer (à la GPT) from scratch to simulate the good old game of Snake 🐍. The model takes a text-based board state and a user input as a prompt, and responds with the next frame, token by token, effectively acting as a learned physics engine.

(The game logic below is generated in real-time by the neural network.)


<p align="center">
  <img src="media/demo.gif" width="600" />
</p>


In this project we treat the game state not as a grid of pixels, but as a sequence of tokens. By training on a lot of state transitions ($S_t, A_t \rightarrow S_{t+1}$), the model approximates the game's update function. To succeed in generating a snake world, it must internalize it's implicit rules and topology.


It's slow, it's an overkill. It was fun to build. 🤷🏻‍♂️


This repo contains the code for the game console, the training pipeline (see [Pre-Training](#pre-training)) and the data generation pipeline (see [Data Generation](#data-generation)). 

I'm also publishing the model weights in [Huggingface](https://huggingface.co/mcrimi/snakeformer) so you can play it right away.


# Quickstart

1. **Clone the repo**
```bash
git clone https://github.com/mcrimi/snakeformer
cd snakeformer
```

2. **Set up a virtual environment and install requirements**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Run the game**
```bash
python play.py
```

4. **Select a game mode** from the main menu:

<p align="center">
  <img src="media/main_menu.png"/>
</p>


5. **Download or select a model** — When you choose SnakeFormer or Shadow mode, you'll be prompted to select a model checkpoint. If you don't have one locally, select **"Download from HuggingFace"** to fetch the pre-trained model directly from the UI.

That's it! Use arrow keys to control the snake and enjoy the most overengineered Snake game ever built.


# How it works

Let the game board be just a string, which can represented as a 16x16 grid:


```
`................`
`................`
`................`
`................`
`................`
`................`
`................`
`................`
`................`
`................`
`................`
`........#.......`
`........O.......`
`........H......F`
`................`
`................`
```

The states and state transitions can be represented trough this vocabulary (16 tokens total):

**Structural Tokens:**
- `B:` **Board State** prefix
- `A:` **Action** prefix (The player's input)
- `T:` **Target State** prefix (What the model *predicts* happens next)
- `:` Separator (used after B, A, T)
- `$` Stop token (end of sequence)
- `\n` Newline

**Direction Tokens:**
- `U` Up
- `D` Down
- `L` Left
- `R` Right

**Game Element Tokens:**
- `H` Snake Head
- `O` Snake Body segment
- `#` Snake Tail
- `F` Food
- `.` Empty space


**Game Condition Tokens:**
- `X` Death (Game Over)

So then the model receives a prompt (board state `B:` plus action taken by the user `A:`) like:
```
B:
................
................
................
................
......#.........
......O.........
......O.........
......OH.....F..
................
................
................
................
................
................
................
................
A:U
```

And generates the next board state `T:` after executing action `U` (Up):

```
T:
................
................
................
................
................
......#.........
......OH........
......OO.....F..
................
................
................
................
................
................
................
................
$
```


# The Modules

## Data Generation

If you want to to train your own model model, you'll first need a gameplay dataset. The `dataset/data_gen.py` script gives you a couple of options on how to do this:

Run:
```bash
python3 dataset/data_gen.py
```

<p align="center">
  <img src="media/data_gen.png"/>
</p>

You have 3 options:

1. **Autoplay (Curriculum)**: A hard-coded heuristic bot plays thousands of games using varied heuristic strategies for representative sample coverage of the gamespace.
    - **Output**: `dataset/snake_data_curriculum.txt`
    - **Best for**: Generating the initial bulk pre-training dataset (~500k samples).

2. **Manual Play**: You play the game manually to demonstrate specific behaviors.
    - **Output**: `dataset/snake_data_curriculum.txt` (Appended to the same file)
    - **Best for**: Adding specific human-like moves or edge cases the autoplay bot misses.

This produces `dataset/snake_data_curriculum.txt`, which serves as the training corpus.

3. **DAgger (Fine-tuning)**: **Requires a pre-trained model.** The heuristic bot plays while the Neural Model "shadows" it in the background. When the Neural Model makes a prediction error (hallucination) compared to the bot's ground truth, we record that specific "hard example".
    - **Output**: `dataset/snake_curriculum_dagger_fixes.txt`
    - **Best for**: Creating a high-quality fine-tuning dataset to fix specific model weaknesses.


You also have a script to analyze the dataset for correctness and coverage of the gamespace. Run:
```bash
python dataset/analyze_dataset.py
```

## Pre-Training

Once the data is generated, you can train the Transformer model. The `training/train.py` script handles this process. It uses a standard PyTorch training loop that minimizes the cross-entropy loss between the predicted next character and the ground-truth character. For more technical details, see [Model Architecture](#model-architecture).

### Basic Usage

**Pre-train a new model from scratch:**
```bash
python training/train.py pretrain --data_file snake_data_curriculum.txt
```

**Fine-tune an existing model:**
```bash
python training/train.py finetune --base_model snake_model.pt --new_model_name snake_model_v2.pt
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data_file` | `dataset/snake_data_curriculum.txt` | Path to the training dataset |
| `--model_name` | `snake_model.pt` | Output model filename (pretrain) |
| `--max_iters` | `20000` | Number of training iterations |
| `--batch_size` | `64` | Batch size |
| `--lr` | `0.001` | Learning rate |
| `--eval_interval` | `1000` | How often to evaluate and print losses |
| `--force` | `False` | Overwrite existing model files |

### Examples

**Train with custom hyperparameters:**
```bash
python training/train.py pretrain \
  --max_iters 30000 \
  --batch_size 32 \
  --lr 0.0005 \
  --eval_interval 500
```

**Fine-tune on DAgger-generated data:**
```bash
python training/train.py finetune \
  --base_model snake_model.pt \
  --data_file dataset/snake_curriculum_dagger_fixes.txt \
  --new_model_name snake_model_finetuned.pt \
  --max_iters 2000 \
  --lr 0.0001
```

### Weights & Biases Integration

The script supports **Weights & Biases** for experiment tracking:
```bash
python training/train.py pretrain \
  --wandb \
  --wandb_project snakeformer \
  --run_name experiment_v1
```
Set your API key via environment variable or pass it directly:
```bash
export WANDB_API_KEY=your_key_here
# or
python training/train.py pretrain --wandb --wandb_key your_key_here
```

### Help

For a full list of options:
```bash
python training/train.py pretrain --help
python training/train.py finetune --help
```

### Hardware Reference

The model checkpoint available on Hugging Face has been pre-trained on Google Colab for 9 hours with the following hyperparameters:

```
Iterations: 30000
Batch size: 32
LR: 0.0005
```
And using the following hardware:

```
CPU count	4
Logical CPU count	8
GPU count	1
GPU type	Tesla T4
```

### Val Loss Plot (W&B)
<p align="center">
  <img src="media/val_loss.png" width="800"/>
</p>


## Online Training
Ok, this I think it's cool. I've implemented an online training mechanism (implemented in `games/shadow_neural_snake.py`) that allows you to fine-tune the model *during gameplay*.


When playing in **Shadow Mode**, if the Model Engine output diverges from the deterministic Shadow Engine:
1. The game pauses.
2. **🚨 DIVERGENCE DETECTED 🚨** A divergence menu appears.
3. Press **'T'** to trigger an immediate training optimization step.


<p align="center">
  <img src="media/online_train_demo.gif" width="700" />
</p>


The game constructs a mini-batch consisting of the *exact* context that caused the error, combined with the *correct* next token from the Shadow Engine. It runs a backward pass to update the model weights, heavily penalizing the mistake.

**The Penalization Logic:**
- **Standard Error**: 1x Weight
- **Critical Error**: **50x Weight** (if the error involves Head `H`, Food `F`, or Death `X`)

This 50x multiplier forces the model to prioritize "keeping the snake alive" and "eating food" over merely getting the background dots correct. This might be an overkill, but it worked in my experiments. Feel free to tinke around.

After a few iterations, the model learns to correct its behavior and the game resumes. You can then save the enlightened model back to disk.


# Model Architecture

Here's a bespoke transformer model for reptiles:
  
- **Architecture**: Decoder-only Transformer (GPT)
- **Parameters**: ~0.93 Million 
- **Layers**: 4
- **Attention Heads**: 8
- **Embedding Dimension**: 128
- **Context Window**: 1024 tokens
- **Vocabulary Size**: 16

![Model](media/model.png)

# Learnings

## The Tail Problem

I initially didn't recognize the importance of clearly defining the snake's tail (`#`) as part of the board state. Why was that important?Becaue I formulated and engineerd inference as if we would be under Markovian Process, meaning that the next state depends **only on the current state**. This is impossible without clearly determining where the tale is

Take the following snake without explict tail:

```text
O H
O O
```

It is impossible to tell if the tail of the snake is to the left or below the head snake — that depends on how it got to be in that position.

When the model is unsure which segment to delete (the tail) in the next board state, it often deletes nothing to minimize loss, causing the snake to grow indefinitely.

Changing the tail character to a unique token (`T`) solves this instantly. It makes the state fully observable from a single frame, allowing the model to learn the simple rule: *"Find `#T` and turn it into `.`"*.

I could also have changed training and gamplay+inference to cater for longer context wiindows, but this did the trick alrady.


## Can randomness be learned?

I had an interesting problem with food spawning. Originally the game dynamics (and therefore the training) considered a random spawning of initial and new food. But then during training, the model was having a really hard time capturing this and I had to switch to a deterministic approach for food spawning, something that the model can learn.

The current deterministic spawning logic computes food position from the snake's head:

```python
target_r = (head[0] + 5) % game_height
target_c = (head[1] + 7) % game_width
# Then scans for first empty cell if position is occupied
```

The fundamental issue is that supervised learning needs a deterministic mapping from input to output. With random food spawning:

1. The same board state + action could produce many different next states (food appearing at any empty cell)
2. The model is trained with cross-entropy loss to predict the *most likely* next token
3. When faced with many equally valid outputs, the model either:
   - Averages probabilities across all valid food positions (predicting nothing confidently)
   - Overfits to specific examples (memorizing rather than generalizing)
   - Hallucinates "compromise" positions that don't match any training example (no food in the board)

**Could have I solved for this differently?**

Theoretically yes, but not without adding quite some complexity:

- Pass a random seed as input so the model can learn (board, action, seed) → next_state
- Use an architecture designed for multimodal outputs (VAEs, diffusion). Or at least this is what Claude Opus 4.5 told me. I would like to try that.


# Tech Stack

- **PyTorch**: The core deep learning framework used to build and train the Transformer model.
- **Curses**: The standard Python library used to create the retro, terminal-based user interface (TUI).
- **Weights & Biases**: Used for experiment tracking, visualizing loss curves, and monitoring training progress.
- **Pickle**: Used for efficient serialization of dataset metadata and vocabulary.



**Give it a spin!** If you manage to teach it to play a perfect game just by scolding it every time it cheats, let me know. 🚀

