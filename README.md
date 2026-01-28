

![[image.png]]

> The overengineered snake game that nobody asked for

  
Welcome to **Snakeformer**, an experiment in replacing hard-coded game logic with a neural network. This isn't an AI *playing* Snake. This is an AI *simulating* the universe of Snake.


[Recording of the game]

When you press "Up", the code doesn't check `y -= 1`. It asks a GPT-style language model: *"Given this board state and an 'Up' input, what should the next frame of reality look like?"*

It's slow, it's an overkill. It's fun to build.

# Quickstart

1. Download the model weights from Hugging Face

2. **Clone repo**
```bash
git clone https://github.com/mcrimi/snakeformer
cd snakeformer
```


2. **Run the Game:**
```bash
python play.py
```

![[CleanShot 2026-01-27 at 18.56.46.png]]

4. **Select Mode:**

- **SnakeFormer:** The neural engine (Recommended) - load the model that you downloaded from Hugging Face

# The idea

So I've trained this transformer based model. The main idea was to ASCII-fy a snake game. What if the board would look like this:

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

So the board `16x16` is a grid of characters:

- `.`: **Empty Space**
- `F`: **Food**
- `H`: **Head**
- `O`: **Body**
- `#`: **Tail**


Ok, we could already play this deterministically using the rules of the snake game. That's no fun. What if we trained a language model to learn this rules so that it could generate the ascii representation of the board based on a prompt? That way the language mode would BE the game and we can play it just like the deterministic version.

Here's the idea:

We want to be able to input this an ASCII Board to the model together with the action from the user.


```
B:
................
................
................
................
................
................
................
................
................
................
........#.......
........O.......
........H.......
...............F
................
................
```

Plus the action that the user took:

```
Action: Go Right
```

And we want the model respond to use with the next board configuration:

```
................
................
................
................
................
................
................
................
................
................
................
........#.......
........0H......
...............F
................
................
```


So we basically train the model using a lot of this sequence pairs, in the hopes that it learns the deterministic dynamics.


```
B:
................
................
................
................
................
................
................
................
................
................
........#.......
........O.......
........H.......
...............F
................
................
A:R
T:
................
................
................
................
................
................
................
................
................
................
................
........#.......
........0H......
...............F
................
................
$
```


- `B:` **Board State**
- `A:` **Action** (The player's input)
- `U`, `D`, `L`, `R`: The desired direction
- `T:` **Target State** (What the model *predicts* happens next)
- `X`: **Death** (Game Over condition)
- `$`: Stop token


# The Modules

## Data Generation

If you want to to train your own model model, you'll first need a gameplay dataset. The `dataset/data_gen.py` script gives you a couple of options on how to do this:

Run:
```bash

![Data Generation Menu](<media/data_gen.png>)

You have 3 options:

1. **Autoplay (Curriculum)**: A hard-coded heuristic bot plays thousands of games using varied strategies (Standard, Glutton, Fat Snake, etc.).
    - **Output**: `dataset/snake_data_curriculum.txt`
    - **Best for**: Generating the initial bulk pre-training dataset (~500k samples).

2. **Manual Play**: You play the game manually to demonstrate specific behaviors.
    - **Output**: `dataset/snake_data_curriculum.txt` (Appended to the same file)
    - **Best for**: Adding specific human-like moves or edge cases the bot misses.

3. **DAgger (Fine-tuning)**: **Requires a pre-trained model.** The heuristic bot plays while the Neural Model "shadows" it in the background. When the Neural Model makes a prediction error (hallucination) compared to the bot's ground truth, we record that specific "hard example".
    - **Output**: `dataset/snake_curriculum_dagger_fixes.txt`
    - **Best for**: Creating a high-quality fine-tuning dataset to fix specific model weaknesses.

This produces `dataset/snake_data_curriculum.txt`, which serves as the training corpus.

## Pre-Training
Once the data is generated, you can train the Transformer model. The `training/train.py` script handles this. It's a standard PyTorch training loop that minimizes the cross-entropy loss between the predicted next character and the actual next character.

To train a fresh model:
```bash
python training/train.py pretrain --max_iters 20000 --batch_size 64
# Or use the interactive menu:
python training/train.py
```

The script supports **Weights & Biases** logging if you want to track loss curves:
```bash
python training/train.py pretrain --wandb
```

## Online Training
Online training (implemented in `games/shadow_neural_snake.py`) is the unique feature of Snakeformer. It allows you to fine-tune the model *during gameplay*.

When playing in **Shadow Mode**, if the Neural Engine diverges from the deterministic Shadow Engine:
1. The game pauses.
2. A divergence menu appears.
3. Press **'T'** to trigger an immediate training optimization step.

The game constructs a mini-batch consisting of the *exact* context that caused the error, combined with the *correct* next token from the Shadow Engine. It runs a backward pass to update the model weights, heavily penalizing the mistake. You can then save the "enlightened" model back to disk.




# Model Architecture

At the heart of this project is a small, bespoke Transformer model for reptiles).
  

- **Architecture**: Decoder-only Transformer (GPT)
- **Parameters**: ~0.8 Million 
- **Layers**: 4
- **Attention Heads**: 8
- **Embedding Dimension**: 128
- **Context Window**: 1024 tokens
- **Vocabulary Size**: ~20 characters (ASCII board elements + control tokens)

![[Snakeformer Readme 2026-01-27 18.31.16.excalidraw]]

```mermaid_text

graph TD
subgraph Inputs
I[Input Sequence] -->|Indices| TE[Token Embedding]
P[Positions] -->|0..T| PE[Position Embedding]
end
TE & PE --> Add((+))
Add --> B1[Transformer Block 1]
B1 --> B2[Transformer Block 2]
B2 --> B3[Transformer Block 3]
B3 --> B4[Transformer Block 4]
subgraph Block Structure [Internal Block Details]
direction TB
BZ[Input] --> LN1[LayerNorm]
LN1 --> MHA[Multi-Head Attention]
MHA --> ADD1((+))
BZ --> ADD1
ADD1 --> LN2[LayerNorm]
LN2 --> FFN[Feed Forward]
FFN --> ADD2((+))
ADD1 --> ADD2
ADD2 --> BO[Output]
end
B4 --> LN_F[LayerNorm Final]
LN_F --> LH[Linear Head]
LH --> SM[Softmax]
SM --> O[Next Token Probabilities]
style Inputs fill:#f9f,stroke:#333
style Block Structure fill:#eee,stroke:#333

```




## 🔦 Shadow Mode (The "Trust But Verify" System)

  

Neural networks are creative. Physics engines are not supposed to be.

  

To make this a playable demo rather than a chaotic hallucination generator, I built **Shadow Mode**.

  

1. **The Neural Engine** generates the next frame in real-time.
2. **The Shadow Engine** (a boring, deterministic Python script) calculates what *should* have happened.
3. They run in parallel.

4. If they disagree? **🚨 DIVERGENCE DETECTED 🚨**

  

The game pauses. You get to see exactly what the model *thought* should happen vs. reality. Maybe it tried to teleport the apple. Maybe it decided the snake should have two heads.


  



## 🛠️ Tech Stack

- **PyTorch**: For the brain.

- **Curses**: For the retro UI.

- **Terry Pratchett**: For the spiritual guidance on making computers think.

  


**Give it a spin!** If you manage to teach it to play a perfect game just by scolding it every time it cheats, let me know. 🚀

