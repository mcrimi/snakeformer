import torch
import pickle
from huggingface_hub import hf_hub_download


print("Downloading from HuggingFace Hub...")
model_path = hf_hub_download(repo_id="mcrimi/snakeformer", filename="snake_model.pt")
meta_path = hf_hub_download(repo_id="mcrimi/snakeformer", filename="meta.pkl")

print(f"Loading metadata from: {meta_path}")
with open(meta_path, "rb") as f:
    meta = pickle.load(f)

# Extract tokenizer mappings
stoi = meta["stoi"]  # string to index
itos = meta["itos"]  # index to string

def encode(s):
    """Convert string to list of token IDs."""
    return [stoi[c] for c in s]

def decode(ids):
    """Convert list of token IDs back to string."""
    return "".join([itos[i] for i in ids])

from model.gpt import GPT, GPTConfig

config = GPTConfig(
    vocab_size=meta["vocab_size"],
    block_size=meta.get("block_size", 1024),
    n_embd=meta.get("n_embd", 128),
    n_head=meta.get("n_head", 8),
    n_layer=meta.get("n_layer", 4),
)

print(f"\nLoading model from: {model_path}")
model = GPT(config)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

num_params = sum(p.numel() for p in model.parameters())
print(f"Model loaded: {num_params / 1e6:.2f}M parameters")

# -----------------------------------------------------------------------------
# Example: Generate next board state
# -----------------------------------------------------------------------------

# Valid 16x16 board with snake (H=head, O=body, #=tail) and food (F)
board = """\
................
................
................
................
................
................
................
........#.......
........O.......
........H......F
................
................
................
................
................
................"""

action = "R"  # Move right (towards the food)

# Build prompt in the expected format
prompt = f"B:\n{board}\nA:{action}\nT:\n"

print("\n=== Input Prompt ===")
print(prompt)

# Encode prompt to token IDs
input_ids = encode(prompt)
print(f"\n=== Encoded ({len(input_ids)} tokens) ===")
print(f"First 20 tokens: {input_ids[:20]}...")

# Convert to tensor
input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)  # (1, seq_len)
print(f"Input tensor shape: {input_tensor.shape}")

# Generate output
print("\n=== Generating... ===")
stop_token_id = stoi.get("$")
print(f"Stop token ID: {stop_token_id}")

with torch.no_grad():
    output_ids = model.generate(
        input_tensor,
        max_new_tokens=300,  # Board is ~16*17 = 272 chars + some overhead
        stop_token_id=stop_token_id,
    )

# Decode output
output_text = decode(output_ids[0].tolist())

print("\n=== Full Output ===")
print(output_text)

# Extract just the generated part (after "T:\n")
generated = output_text[len(prompt):].split("$")[0]
print("\n=== Generated Board State ===")
print(generated)