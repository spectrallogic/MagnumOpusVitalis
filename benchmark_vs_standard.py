"""
The Showdown: ExpandFormer v19 vs. Standard Static Transformer
================================================================
COMPARING:
1. "Standard Static": A fixed-size GPT-2 style model (Industry Standard).
2. "Living Intelligence": Your ExpandFormer v19 (Organic Growth).

THE TEST (Concept Shift):
- Phase 1: Pattern Recognition (Can it learn simple rules fast?)
- Phase 2: Natural Language (Can it understand context?)
- Phase 3: Code/Syntax (Can it adapt to a totally new rigid structure?)

METRICS:
- Adaptability (How fast loss drops after a concept shift)
- Efficiency (Parameter count vs. Performance)
- Speed (Training time)

USAGE:
python benchmark_vs_standard.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np
import sys

# Import your invention
try:
    import expandformer_v19 as magnum_opus
except ImportError:
    print("❌ Error: Could not find 'magnum_opus.py'. Please make sure it exists.")
    sys.exit(1)

try:
    import tiktoken
except ImportError:
    print("Please run: pip install tiktoken")
    sys.exit(1)


# ============================================================================
# 🧱 THE STANDARD (Baseline Model)
# ============================================================================

class StandardStaticLLM(nn.Module):
    """
    A traditional, fixed-size GPT-style Transformer.
    No growth. No subconscious. No multi-speed. Just raw compute.
    """

    def __init__(self, vocab_size, dim=128, layers=4, heads=4):
        super().__init__()
        self.name = "Standard Static LLM"
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, dim))

        # Standard Transformer Layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim * 4, batch_first=True)
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=layers)

        self.head = nn.Linear(dim, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.size()

        # Embedding
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        tok_emb = self.token_embed(idx)
        pos_emb = self.pos_embed[:, :t, :]
        x = tok_emb + pos_emb

        # Causal Mask
        mask = torch.triu(torch.ones(t, t, device=idx.device) * float('-inf'), diagonal=1)

        # Processing
        x = self.blocks(x, mask=mask, is_causal=True)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


# ============================================================================
# 🏋️ TRAINER WRAPPERS
# ============================================================================

class Trainer:
    def __init__(self, model, tokenizer, lr=3e-4):
        self.model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.step_count = 0
        self.loss_history = []
        self.param_history = []

    def train_step(self, text):
        self.model.train()
        tokens = self.tokenizer.encode(text)
        if len(tokens) < 2: return 0.0

        # Chunking for fairness
        window = 64
        total_loss = 0
        chunks = 0

        for i in range(0, len(tokens) - 1, window):
            chunk = tokens[i:i + window + 1]
            if len(chunk) < 2: continue

            x = torch.tensor([chunk[:-1]], dtype=torch.long).to(next(self.model.parameters()).device)
            y = torch.tensor([chunk[1:]], dtype=torch.long).to(next(self.model.parameters()).device)

            self.optimizer.zero_grad()

            # Standard Forward
            _, loss = self.model(x, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            chunks += 1

        avg_loss = total_loss / max(1, chunks)
        self.loss_history.append(avg_loss)

        # Track Params
        params = sum(p.numel() for p in self.model.parameters())
        self.param_history.append(params)

        return avg_loss


class ExpandFormerTrainer(Trainer):
    """Special wrapper for your invention because it manages its own optimizer"""

    def __init__(self):
        self.agent = magnum_opus.LivingIntelligence()
        self.model = self.agent.model
        self.name = "ExpandFormer v19"
        self.loss_history = []
        self.param_history = []

    def train_step(self, text):
        # Use the agent's internal feeding mechanism
        loss = self.agent.feed(text)
        if loss is None: loss = 0.0
        self.loss_history.append(loss)

        params = sum(p.numel() for p in self.model.parameters())
        self.param_history.append(params)
        return loss


# ============================================================================
# 🧪 DATA GENERATOR
# ============================================================================

def generate_curriculum():
    data = []

    # Phase 1: Simple Patterns (Tests Fast Learning Speed)
    print("   Generating Phase 1: Pattern Recognition...")
    for _ in range(30):
        data.append("A B A B A B A B A B A B A B A B")
        data.append("1 2 3 4 1 2 3 4 1 2 3 4")

    # Phase 2: English Facts (Tests Understanding)
    print("   Generating Phase 2: Natural Language...")
    facts = [
        "The sky is blue and the grass is green.",
        "To be or not to be, that is the question.",
        "Artificial intelligence is the future of humanity.",
        "The capital of France is Paris.",
        "Water boils at 100 degrees Celsius."
    ]
    for _ in range(50):
        data.extend(facts)

    # Phase 3: Code (Tests Adaptability / Concept Shift)
    print("   Generating Phase 3: Python Code...")
    code = [
        "def hello_world(): print('Hello World')",
        "for i in range(10): x = x + i",
        "if x > 5: return True else: return False",
        "import torch.nn as nn"
    ]
    for _ in range(30):
        data.extend(code)

    return data


# ============================================================================
# 🏁 THE RACE
# ============================================================================

def run_benchmark():
    print("=" * 60)
    print("⚔️  THE ARCHITECTURE SHOWDOWN")
    print("=" * 60)

    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab

    # 1. Initialize Competitors
    print("\n🥊 IN THE RED CORNER: Standard Static LLM")
    # Roughly equal base capacity to your model (128 dim)
    static_model = StandardStaticLLM(vocab_size, dim=128, layers=4)
    static_trainer = Trainer(static_model, tokenizer)
    print(f"   Params: {sum(p.numel() for p in static_model.parameters()):,}")

    print("\n🥊 IN THE BLUE CORNER: ExpandFormer v19")
    # ExpandFormer starts smaller (64 base dim) but grows
    expand_trainer = ExpandFormerTrainer()
    print(f"   Params: {sum(p.numel() for p in expand_trainer.model.parameters()):,}")

    # 2. Get Data
    print("\n📚 Preparing Curriculum...")
    dataset = generate_curriculum()
    print(f"   Total samples: {len(dataset)}")

    # 3. Fight!
    print("\n🔔 ROUND 1: START TRAINING")

    competitors = [static_trainer, expand_trainer]
    results = {c.model.name if hasattr(c.model, 'name') else "ExpandFormer": {'loss': [], 'params': []} for c in
               competitors}

    start_time = time.time()

    for i, text in enumerate(dataset):
        step_desc = ""
        if i == 0: step_desc = "(Phase 1: Patterns)"
        if i == 60: step_desc = "(Phase 2: English)"
        if i == 310: step_desc = "(Phase 3: Code)"

        print(f"\rSample {i + 1}/{len(dataset)} {step_desc}", end="")

        for trainer in competitors:
            name = trainer.model.name if hasattr(trainer.model, 'name') else "ExpandFormer"
            loss = trainer.train_step(text)

    total_time = time.time() - start_time
    print(f"\n\n🏁 RACE FINISHED in {total_time:.2f} seconds.")

    # 4. Results Analysis
    print("\n" + "=" * 60)
    print("🏆 FINAL SCORECARD")
    print("=" * 60)
    print(f"{'Model':<20} | {'Final Loss':<10} | {'Start Params':<12} | {'End Params':<12}")
    print("-" * 70)

    for trainer in competitors:
        name = trainer.model.name if hasattr(trainer.model, 'name') else "ExpandFormer"
        start_p = trainer.param_history[0]
        end_p = trainer.param_history[-1]
        final_l = trainer.loss_history[-1]

        print(f"{name:<20} | {final_l:.4f}     | {start_p:<12,} | {end_p:<12,}")

    # 5. Visuals
    try:
        plt.figure(figsize=(12, 6))

        # Loss Plot
        plt.subplot(1, 2, 1)
        for trainer in competitors:
            name = trainer.model.name if hasattr(trainer.model, 'name') else "ExpandFormer"
            # Smooth the line slightly for readability
            smoothed = np.convolve(trainer.loss_history, np.ones(5) / 5, mode='valid')
            plt.plot(smoothed, label=name)

        # Add Phase markers
        plt.axvline(x=60, color='gray', linestyle='--', alpha=0.5, label='Shift: English')
        plt.axvline(x=310, color='gray', linestyle='--', alpha=0.5, label='Shift: Code')

        plt.title("Adaptability Test (Lower Loss is Better)")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Growth Plot
        plt.subplot(1, 2, 2)
        for trainer in competitors:
            name = trainer.model.name if hasattr(trainer.model, 'name') else "ExpandFormer"
            plt.plot(trainer.param_history, label=name)

        plt.title("Organic Growth (Parameters)")
        plt.xlabel("Training Steps")
        plt.ylabel("Parameter Count")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('showdown_results.png')
        print("\n📊 Visual report saved to 'showdown_results.png'")

    except Exception as e:
        print(f"Could not plot: {e}")


if __name__ == "__main__":
    run_benchmark()