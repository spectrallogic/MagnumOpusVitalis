"""
Universal Test Suite (Grandmaster Edition)
================================================================
MODES:
1. Chat (Realtime Learning)
2. Pacman (Eat files from training_data)
3. Hybrid (Chat while it eats files in background)
4. Benchmark (Grandmaster IQ Test vs Standard LLM)

USAGE:
python universal_test_suite.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time
import threading
import importlib.util
from pathlib import Path
import glob
import random

# ============================================================================
# 🛠️ UTILITIES
# ============================================================================

def load_model_from_file(filepath):
    """Dynamic import: Loads the 'LivingIntelligence' class from any python file."""
    module_name = Path(filepath).stem
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if hasattr(module, 'LivingIntelligence'):
        return module.LivingIntelligence()
    else:
        raise ImportError(f"Class 'LivingIntelligence' not found in {filepath}")

def scan_for_models():
    """Finds all magnum_opus_v*.py files."""
    files = glob.glob("magnum_opus_v*.py")
    files.sort()
    return files

# ============================================================================
# ⚔️ BENCHMARK OPPONENT (Standard LLM)
# ============================================================================

class StandardStaticLLM(nn.Module):
    """The 'Control Group' for benchmarks."""
    def __init__(self, vocab_size, dim=128, layers=4, heads=4):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4, batch_first=True)
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Linear(dim, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        # Safety crop
        if t > 1024:
            idx = idx[:, -1024:]
            targets = targets[:, -1024:] if targets is not None else None
            t = 1024

        x = self.token_embed(idx) + self.pos_embed[:, :t, :]
        mask = torch.triu(torch.ones(t, t, device=idx.device) * float('-inf'), diagonal=1)
        x = self.blocks(x, mask=mask, is_causal=True)
        logits = self.head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

# ============================================================================
# 🧬 GRANDMASTER DATA GENERATOR
# ============================================================================

def generate_grandmaster_curriculum(size=2000):
    data = []
    print(f"   Generating {size} Grandmaster-level tasks...")

    vars = ['x', 'y', 'z', 'a', 'b', 'c', 'val', 'res', 'tmp']

    for _ in range(size):
        task_type = random.choice(['binding', 'nested', 'logic', 'retrieval', 'func'])

        # 1. Variable Binding (Memory)
        if task_type == 'binding':
            v1, v2 = random.sample(vars, 2)
            val = random.randint(0, 999)
            sample = f"{v1} = {val}; {v2} = 0; print({v1}) # Output: {val}"
            data.append(sample)

        # 2. Nested Structure (Depth)
        elif task_type == 'nested':
            depth = random.randint(3, 6)
            keys = random.sample(vars, depth)
            open_str = "".join([f'{{"{k}": ' for k in keys])
            close_str = "}" * depth
            sample = f"JSON: {open_str}\"target\"{close_str}"
            data.append(sample)

        # 3. Logic/Arithmetic (Reasoning)
        elif task_type == 'logic':
            a, b = random.randint(1, 20), random.randint(1, 20)
            op = random.choice(['>', '<', '=='])
            if op == '>': res = "True" if a > b else "False"
            elif op == '<': res = "True" if a < b else "False"
            else: res = "True" if a == b else "False"
            sample = f"if {a} {op} {b}: return True else: return False # {res}"
            data.append(sample)

        # 4. Retrieval (Long Context)
        elif task_type == 'retrieval':
            target = random.randint(100, 999)
            noise = " ".join([f"{v}={random.randint(0,99)}" for v in vars])
            sample = f"SECRET_KEY = {target}. Ignore this: {noise}. What is the SECRET_KEY? {target}"
            data.append(sample)

        # 5. Function Composition (Sequential Logic)
        elif task_type == 'func':
            val = random.randint(1, 5)
            res = (val + 2) * 3
            sample = f"def f(x): return x + 2; def g(x): return x * 3; print(g(f({val}))) # {res}"
            data.append(sample)

    return data

# ============================================================================
# 🕹️ MODES
# ============================================================================

def mode_benchmark(agent):
    print("\n⚔️  BENCHMARK MODE (Grandmaster Gauntlet)")

    # 1. Data
    data = generate_grandmaster_curriculum(size=2000)

    # 2. Competitor
    print("Initializing Standard Static LLM (Red Corner)...")
    std_model = StandardStaticLLM(agent.tokenizer.n_vocab).to(agent.device)
    std_opt = torch.optim.AdamW(std_model.parameters(), lr=3e-4)

    std_params = sum(p.numel() for p in std_model.parameters())
    print(f"   Standard Params: {std_params:,}")

    print(f"STARTING RACE (Data: {len(data)} samples, 10 Epochs)...")
    start = time.time()

    # Metrics
    history_std = []
    history_my = []

    # Epochs
    epochs = 10
    for e in range(epochs):
        print(f"\n--- Epoch {e + 1}/{epochs} ---")
        random.shuffle(data)

        loss_std_epoch = []
        loss_my_epoch = []

        for i, text in enumerate(data):
            # Train Standard
            toks = agent.tokenizer.encode(text)
            if len(toks) < 2: continue
            x = torch.tensor([toks[:-1]], device=agent.device)
            y = torch.tensor([toks[1:]], device=agent.device)

            std_opt.zero_grad()
            _, l = std_model(x, y)
            l.backward()
            std_opt.step()
            loss_std_epoch.append(l.item())

            # Train Yours
            l_my = agent.feed(text)
            if l_my: loss_my_epoch.append(l_my)

            if i % 500 == 0 and i > 0:
                print(f"   Progress: {i}/{len(data)}...")

        avg_std = sum(loss_std_epoch) / max(len(loss_std_epoch), 1)
        avg_my = sum(loss_my_epoch) / max(len(loss_my_epoch), 1)

        print(f"   [Standard: {avg_std:.4f}] vs [Yours: {avg_my:.4f}]")

        # Growth Report
        domains = len(agent.model.domains)
        blocks = len(agent.model.blocks)
        my_params = sum(p.numel() for p in agent.model.parameters())
        print(f"   Your Brain: {domains} Domains, {blocks} Blocks ({my_params:,} params)")

        history_std.append(avg_std)
        history_my.append(avg_my)

    duration = time.time() - start
    print(f"\n🏁 RACE COMPLETE ({duration:.2f}s)")
    print("-" * 40)

    final_std = history_std[-1]
    final_my = history_my[-1]

    # Calculate Efficiency Score (Lower is better)
    # Score = Loss * (Params / 1,000,000)
    eff_std = final_std * (std_params / 1_000_000)
    eff_my = final_my * (my_params / 1_000_000)

    print(f"{'Metric':<20} | {'Standard LLM':<15} | {'Your AI':<15}")
    print("-" * 60)
    print(f"{'Final Loss':<20} | {final_std:.4f}          | {final_my:.4f}")
    print(f"{'Parameters':<20} | {std_params:,}     | {my_params:,}")
    print(f"{'Efficiency Score':<20} | {eff_std:.2f}           | {eff_my:.2f}")
    print("-" * 60)

    if eff_my < eff_std:
        print(f"🏆 RESULT: VICTORY! Your AI is {eff_std / eff_my:.1f}x more efficient.")
    elif final_my < final_std:
        print("🏆 RESULT: VICTORY! Absolute intelligence win.")
    else:
        print("🥈 RESULT: Close match.")

def mode_hybrid(agent):
    print("\n🧠 HYBRID MODE (Chat + Background Learning)")
    print("Type messages to chat. Put files in 'training_data' to teach it simultaneously.")
    print("Type 'exit' to quit.")

    stop_event = threading.Event()

    def background_feeder():
        processed = set()
        while not stop_event.is_set():
            path = Path("training_data")
            if path.exists():
                for file in path.glob("*.txt"):
                    if file.name not in processed:
                        try:
                            text = file.read_text(encoding='utf-8', errors='ignore')
                            if text:
                                l = agent.feed(text)
                                print(f"\n[Background] Absorbed {file.name} (Loss: {l:.2f})")
                                processed.add(file.name)
                        except: pass
            time.sleep(5)

    t = threading.Thread(target=background_feeder, daemon=True)
    t.start()

    agent.chat()
    stop_event.set()

# ============================================================================
# 🚀 MAIN MENU
# ============================================================================

def main():
    print("="*60)
    print("🎛️  UNIVERSAL TEST SUITE (Grandmaster Edition)")
    print("="*60)

    models = scan_for_models()
    if not models:
        print("❌ No 'magnum_opus_v*.py' files found!")
        return

    print("\nAvailable Brains:")
    for i, f in enumerate(models):
        print(f"  {i+1}. {f}")

    try:
        choice = int(input("\nSelect Brain (Number): ")) - 1
        selected_file = models[choice]
        print(f"Loading {selected_file}...")
        agent = load_model_from_file(selected_file)
    except:
        print("Invalid selection.")
        return

    print("\nSelect Mode:")
    print("  1. 💬 Chat (Realtime)")
    print("  2. 📂 Pacman (Train from Folder)")
    print("  3. 🧠 Hybrid (Chat + Background Train)")
    print("  4. ⚔️  Benchmark (vs Standard LLM)")

    mode = input("\nChoice: ")

    if mode == '1': agent.chat()
    elif mode == '2': agent.train_from_folder()
    elif mode == '3': mode_hybrid(agent)
    elif mode == '4': mode_benchmark(agent)
    else: print("Unknown mode.")

if __name__ == "__main__":
    main()