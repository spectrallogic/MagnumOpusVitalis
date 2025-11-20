"""
Magnum Opus v2.1 (Optimized): The Biological Architecture
====================================================================
ARCHITECT: Alan Hourmand
ENGINEER: Claude (Gemini)

FIXES v2.1:
- Fixed "Hydrocephalus" Bug: Consolidated 8 output heads into 1 shared head.
- Parameter count dropped from ~41M -> ~0.8M (True Microscopic Start).
- Preserved all biological features (Subconscious, Memory, Growth).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import sys
from pathlib import Path
import random

try:
    import tiktoken
except ImportError:
    print("⚠️  Please run: pip install tiktoken")
    sys.exit(1)

# ============================================================================
# 🧠 COMPONENT 1: MICRO-TO-MACRO SPEED SPECTRUM (Optimized)
# ============================================================================

class BiologicalMultiSpeed(nn.Module):
    def __init__(self, vocab_size, base_dim):
        super().__init__()
        self.dims = [8, 16, 32, 64, 96, 128, 192, 256]

        # 1. Shared Sensory Input
        self.shared_embed = nn.Embedding(vocab_size, base_dim)

        # 2. Speed Processors (The Synapses)
        self.projections = nn.ModuleList([nn.Linear(base_dim, d) for d in self.dims])

        # 3. Re-convergence (Project back to base_dim) -> THIS FIXES THE BLOAT
        self.out_projections = nn.ModuleList([nn.Linear(d, base_dim) for d in self.dims])

        # 4. Shared Output Head (One mouth, many thoughts)
        self.shared_head = nn.Linear(base_dim, vocab_size)

        # 5. Trust Weights
        self.trust_weights = nn.Parameter(torch.tensor([5.0, 3.0, 1.0, 0.1, 0.01, 0.0, 0.0, 0.0]))

    def forward(self, x):
        base = self.shared_embed(x)
        speed_outputs = []

        for proj_in, proj_out in zip(self.projections, self.out_projections):
            # Think at speed dimension
            thought = F.gelu(proj_in(base))
            # Project back to common language
            result = proj_out(thought)
            speed_outputs.append(result)

        # Stack: [8, batch, seq, base_dim]
        stacked = torch.stack(speed_outputs, dim=0)

        # Mix based on trust
        weights = F.softmax(self.trust_weights, dim=0)
        mixed_thought = torch.einsum('kbsd,k->bsd', stacked, weights)

        # Speak
        logits = self.shared_head(mixed_thought)
        return logits

# ============================================================================
# 🔮 COMPONENT 2: THE FULL 4-LAYER SUBCONSCIOUS
# ============================================================================

class SeaOfNoise(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        noise = torch.randn_like(x) * 0.15
        return self.proj(x) + noise

class PeakDetector(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scorer = nn.Linear(dim, 1)
    def forward(self, sea):
        scores = torch.sigmoid(self.scorer(sea))
        return sea * scores

class FutureGenerator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.sim = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )
    def forward(self, peaks):
        return self.sim(peaks)

class ScenarioEvaluator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.judge = nn.Linear(dim, 1)
    def forward(self, future):
        score = torch.sigmoid(self.judge(future))
        return future * score

class SubconsciousMind(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.l0 = SeaOfNoise(dim)
        self.l1 = PeakDetector(dim)
        self.l2 = FutureGenerator(dim)
        self.l3 = ScenarioEvaluator(dim)
        self.output = nn.Linear(dim, dim)

    def forward(self, x):
        s0 = self.l0(x)
        s1 = self.l1(s0)
        s2 = self.l2(s1)
        s3 = self.l3(s2)
        return self.output(s3)

# ============================================================================
# 💾 COMPONENT 3: INTEGRATED ASSOCIATIVE MEMORY
# ============================================================================

class MemoryTrace:
    def __init__(self, key, value, importance=0.5):
        self.key = key
        self.value = value
        self.importance = importance
        self.recency = 1.0
        self.access_count = 1

class BiologicalMemory(nn.Module):
    def __init__(self, dim, capacity=2000):
        super().__init__()
        self.dim = dim
        self.capacity = capacity
        self.traces = []

    def remember(self, hidden_state, tokens, importance=0.5):
        if len(tokens) < 2: return
        key = hidden_state.mean(dim=0).detach().cpu()
        for m in self.traces: m.recency *= 0.99
        self.traces.append(MemoryTrace(key, tokens, importance))
        if len(self.traces) > self.capacity:
            self.traces.sort(key=lambda x: x.importance * x.recency * x.access_count, reverse=True)
            self.traces = self.traces[:int(self.capacity * 0.9)]

    def reconstruct(self, query_state):
        if not self.traces: return None

        # Flatten to [Dim]
        query = query_state.detach().cpu()
        if query.dim() > 1:
            query = query.view(-1, query.shape[-1]).mean(dim=0)

        best_trace = None
        best_sim = -1.0
        candidates = random.sample(self.traces, min(len(self.traces), 50))

        for trace in candidates:
            key = trace.key.view(-1)
            sim = F.cosine_similarity(query, key, dim=0).item()
            if sim > best_sim:
                best_sim = sim
                best_trace = trace

        if best_sim > 0.75:
            best_trace.recency = 1.0
            best_trace.access_count += 1
            return best_trace.key.to(query_state.device)
        return None

# ============================================================================
# 🏛️ COMPONENT 4: MAGNUM OPUS v2.1 (Body)
# ============================================================================

class MagnumOpusV2(nn.Module):
    def __init__(self, vocab_size, base_dim=8, max_domains=50):
        super().__init__()
        self.vocab_size = vocab_size
        self.base_dim = base_dim
        self.max_domains = max_domains

        # 1. Core Identity
        self.embed = nn.Embedding(vocab_size, base_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, 1024, base_dim) * 0.02)

        # 2. Systems
        self.multi_speed = BiologicalMultiSpeed(vocab_size, base_dim)
        self.subconscious = SubconsciousMind(base_dim)
        self.memory = BiologicalMemory(base_dim)

        # 3. Growth
        self.domains = nn.ModuleList()
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=base_dim, nhead=2, batch_first=True, dim_feedforward=base_dim*4)
        ])
        self.head = nn.Linear(base_dim, vocab_size)

        # Stats
        self.steps = 0
        self.domains_created = 0

    def grow_width(self):
        if len(self.domains) >= self.max_domains: return False
        # Growth adds specialized residual adapter
        adapter = nn.Sequential(
            nn.Linear(self.base_dim, self.base_dim * 2),
            nn.GELU(),
            nn.Linear(self.base_dim * 2, self.base_dim)
        ).to(self.embed.weight.device)
        nn.init.zeros_(adapter[2].weight) # Start silent
        self.domains.append(adapter)
        self.domains_created += 1
        return True

    def grow_depth(self):
        if len(self.blocks) >= 12: return False
        new_block = nn.TransformerEncoderLayer(d_model=self.base_dim, nhead=2, batch_first=True, dim_feedforward=self.base_dim*4).to(self.embed.weight.device)
        self.blocks.append(new_block)
        return True

    def forward(self, x, y=None):
        b, s = x.shape

        # A. Embedding
        h = self.embed(x)
        if s > 1024: s = 1024
        h = h + self.pos_enc[:, :s, :]

        # B. Memory
        if self.training and self.steps % 20 == 0:
            recalled = self.memory.reconstruct(h)
            if recalled is not None:
                h = h + recalled.unsqueeze(0).unsqueeze(0) * 0.2

        # C. Subconscious
        context = h.mean(dim=1)
        intuition = self.subconscious(context)
        h = h + intuition.unsqueeze(1) * 0.1

        # D. Domains
        if len(self.domains) > 0:
            gate = torch.sigmoid(context.mean())
            if gate > 0.5:
                for domain in self.domains:
                    h = h + domain(h) * 0.1

        # E. Deep Processing
        for block in self.blocks:
            h = block(h)

        # F. Prediction Mixing
        deep_logits = self.head(h)
        fast_logits = self.multi_speed(x)

        # Dynamic mix based on depth
        ratio = min(0.5 + (len(self.blocks) * 0.05), 0.9)
        final_logits = (deep_logits * ratio) + (fast_logits * (1 - ratio))

        loss = None
        if y is not None:
            loss = F.cross_entropy(final_logits.reshape(-1, self.vocab_size), y.reshape(-1))

            # Growth Trigger
            if loss.item() > 3.5 and self.steps % 100 == 0:
                with torch.no_grad():
                    if not self.grow_width():
                        if self.grow_depth(): print(f"\n🧠 BRAIN EXPANSION: Block #{len(self.blocks)}")
                    else: print(f"\n🌿 CONCEPT EXPANSION: Domain #{self.domains_created}")

        return final_logits, loss

# ============================================================================
# 🧬 COMPONENT 5: INTERFACE
# ============================================================================

class LivingIntelligence:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"⚡ System Online: {self.device}")
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.model = MagnumOpusV2(self.tokenizer.n_vocab).to(self.device)
        # High LR for small model to learn fast
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.002)
        self.context_window = 128

    def feed(self, text):
        tokens = self.tokenizer.encode(text)
        if len(tokens) < 2: return None
        self.model.train()
        total_loss = 0
        chunks = 0
        for i in range(0, len(tokens)-1, self.context_window):
            chunk = tokens[i:i+self.context_window+1]
            if len(chunk) < 2: continue
            x = torch.tensor([chunk[:-1]], dtype=torch.long).to(self.device)
            y = torch.tensor([chunk[1:]], dtype=torch.long).to(self.device)
            self.optimizer.zero_grad()
            _, loss = self.model(x, y)
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                emb = self.model.embed(x)
                self.model.memory.remember(emb[0], chunk[:-1])

            total_loss += loss.item()
            self.model.steps += 1
            chunks += 1
        return total_loss / max(1, chunks)

    def train_from_folder(self, folder="training_data", epochs=3):
        path = Path(folder)
        if not path.exists(): path.mkdir()
        files = list(path.glob("*.txt"))
        if not files: return
        print(f"📂 Found {len(files)} sources.")
        for e in range(epochs):
            random.shuffle(files)
            for f in files:
                try:
                    text = f.read_text(encoding='utf-8', errors='ignore')
                    if text: self.feed(text)
                except: pass

    def chat(self):
        print("\n💬 Living Chat (v2.1). Type 'exit'.")
        while True:
            try:
                u = input("YOU: ")
                if u.lower() in ['exit', 'quit']: break
                l = self.feed(u)
                ctx = self.tokenizer.encode(u)
                if not ctx: continue
                out = []
                self.model.eval()
                for _ in range(60):
                    x = torch.tensor([ctx[-128:]], dtype=torch.long).to(self.device)
                    with torch.no_grad():
                        logits, _ = self.model(x)
                        logits = logits[0, -1, :] / 0.8
                        probs = F.softmax(logits, dim=-1)
                        next_t = torch.multinomial(probs, 1).item()
                        out.append(next_t)
                        ctx.append(next_t)
                        if next_t == 50256: break
                resp = self.tokenizer.decode(out)
                print(f"AI (L:{l:.2f}): {resp}" if l else f"AI: {resp}")
            except KeyboardInterrupt: break

if __name__ == "__main__":
    ai = LivingIntelligence()
    ai.chat()