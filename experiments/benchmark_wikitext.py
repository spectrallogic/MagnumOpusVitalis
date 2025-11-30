"""
MagnumOpusVitalis Benchmark Suite
=================================
Reproducible benchmarks comparing growing architecture vs static transformers.

Uses WikiText-103 (standard NLP benchmark) for fair comparison.

Key metrics:
1. Final perplexity (lower = better)
2. Training FLOPs (compute efficiency)
3. Parameter count over time
4. Time to convergence

Author: Alan Hourmand
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import math

# Import our core
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))
from magnum_opus_core import MagnumOpusCore, GrowthConfig, train_on_text


# ============================================================================
# STANDARD STATIC TRANSFORMER (BASELINE)
# ============================================================================

class StaticTransformer(nn.Module):
    """
    Standard transformer for baseline comparison.

    This is a FIXED-SIZE model - same architecture as typical LLMs,
    just smaller for fair comparison.
    """

    def __init__(self, vocab_size: int, dim: int = 128, layers: int = 4, heads: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim

        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            batch_first=True,
            norm_first=True  # Pre-norm like modern transformers
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=layers)

        self.output_norm = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, vocab_size, bias=False)
        self.output_proj.weight = self.embed.weight  # Tie weights

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        b, t = idx.size()

        # Safety crop
        if t > 1024:
            idx = idx[:, -1024:]
            targets = targets[:, -1024:] if targets is not None else None
            t = 1024

        x = self.embed(idx) + self.pos_embed[:, :t, :]

        # Causal mask
        mask = torch.triu(
            torch.ones(t, t, device=idx.device) * float('-inf'),
            diagonal=1
        )

        x = self.blocks(x, mask=mask, is_causal=True)
        x = self.output_norm(x)
        logits = self.output_proj(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return {'logits': logits, 'loss': loss}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# DATA LOADING
# ============================================================================

class TextDataset(Dataset):
    """Simple text dataset for training"""

    def __init__(self, text: str, seq_len: int, vocab: Dict[str, int]):
        self.seq_len = seq_len
        self.vocab = vocab

        # Encode text
        self.data = torch.tensor(
            [vocab.get(c, 0) for c in text],
            dtype=torch.long
        )

    def __len__(self):
        return max(1, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


def load_wikitext103(path: str = "data/wikitext-103") -> Tuple[str, str]:
    """
    Load WikiText-103 dataset.

    If not present, downloads a sample or provides instructions.

    Returns (train_text, test_text)
    """
    train_path = Path(path) / "wiki.train.tokens"
    test_path = Path(path) / "wiki.test.tokens"

    if train_path.exists() and test_path.exists():
        with open(train_path, 'r', encoding='utf-8') as f:
            train_text = f.read()
        with open(test_path, 'r', encoding='utf-8') as f:
            test_text = f.read()
        return train_text, test_text

    # Provide sample data if WikiText not available
    print("WikiText-103 not found. Using built-in sample corpus.")
    print("For full benchmark, download from: https://huggingface.co/datasets/wikitext")
    print()

    # Generate a diverse sample corpus
    sample = generate_sample_corpus(size=100000)

    # Split 90/10
    split = int(len(sample) * 0.9)
    return sample[:split], sample[split:]


def generate_sample_corpus(size: int = 100000) -> str:
    """Generate diverse sample text for testing"""
    import random

    patterns = [
        # Language patterns
        "The {adj} {noun} {verb} the {adj} {noun}. ",
        "{name} said that {clause}. ",
        "In the {time}, {event} occurred. ",
        "According to {source}, {fact}. ",

        # Technical patterns
        "The function {func}() returns {type}. ",
        "Variable {var} is assigned to {value}. ",
        "Loop iteration {n} processes {item}. ",

        # Reasoning patterns
        "If {condition}, then {consequence}. ",
        "Because {cause}, we observe {effect}. ",
        "Given {premise}, we conclude {conclusion}. ",
    ]

    adjs = ["quick", "lazy", "bright", "dark", "small", "large", "old", "new", "hot", "cold"]
    nouns = ["fox", "dog", "cat", "bird", "tree", "house", "car", "book", "day", "night"]
    verbs = ["jumps over", "runs past", "watches", "follows", "ignores", "approaches"]
    names = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"]
    funcs = ["compute", "process", "analyze", "transform", "validate"]
    types = ["integer", "string", "boolean", "list", "dictionary"]

    text = []
    while len(''.join(text)) < size:
        pattern = random.choice(patterns)
        filled = pattern.format(
            adj=random.choice(adjs),
            noun=random.choice(nouns),
            verb=random.choice(verbs),
            name=random.choice(names),
            clause=f"the {random.choice(adjs)} {random.choice(nouns)} {random.choice(verbs)} something",
            time=f"year {random.randint(1900, 2024)}",
            event=f"a {random.choice(adjs)} discovery",
            source=f"{random.choice(names)}'s research",
            fact=f"{random.choice(nouns)}s exhibit {random.choice(adjs)} behavior",
            func=random.choice(funcs),
            type=random.choice(types),
            var=f"{random.choice('xyz')}_{random.randint(1, 99)}",
            value=str(random.randint(0, 1000)),
            n=str(random.randint(1, 100)),
            item=random.choice(nouns),
            condition=f"{random.choice(nouns)} is {random.choice(adjs)}",
            consequence=f"we expect {random.choice(adjs)} outcomes",
            cause=f"the {random.choice(noun)} changed",
            effect=f"{random.choice(adjs)} patterns",
            premise=f"{random.choice(adjs)} {random.choice(nouns)}",
            conclusion=f"the hypothesis holds"
        )
        text.append(filled)

    return ''.join(text)[:size]


# ============================================================================
# BENCHMARK METRICS
# ============================================================================

@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""
    model_name: str
    final_loss: float
    final_perplexity: float
    final_params: int
    initial_params: int
    training_steps: int
    training_time_seconds: float
    estimated_flops: int
    growth_events: int
    loss_history: List[float]
    param_history: List[int]

    def efficiency_score(self) -> float:
        """Lower is better: perplexity * params / 1M"""
        return self.final_perplexity * (self.final_params / 1_000_000)

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_training_flops(
        model: nn.Module,
        num_steps: int,
        batch_size: int,
        seq_len: int,
        dim: int,
        layers: int
) -> int:
    """Rough estimate of training FLOPs"""
    # Forward + backward ≈ 3x forward
    # Transformer forward: 2 * layers * seq * dim^2 * batch (attention + FFN)
    per_step = 3 * 2 * layers * seq_len * dim * dim * batch_size
    return per_step * num_steps


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def run_benchmark(
        train_text: str,
        test_text: str,
        max_steps: int = 2000,
        batch_size: int = 8,
        seq_len: int = 64,
        device: str = 'cpu',
        verbose: bool = True
) -> Tuple[BenchmarkResult, BenchmarkResult]:
    """
    Run full benchmark comparing MagnumOpus vs Static Transformer.

    Returns (magnum_result, static_result)
    """

    # Build vocabulary from training text
    chars = sorted(list(set(train_text)))
    vocab = {c: i for i, c in enumerate(chars)}
    vocab_size = len(vocab)

    if verbose:
        print(f"Vocabulary size: {vocab_size}")
        print(f"Training corpus: {len(train_text):,} characters")
        print(f"Test corpus: {len(test_text):,} characters")
        print(f"Device: {device}")
        print()

    # Create datasets
    train_dataset = TextDataset(train_text, seq_len, vocab)
    test_dataset = TextDataset(test_text, seq_len, vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    results = []

    # =========== MAGNUM OPUS ===========
    if verbose:
        print("=" * 60)
        print("Training MagnumOpusVitalis (growing architecture)")
        print("=" * 60)

    config = GrowthConfig(
        initial_dim=64,
        max_dim=256,
        max_layers=8,
        plateau_patience=150,
        lr_boost_attempts=2,
        min_age_for_growth=200
    )

    magnum = MagnumOpusCore(vocab_size=vocab_size, config=config).to(device)
    magnum_optimizer = torch.optim.AdamW(magnum.parameters(), lr=1e-3)

    magnum_history = {'loss': [], 'params': []}
    magnum_growths = 0
    initial_params = magnum.count_parameters()

    start_time = time.time()
    step = 0

    for epoch in range(100):  # Max epochs
        for inputs, targets in train_loader:
            if step >= max_steps:
                break

            inputs, targets = inputs.to(device), targets.to(device)

            metrics = magnum.training_step(inputs, targets, magnum_optimizer, base_lr=1e-3)

            magnum_history['loss'].append(metrics['loss'])
            magnum_history['params'].append(metrics['params'])

            if metrics['grew']:
                magnum_growths += 1

            if verbose and step % 200 == 0:
                print(f"Step {step}: loss={metrics['loss']:.4f}, params={metrics['params']:,}, "
                      f"layers={metrics['layers']}, reason={metrics['growth_reason']}")

            step += 1

        if step >= max_steps:
            break

    magnum_time = time.time() - start_time

    # Evaluate on test set
    magnum.eval()
    test_losses = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = magnum(inputs, targets)
            test_losses.append(outputs['loss'].item())

    magnum_test_loss = sum(test_losses) / len(test_losses)
    magnum_perplexity = math.exp(magnum_test_loss)

    magnum_result = BenchmarkResult(
        model_name="MagnumOpusVitalis",
        final_loss=magnum_test_loss,
        final_perplexity=magnum_perplexity,
        final_params=magnum.count_parameters(),
        initial_params=initial_params,
        training_steps=step,
        training_time_seconds=magnum_time,
        estimated_flops=estimate_training_flops(
            magnum, step, batch_size, seq_len,
            magnum.current_dim, len(magnum.cortex)
        ),
        growth_events=magnum_growths,
        loss_history=magnum_history['loss'],
        param_history=magnum_history['params']
    )

    if verbose:
        print(f"\nMagnumOpus Results:")
        print(f"  Test Perplexity: {magnum_perplexity:.2f}")
        print(f"  Final Params: {magnum.count_parameters():,} (grew from {initial_params:,})")
        print(f"  Growth Events: {magnum_growths}")
        print(f"  Training Time: {magnum_time:.1f}s")

    # =========== STATIC TRANSFORMER ===========
    if verbose:
        print()
        print("=" * 60)
        print("Training Static Transformer (fixed architecture)")
        print("=" * 60)

    # Use same FINAL size as grown MagnumOpus for fair comparison
    final_dim = magnum.current_dim
    final_layers = len(magnum.cortex)

    static = StaticTransformer(
        vocab_size=vocab_size,
        dim=final_dim,
        layers=final_layers,
        heads=max(1, final_dim // 32)
    ).to(device)

    static_optimizer = torch.optim.AdamW(static.parameters(), lr=1e-3)
    static_initial_params = static.count_parameters()

    static_history = {'loss': [], 'params': []}

    start_time = time.time()
    step = 0

    for epoch in range(100):
        for inputs, targets in train_loader:
            if step >= max_steps:
                break

            inputs, targets = inputs.to(device), targets.to(device)

            static_optimizer.zero_grad()
            outputs = static(inputs, targets)
            loss = outputs['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(static.parameters(), 1.0)
            static_optimizer.step()

            static_history['loss'].append(loss.item())
            static_history['params'].append(static.count_parameters())

            if verbose and step % 200 == 0:
                print(f"Step {step}: loss={loss.item():.4f}, params={static.count_parameters():,}")

            step += 1

        if step >= max_steps:
            break

    static_time = time.time() - start_time

    # Evaluate
    static.eval()
    test_losses = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = static(inputs, targets)
            test_losses.append(outputs['loss'].item())

    static_test_loss = sum(test_losses) / len(test_losses)
    static_perplexity = math.exp(static_test_loss)

    static_result = BenchmarkResult(
        model_name="StaticTransformer",
        final_loss=static_test_loss,
        final_perplexity=static_perplexity,
        final_params=static.count_parameters(),
        initial_params=static_initial_params,
        training_steps=step,
        training_time_seconds=static_time,
        estimated_flops=estimate_training_flops(
            static, step, batch_size, seq_len,
            final_dim, final_layers
        ),
        growth_events=0,
        loss_history=static_history['loss'],
        param_history=static_history['params']
    )

    if verbose:
        print(f"\nStatic Transformer Results:")
        print(f"  Test Perplexity: {static_perplexity:.2f}")
        print(f"  Params: {static.count_parameters():,} (fixed)")
        print(f"  Training Time: {static_time:.1f}s")

    return magnum_result, static_result


def print_comparison(magnum: BenchmarkResult, static: BenchmarkResult):
    """Print formatted comparison table"""
    print()
    print("=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print()
    print(f"{'Metric':<25} {'MagnumOpus':>20} {'Static':>20}")
    print("-" * 70)
    print(f"{'Test Perplexity':<25} {magnum.final_perplexity:>20.2f} {static.final_perplexity:>20.2f}")
    print(f"{'Final Params':<25} {magnum.final_params:>20,} {static.final_params:>20,}")
    print(f"{'Initial Params':<25} {magnum.initial_params:>20,} {static.initial_params:>20,}")
    print(f"{'Growth Events':<25} {magnum.growth_events:>20} {static.growth_events:>20}")
    print(f"{'Training Time (s)':<25} {magnum.training_time_seconds:>20.1f} {static.training_time_seconds:>20.1f}")
    print(f"{'Est. FLOPs':<25} {magnum.estimated_flops:>20,} {static.estimated_flops:>20,}")
    print(f"{'Efficiency Score':<25} {magnum.efficiency_score():>20.2f} {static.efficiency_score():>20.2f}")
    print("-" * 70)

    # Verdict
    print()
    if magnum.final_perplexity < static.final_perplexity:
        ppl_winner = "MagnumOpus"
        ppl_margin = (static.final_perplexity - magnum.final_perplexity) / static.final_perplexity * 100
    else:
        ppl_winner = "Static"
        ppl_margin = (magnum.final_perplexity - static.final_perplexity) / magnum.final_perplexity * 100

    if magnum.efficiency_score() < static.efficiency_score():
        eff_winner = "MagnumOpus"
        eff_ratio = static.efficiency_score() / magnum.efficiency_score()
    else:
        eff_winner = "Static"
        eff_ratio = magnum.efficiency_score() / static.efficiency_score()

    print(f"Perplexity Winner: {ppl_winner} ({ppl_margin:.1f}% better)")
    print(f"Efficiency Winner: {eff_winner} ({eff_ratio:.1f}x more efficient)")


# ============================================================================
# ABLATION STUDIES
# ============================================================================

def run_ablation_growth_patience(
        train_text: str,
        test_text: str,
        patience_values: List[int] = [50, 100, 200, 500],
        device: str = 'cpu'
) -> Dict[int, BenchmarkResult]:
    """
    Ablation study: How does growth patience affect results?

    Tests different patience values for the growth controller.
    """
    chars = sorted(list(set(train_text)))
    vocab = {c: i for i, c in enumerate(chars)}
    vocab_size = len(vocab)

    results = {}

    for patience in patience_values:
        print(f"\n--- Testing patience={patience} ---")

        config = GrowthConfig(
            initial_dim=64,
            max_dim=256,
            max_layers=8,
            plateau_patience=patience,
            lr_boost_attempts=2
        )

        model = MagnumOpusCore(vocab_size=vocab_size, config=config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Quick training
        train_dataset = TextDataset(train_text, 64, vocab)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

        for epoch in range(3):
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                model.training_step(inputs, targets, optimizer)

        # Evaluate
        test_dataset = TextDataset(test_text, 64, vocab)
        test_loader = DataLoader(test_dataset, batch_size=8)

        model.eval()
        test_losses = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs, targets)
                test_losses.append(outputs['loss'].item())

        test_loss = sum(test_losses) / len(test_losses)

        result = BenchmarkResult(
            model_name=f"MagnumOpus_patience{patience}",
            final_loss=test_loss,
            final_perplexity=math.exp(test_loss),
            final_params=model.count_parameters(),
            initial_params=1000,  # Approximate
            training_steps=0,
            training_time_seconds=0,
            estimated_flops=0,
            growth_events=model.growth_ctrl.state.total_growths,
            loss_history=[],
            param_history=[]
        )

        results[patience] = result
        print(f"  Perplexity: {result.final_perplexity:.2f}, "
              f"Growths: {result.growth_events}, "
              f"Final Params: {result.final_params:,}")

    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MagnumOpusVitalis Benchmark")
    parser.add_argument("--steps", type=int, default=1000, help="Training steps")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--ablation", action="store_true", help="Run ablation studies")
    parser.add_argument("--save", type=str, default=None, help="Save results to JSON")
    args = parser.parse_args()

    # Load data
    train_text, test_text = load_wikitext103()

    if args.ablation:
        print("Running ablation studies...")
        ablation_results = run_ablation_growth_patience(
            train_text[:50000],  # Use subset for speed
            test_text[:10000],
            device=args.device
        )

        print("\n=== ABLATION RESULTS ===")
        for patience, result in sorted(ablation_results.items()):
            print(f"Patience {patience}: PPL={result.final_perplexity:.2f}, "
                  f"Growths={result.growth_events}")
    else:
        # Main benchmark
        magnum_result, static_result = run_benchmark(
            train_text[:100000],  # Use 100k chars for reasonable time
            test_text[:10000],
            max_steps=args.steps,
            device=args.device,
            verbose=True
        )

        print_comparison(magnum_result, static_result)

        if args.save:
            results = {
                'magnum': magnum_result.to_dict(),
                'static': static_result.to_dict()
            }
            with open(args.save, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.save}")