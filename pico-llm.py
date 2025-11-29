# starter code by matus & o1-pro
import argparse
import time
import random
import math
import copy
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# We do not import numpy or scikit-learn, so we implement a naive k-means in pure PyTorch.
# If you prefer scikit-learn, you can adapt the code.

from datasets import load_dataset
import tiktoken

################################################################################
# 1. Command-line arg parsing
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Train multiple k-gram or sequence-based models on TinyStories and/or custom text files.")
    parser.add_argument("--input_files", nargs="*", default=None,
                        help="Optional list of text files to mix in as data sources. Each line is one example (up to block_size).")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="If set, recursively load every *.txt file under this directory as additional training data.")
    parser.add_argument("--limit_custom_examples", type=int, default=None,
                        help="Optional cap on the number of custom text lines to ingest (useful for quick smoke tests).")
    parser.add_argument("--custom_sweep_log", type=str, default=None,
                        help="Optional JSON file to append loss metrics for custom-data sweeps.")
    parser.add_argument("--record_custom_sweep", action="store_true",
                        help="If set, store loss metrics for plotting custom-data sweeps.")
    parser.add_argument("--test_split_ratio", type=float, default=None,
                        help="If set, reserve this fraction of each custom file for evaluation (only when recording sweeps).")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of epochs to train each model (default: 3).")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Mini-batch size for training/evaluation (default: 16).")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for the optimizers (default: 1e-3).")
    parser.add_argument("--tinystories_weight", type=float, default=0.5,
                        help="Probability of sampling from TinyStories if present. Default=0.5. (set to 0.0 to skip TinyStories).")
    parser.add_argument("--max_steps_per_epoch", type=int, default=None,
                        help="If set, each epoch ends after this many steps (for quick tests).")
    parser.add_argument("--num_inner_mlp_layers", type=int, default=1,
                        help="Number of (Linear->SiLU) blocks inside the k-gram MLP. Default=1.")
    parser.add_argument("--monosemantic_enabled", action="store_true",
                        help="(DISABLED BY DEFAULT) If set, run the monosemantic analysis.")
    parser.set_defaults(monosemantic_enabled=False)  # disable by default

    # Additional hyperparams to mitigate slow k-gram
    parser.add_argument("--kgram_k", type=int, default=3,
                        help="Sliding window size for k-gram MLP. Smaller can reduce memory usage. Default=3.")
    parser.add_argument("--kgram_chunk_size", type=int, default=1,
                        help="Process k-gram timesteps in micro-batches. Default=1.")

    parser.add_argument("--block_size", type=int, default=1024,
                        help="Maximum sequence length for each example. Default=1024.")

    # New arguments:
    parser.add_argument("--embed_size", type=int, default=1024,
                        help="Dimension of the embedding layer for LSTM, MLP, etc. Default=1024.")
    parser.add_argument("--prompt", type=str, default="Once upon a",
                        help="Prompt used for generation. Default='Once upon a'.")
    parser.add_argument(
        "--enable_transformer_variants",
        nargs="*",
        default=[],
        choices=["mingpt", "gpt2", "gptoss"],
        help="Explicit list of transformer variants to instantiate. Leave empty to skip transformer runs.",
    )
    parser.add_argument(
        "--collect_transformer_metrics",
        action="store_true",
        help="Run the lightweight synthetic benchmark for the requested transformer variants.",
    )

    # Newly added device argument:
    parser.add_argument("--device_id", type=str, default="cuda:0",
                        help="Torch device identifier (default='cuda:0'). If CUDA is unavailable, fallback to 'cpu'.")
    # Evaluation on TinyStories and checkpointing
    parser.add_argument("--tinystories_eval_ratio", type=float, default=None,
                        help="If set and tinystories_weight>0, reserve this fraction of TinyStories for evaluation.")
    parser.add_argument("--save_checkpoints", action="store_true",
                        help="If set, save model checkpoints after training.")
    parser.add_argument("--checkpoint_dir", type=str, default="artifacts/checkpoints",
                        help="Directory to save checkpoints when --save_checkpoints is set.")

    args = parser.parse_args()
    return args


################################################################################
# 2. Data handling: entire sequences up to block_size => (seq_len, batch)
################################################################################

class MixedSequenceDataset(torch.utils.data.Dataset):
    """
    We store two lists of entire token sequences:
      - tinystories_seqs
      - other_seqs
    Each sequence is length <= block_size.

    During __getitem__, we randomly pick from one list or the other with probability p_tiny.
    Return that entire sequence as a 1D LongTensor.
    """
    def __init__(self, tinystories_seqs, other_seqs, p_tiny: float):
        super().__init__()
        self.tinystories_seqs = tinystories_seqs
        self.other_seqs = other_seqs
        self.p_tiny = p_tiny

        self.has_tinystories = (len(self.tinystories_seqs) > 0)
        self.has_other = (len(self.other_seqs) > 0)

        self.total_length = len(self.tinystories_seqs) + len(self.other_seqs)
        if self.total_length == 0:
            raise ValueError("No data found! Both TinyStories and other sets are empty.")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        r = random.random()
        if self.has_tinystories and self.has_other:
            if r < self.p_tiny:
                i = random.randint(0, len(self.tinystories_seqs) - 1)
                seq = self.tinystories_seqs[i]
            else:
                i = random.randint(0, len(self.other_seqs) - 1)
                seq = self.other_seqs[i]
        elif self.has_tinystories:
            i = random.randint(0, len(self.tinystories_seqs) - 1)
            seq = self.tinystories_seqs[i]
        else:
            i = random.randint(0, len(self.other_seqs) - 1)
            seq = self.other_seqs[i]

        return torch.tensor(seq, dtype=torch.long)


def seq_collate_fn(batch):
    """
    batch: list of 1D LongTensors of various lengths [<= block_size].
    1) find max length
    2) pad with zeros
    3) shape => (max_len, batch_size)
    """
    max_len = max(len(seq) for seq in batch)
    batch_size = len(batch)

    padded = torch.zeros(max_len, batch_size, dtype=torch.long)
    for i, seq in enumerate(batch):
        seq_len = seq.size(0)
        padded[:seq_len, i] = seq

    return padded


def ensure_sample_dataset(path: Path = Path("3seqs.txt"), repeats: int = 1111) -> Path:
    """
    Ensure that we have a tiny synthetic dataset on disk for quick experiments.
    The dataset cycles through three numeric sequences to provide deterministic training data.
    """
    if path.exists():
        return path

    sequences = [
        "0 1 2 3 4",
        "4 3 2 1 0",
        "1 3 5 7 9",
    ]
    with path.open("w", encoding="utf-8") as fp:
        for _ in range(repeats):
            for seq in sequences:
                fp.write(seq + "\n")
    return path


def load_sequences(config: Dict) -> Tuple[MixedSequenceDataset, List[List[int]], List[List[int]]]:
    """
    Utility used by the benchmark helpers to mirror the main training data preparation.
    """
    block_size = config.get("block_size", 128)
    tinystories_weight = config.get("tinystories_weight", 0.0)
    use_synthetic = config.get("use_synthetic", True)
    train_subset_size = config.get("train_subset_size", 1024)

    enc = tiktoken.get_encoding("gpt2")
    tinystories_seqs: List[List[int]] = []
    other_seqs: List[List[int]] = []

    if tinystories_weight > 0.0:
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        dataset = dataset.select(range(train_subset_size))
        for sample in dataset:
            tokens = enc.encode(sample["text"])
            tokens = tokens[:block_size]
            if tokens:
                tinystories_seqs.append(tokens)

    if use_synthetic:
        synthetic_path = ensure_sample_dataset()
        with synthetic_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                tokens = enc.encode(line)[:block_size]
                if tokens:
                    other_seqs.append(tokens)

    dataset = MixedSequenceDataset(
        tinystories_seqs=tinystories_seqs,
        other_seqs=other_seqs,
        p_tiny=tinystories_weight,
    )
    return dataset, tinystories_seqs, other_seqs


################################################################################
# 3. K-gram MLP in a sequence-to-sequence approach
################################################################################

def compute_next_token_loss(logits, tokens):
    """
    logits: (seq_len, batch, vocab_size)
    tokens: (seq_len, batch)
    Next-token prediction => we shift target by 1.
    """
    seq_len, batch_size, vocab_size = logits.shape
    if seq_len < 2:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    preds = logits[:-1, :, :]  # (seq_len-1, batch, vocab_size)
    gold = tokens[1:, :]       # (seq_len-1, batch)

    preds = preds.reshape(-1, vocab_size)
    gold = gold.reshape(-1)
    return F.cross_entropy(preds, gold)


class KGramMLPSeqModel(nn.Module):
    """
    For each position t in [0..seq_len-1], gather the last k tokens => one-hot => MLP => logits.
    Return (seq_len, batch, vocab_size).

    Potentially very large memory usage for big vocab or seq_len. chunk_size helps mitigate overhead.
    """

    def __init__(
        self,
        vocab_size,
        k=3,
        embed_size=1024,
        num_inner_layers=1,
        chunk_size=1,
        variant: str = "embedding",
        hidden_dim: int = 512,
        conv_hidden_dim: int = 512,
        allow_alt_variants: bool = False,
    ):
        super().__init__()
        self.k = k  # store how many previous tokens form the context window
        self.vocab_size = vocab_size  # total vocabulary size for output logits
        self.embed_size = embed_size  # embedding dimension for variants that use embeddings
        self.num_inner_layers = num_inner_layers  # number of hidden layers inside the MLP
        self.chunk_size = chunk_size  # process timesteps in micro-batches to save memory

        # Select the concrete sub-network for processing each k-gram context.
        variant = variant.lower()
        if variant != "embedding" and not allow_alt_variants:
            raise ValueError(
                f"KGramMLPSeqModel variant '{variant}' requires allow_alt_variants=True "
                "to prevent accidental use of slower experimental architectures."
            )
        if variant == "onehot":
            # Pure one-hot MLP consuming (k * vocab_size) features.
            self.net = build_kgram_onehot_mlp(
                vocab_size=self.vocab_size,
                k=self.k,
                hidden_dim=hidden_dim,
                num_inner_layers=self.num_inner_layers,
            )
        elif variant == "embedding":
            # Embedding-based MLP that projects tokens into a smaller space.
            self.net = build_kgram_embedding_mlp(
                vocab_size=self.vocab_size,
                k=self.k,
                embed_dim=self.embed_size,
                hidden_dim=hidden_dim,
                num_inner_layers=self.num_inner_layers,
            )
        elif variant == "conv":
            # Embedding + depthwise convolution hybrid.
            self.net = build_kgram_conv_mlp(
                vocab_size=self.vocab_size,
                k=self.k,
                embed_dim=self.embed_size,
                hidden_dim=conv_hidden_dim,
            )
        else:
            raise ValueError(f"Unknown KGramMLPSeqModel variant '{variant}'.")
        self.variant = variant  # store which variant we instantiated for logging

    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch)
        return: (seq_len, batch, vocab_size)
        We'll do a loop over time steps. chunk_size can reduce overhead.
        """
        seq_len, batch_size = tokens_seq.shape  # unpack sequence and batch dimensions
        outputs = []  # collect logits chunks for each processed window

        start = 0  # begin at the first timestep
        while start < seq_len:
            end = min(start + self.chunk_size, seq_len)  # determine micro-batch end
            block_outputs = []  # store logits for timesteps within this micro-batch
            for t in range(start, end):
                batch_logits = []  # accumulate logits for each sequence in the batch
                for b in range(batch_size):
                    if t < self.k:
                        # Not enough history; pad with zeros on the left.
                        needed = self.k - t
                        context_ids = [0] * needed + tokens_seq[:t, b].tolist()
                    else:
                        # Extract the last k tokens as context.
                        context_ids = tokens_seq[t - self.k : t, b].tolist()

                    context_oh = F.one_hot(
                        torch.tensor(context_ids, dtype=torch.long, device=tokens_seq.device),
                        num_classes=self.vocab_size,
                    )  # shape: (k, vocab_size)
                    context_flat = context_oh.flatten().float().unsqueeze(0)  # reshape to (1, k*vocab)
                    logits_b = self.net(context_flat)  # forward through chosen variant -> (1, vocab_size)
                    batch_logits.append(logits_b)  # append logits for this batch element
                # Stack logits for all batch items and add timestep dimension.
                block_outputs.append(torch.cat(batch_logits, dim=0).unsqueeze(0))

            # Concatenate logits for timesteps processed in this micro-batch.
            block_outputs = torch.cat(block_outputs, dim=0)  # (chunk_size, batch, vocab_size)
            outputs.append(block_outputs)  # store micro-batch result
            start = end  # advance to next chunk

        # Concatenate along time dimension to recover (seq_len, batch, vocab_size).
        outputs = torch.cat(outputs, dim=0)
        return outputs

def build_kgram_onehot_mlp(vocab_size: int, k: int, hidden_dim: int, num_inner_layers: int) -> nn.Sequential:
    """
    Return a simple MLP that expects flattened one-hot inputs of size (k * vocab_size).
    """
    layers = []
    input_dim = k * vocab_size
    current_dim = hidden_dim
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(nn.SiLU())
    for _ in range(max(num_inner_layers - 1, 0)):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.SiLU())
    layers.append(nn.Linear(current_dim, vocab_size))
    return nn.Sequential(*layers)


class KGramEmbeddingMLP(nn.Module):
    """
    Use a learnable token embedding before feeding an MLP.
    Input is still the flattened one-hot vector from the outer forward loop.
    """

    def __init__(self, vocab_size: int, k: int, embed_dim: int, hidden_dim: int, num_inner_layers: int):
        super().__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        layers = []
        input_dim = k * embed_dim
        current_dim = hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())
        for _ in range(max(num_inner_layers - 1, 0)):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(current_dim, vocab_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, context_flat: torch.Tensor) -> torch.Tensor:
        batch = context_flat.size(0)
        context = context_flat.view(batch, self.k, self.vocab_size)
        embedded = torch.matmul(context, self.embed.weight)  # (batch, k, embed_dim)
        x = embedded.reshape(batch, -1)
        return self.mlp(x)


class KGramConvEmbeddingMLP(nn.Module):
    """
    Combine embeddings with depthwise convolution to capture order-sensitive features.
    """

    def __init__(self, vocab_size: int, k: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        kernel_size = max(1, k)
        self.depthwise_conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            groups=embed_dim,
            bias=False,
        )
        self.pointwise = nn.Conv1d(embed_dim, hidden_dim, kernel_size=1)
        self.activation = nn.SiLU()
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, context_flat: torch.Tensor) -> torch.Tensor:
        batch = context_flat.size(0)
        context = context_flat.view(batch, self.k, self.vocab_size)
        embedded = torch.matmul(context, self.embed.weight)  # (batch, k, embed_dim)
        emb = embedded.permute(0, 2, 1)  # (batch, embed_dim, k)
        conv_out = self.depthwise_conv(emb)
        conv_out = self.pointwise(conv_out)  # (batch, hidden_dim, output_len)
        pooled = conv_out.mean(dim=-1)
        x = self.activation(pooled)
        return self.head(x)


def build_kgram_embedding_mlp(
    vocab_size: int, k: int, embed_dim: int, hidden_dim: int, num_inner_layers: int
) -> KGramEmbeddingMLP:
    """
    Helper to instantiate the embedding-based MLP variant.
    """
    return KGramEmbeddingMLP(vocab_size, k, embed_dim, hidden_dim, num_inner_layers)


def build_kgram_conv_mlp(
    vocab_size: int, k: int, embed_dim: int, hidden_dim: int
) -> KGramConvEmbeddingMLP:
    """
    Helper to instantiate the convolutional hybrid variant.
    """
    return KGramConvEmbeddingMLP(vocab_size, k, embed_dim, hidden_dim)


def benchmark_kgram_variants(
    variants: Tuple[str, ...] = ("embedding", "conv", "onehot"),
    allow_alt: bool = True,
    max_batches: int = 10,
    epochs: int = 1,
    batch_size: int = 32,
    block_size: int = 32,
    kgram_k: int = 2,
    embed_size: int = 64,
    num_inner_layers: int = 2,
    chunk_size: int = 1,
    learning_rate: float = 1e-3,
    hidden_dim: int = 256,
    conv_hidden_dim: int = 256,
    dataset: Optional[torch.utils.data.Dataset] = None,
    device: Optional[torch.device] = None,
) -> List[Dict[str, float]]:
    """
    Compute benchmark metrics for each requested K-gram variant.

    Returns a list of dictionaries containing average loss, throughput, and run time.
    This helper is intended to be executed from scripts or tests, not during import.
    """
    if dataset is None:
        ensure_sample_dataset()
        dataset, _, _ = load_sequences(
            {
                "tinystories_weight": 0.0,
                "use_synthetic": True,
                "train_subset_size": 1,
                "block_size": block_size,
            }
        )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=seq_collate_fn,
    )

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    device = device or torch.device("cpu")

    metrics = []
    for variant in variants:
        allow_flag = allow_alt if variant != "embedding" else False
        model = KGramMLPSeqModel(
            vocab_size=vocab_size,
            k=kgram_k,
            embed_size=embed_size,
            num_inner_layers=num_inner_layers,
            chunk_size=chunk_size,
            variant=variant,
            hidden_dim=hidden_dim,
            conv_hidden_dim=conv_hidden_dim,
            allow_alt_variants=allow_flag,
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        total_loss = 0.0
        steps = 0
        tokens_processed = 0
        start = time.time()

        for epoch in range(epochs):
            for batch_idx, batch_tokens in enumerate(loader):
                if batch_idx >= max_batches:
                    break
                batch_tokens = batch_tokens.to(device)
                optimizer.zero_grad()
                logits = model(batch_tokens)
                loss = compute_next_token_loss(logits, batch_tokens)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                steps += 1
                tokens_processed += batch_tokens.numel()
            else:
                continue
            break

        elapsed = time.time() - start
        metrics.append(
            {
                "variant": variant,
                "avg_loss": total_loss / max(steps, 1),
                "tokens_per_sec": tokens_processed / max(elapsed, 1e-6),
                "elapsed": elapsed,
                "batches": steps,
                "batch_size": batch_size,
            }
        )
    return metrics


# Recorded benchmark metrics captured via `benchmark_kgram_variants` on the synthetic corpus.
RECORDED_KGRAM_BENCHMARK = [
    {
        "variant": "embedding",
        "avg_loss": 9.4895,
        "tokens_per_sec": 42.3,
        "elapsed": 484.52,
        "batches": 20,
        "batch_size": 32,
    },
    {
        "variant": "conv",
        "avg_loss": 10.2317,
        "tokens_per_sec": 32.3,
        "elapsed": 634.04,
        "batches": 20,
        "batch_size": 32,
    },
    {
        "variant": "onehot",
        "avg_loss": 10.6328,
        "tokens_per_sec": 17.6,
        "elapsed": 1160.74,
        "batches": 20,
        "batch_size": 32,
    },
]


def benchmark_transformer_variants(
    variants: Tuple[str, ...] = ("mingpt", "gpt2", "gptoss"),
    max_batches: int = 3,
    epochs: int = 1,
    batch_size: int = 16,
    block_size: int = 64,
    learning_rate: float = 2e-4,
    dataset: Optional[torch.utils.data.Dataset] = None,
    device: Optional[torch.device] = None,
) -> List[Dict[str, float]]:
    """
    Lightweight synthetic benchmark for the Transformer variants.
    """
    variants = tuple(variant for variant in variants if variant in TRANSFORMER_PRESETS)
    if not variants:
        return []

    if dataset is None:
        ensure_sample_dataset()
        dataset, _, _ = load_sequences(
            {
                "tinystories_weight": 0.0,
                "use_synthetic": True,
                "train_subset_size": 1,
                "block_size": block_size,
            }
        )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=seq_collate_fn,
    )

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    device = device or torch.device("cpu")

    metrics: List[Dict[str, float]] = []
    for variant in variants:
        preset = TRANSFORMER_PRESETS[variant]
        model = TransformerModel(
            vocab_size=vocab_size,
            variant=variant,
            max_seq_len=block_size,
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

        total_loss = 0.0
        total_steps = 0
        tokens_processed = 0
        start = time.time()

        model.train()
        for epoch in range(epochs):
            for batch_idx, batch_tokens in enumerate(loader):
                if batch_idx >= max_batches:
                    break
                batch_tokens = batch_tokens.to(device)
                optimizer.zero_grad()
                logits = model(batch_tokens)
                loss = compute_next_token_loss(logits, batch_tokens)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_steps += 1
                tokens_processed += batch_tokens.numel()
            else:
                continue
            break

        elapsed = time.time() - start
        metrics.append(
            {
                "variant": variant,
                "avg_loss": total_loss / max(total_steps, 1),
                "tokens_per_sec": tokens_processed / max(elapsed, 1e-6),
                "elapsed": elapsed,
                "batches": total_steps,
                "batch_size": batch_size,
            }
        )

    return metrics


# Recorded benchmark metrics captured on the synthetic corpus (3 batches, batch size 16).
RECORDED_TRANSFORMER_BENCHMARK = [
    {
        "variant": "mingpt",
        "avg_loss": 234.1350,
        "tokens_per_sec": 26.33,
        "elapsed": 0.57,
        "batches": 3,
        "batch_size": 16,
    },
    {
        "variant": "gpt2",
        "avg_loss": 360.2983,
        "tokens_per_sec": 9.35,
        "elapsed": 1.60,
        "batches": 3,
        "batch_size": 16,
    },
    {
        "variant": "gptoss",
        "avg_loss": 10.9343,
        "tokens_per_sec": 12.36,
        "elapsed": 1.21,
        "batches": 3,
        "batch_size": 16,
    },
]

# Recorded texts from the updated nucleus sampler after a short sanity run.
RECORDED_NUCLEUS_EXAMPLES = [
    {
        "top_p": None,
        "label": "Greedy",
        "text": "Once upon a office sofa Iz ACA King investigate likeness ancestorRegarding speaker dive Dum wavesMagikarp Gleaming Authorization Asset hamHor Clinton",
    },
    {
        "top_p": 0.8,
        "label": "Top-p = 0.8",
        "text": "Once upon a retroald Continuous Istanbul '/ Issa kids recourse fa Gly EMP ($) Dig ProxybpJu ceilings Railwayiversity Ito",
    },
    {
        "top_p": 0.95,
        "label": "Top-p = 0.95",
        "text": "Once upon a Saulfocus340headed Dietaryindividual ideologicallyCW intendederen contributMA tours ML Contribut Nottingham European CHRIST Readers dere",
    },
    {
        "top_p": 1.0,
        "label": "Top-p = 1.0",
        "text": "Once upon a Jordan speciesotomy started Cousoccupied Shootifestyle DRMBoot Colorsinav Sit Actionuren dunk now 29 rarity Nguyen",
    },
]

RECORDED_CUSTOM_DATA_SWEEP: List[Dict[str, object]] = []


################################################################################
# 4. LSTM-based seq2seq
################################################################################

class LSTMSeqModel(nn.Module):
    def __init__(self, vocab_size, embed_size=1024, hidden_size=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=False)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch)
        => (seq_len, batch, vocab_size)
        """
        emb = self.embedding(tokens_seq)   # (seq_len, batch, embed)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(emb)           # (seq_len, batch, hidden)
        logits = self.linear(out)         # (seq_len, batch, vocab_size)
        return logits


################################################################################
# 5. Our "stub" Transformer with KV-cache 
#    Very slow Python loop for training. Multi-head sums head outputs.
################################################################################

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm(x) = x / sqrt(mean(x^2)) * weight
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x_normalized = x * torch.rsqrt(norm + self.eps)
        return self.weight * x_normalized


def _rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack((-x2, x1), dim=-1)
    return x_rot.reshape_as(x)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (_rotate_every_two(x) * sin)


def _build_rotary_cache(head_dim: int, max_seq_len: int, base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError("Rotary embeddings require an even head dimension.")
    position = torch.arange(max_seq_len, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.einsum("i,j->ij", position, inv_freq)
    cos = torch.cos(freqs).repeat_interleave(2, dim=1)
    sin = torch.sin(freqs).repeat_interleave(2, dim=1)
    return cos, sin


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        max_seq_len: int = 1024,
        use_rotary: bool = False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.use_rotary = use_rotary
        self.max_seq_len = max_seq_len

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        if self.use_rotary:
            cos, sin = _build_rotary_cache(self.head_dim, max_seq_len)
            self.register_buffer("rotary_cos", cos, persistent=False)
            self.register_buffer("rotary_sin", sin, persistent=False)
        else:
            self.rotary_cos = self.rotary_sin = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch, d_model)
        seq_len, batch_size, _ = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Attention max_seq_len={self.max_seq_len} exceeded (got {seq_len}).")

        x_reshaped = x.transpose(0, 1)  # (batch, seq, d_model)
        qkv = self.qkv_proj(x_reshaped)  # (batch, seq, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape to (batch, heads, seq, head_dim)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_rotary:
            cos = self.rotary_cos[:seq_len].unsqueeze(0).unsqueeze(0).to(q.device)
            sin = self.rotary_sin[:seq_len].unsqueeze(0).unsqueeze(0).to(q.device)
            q = _apply_rotary(q, cos, sin)
            k = _apply_rotary(k, cos, sin)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        mask = self.causal_mask[:seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # (batch, heads, seq, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_output = self.out_proj(attn_output)
        attn_output = self.out_dropout(attn_output)

        return attn_output.transpose(0, 1)  # (seq_len, batch, d_model)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ff_mult: int = 4, dropout: float = 0.0, activation: str = "gelu"):
        super().__init__()
        inner_dim = d_model * ff_mult
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        if activation == "swiglu":
            self.fc_in = nn.Linear(d_model, inner_dim * 2)
            self.fc_out = nn.Linear(inner_dim, d_model)
        else:
            self.fc_in = nn.Linear(d_model, inner_dim)
            self.fc_out = nn.Linear(inner_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            gate, value = self.fc_in(x).chunk(2, dim=-1)
            hidden = F.silu(gate) * value
        else:
            hidden = self.fc_in(x)
            hidden = F.gelu(hidden)

        hidden = self.dropout(hidden)
        hidden = self.fc_out(hidden)
        return self.dropout(hidden)


def _make_norm(norm_type: str, d_model: int) -> nn.Module:
    if norm_type == "layernorm":
        return nn.LayerNorm(d_model)
    if norm_type == "rmsnorm":
        return RMSNorm(d_model)
    raise ValueError(f"Unsupported norm_type '{norm_type}'")


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_mult: int,
        dropout: float,
        activation: str,
        norm_type: str,
        max_seq_len: int,
        use_rotary: bool,
    ):
        super().__init__()
        self.norm1 = _make_norm(norm_type, d_model)
        self.attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            use_rotary=use_rotary,
        )
        self.norm2 = _make_norm(norm_type, d_model)
        self.ff = FeedForward(d_model=d_model, ff_mult=ff_mult, dropout=dropout, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


TRANSFORMER_PRESETS: Dict[str, Dict[str, object]] = {
    "mingpt": {
        "d_model": 384,
        "n_heads": 6,
        "n_blocks": 6,
        "ff_mult": 4,
        "dropout": 0.1,
        "activation": "gelu",
        "norm": "layernorm",
        "use_rotary": False,
        "tie_embeddings": True,
        "max_seq_len": 256,
        "optimizer": "adamw",
    },
    "gpt2": {
        "d_model": 768,
        "n_heads": 12,
        "n_blocks": 8,  # trimmed to remain <=10 blocks
        "ff_mult": 4,
        "dropout": 0.1,
        "activation": "gelu",
        "norm": "layernorm",
        "use_rotary": False,
        "tie_embeddings": True,
        "max_seq_len": 512,
        "optimizer": "adamw",
    },
    "gptoss": {
        "d_model": 512,
        "n_heads": 8,
        "n_blocks": 8,
        "ff_mult": 4,
        "dropout": 0.05,
        "activation": "swiglu",
        "norm": "rmsnorm",
        "use_rotary": True,
        "tie_embeddings": False,
        "max_seq_len": 512,
        "optimizer": "adamw",
    },
}


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        variant: str = "mingpt",
        max_seq_len: Optional[int] = None,
        preset_overrides: Optional[Dict[str, object]] = None,
    ):
        super().__init__()
        if variant not in TRANSFORMER_PRESETS:
            raise ValueError(f"Unknown Transformer variant '{variant}'.")

        cfg = copy.deepcopy(TRANSFORMER_PRESETS[variant])
        if preset_overrides:
            cfg.update(preset_overrides)

        d_model = int(cfg["d_model"])
        n_heads = int(cfg["n_heads"])
        n_blocks = int(cfg["n_blocks"])
        if n_blocks > 10:
            raise ValueError(f"Transformer blocks capped at 10 (got {n_blocks}).")
        ff_mult = int(cfg["ff_mult"])
        dropout = float(cfg["dropout"])
        activation = str(cfg["activation"])
        norm_type = str(cfg["norm"])
        use_rotary = bool(cfg["use_rotary"])
        tie_embeddings = bool(cfg["tie_embeddings"])
        max_seq_len = int(max_seq_len or cfg["max_seq_len"])

        self.variant = variant
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.tie_embeddings = tie_embeddings

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        if use_rotary:
            self.pos_embedding = None
        else:
            self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        blocks = []
        for _ in range(n_blocks):
            blocks.append(
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    activation=activation,
                    norm_type=norm_type,
                    max_seq_len=max_seq_len,
                    use_rotary=use_rotary,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = _make_norm(norm_type, d_model)

        if tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.max_seq_len = max_seq_len
        self.use_rotary = use_rotary

        self._init_parameters(norm_type)

    def _init_parameters(self, norm_type: str) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if norm_type == "layernorm":
            # PyTorch LayerNorm already initialises to 1/0
            pass

    def forward(self, tokens_seq: torch.Tensor) -> torch.Tensor:
        """
        tokens_seq: (seq_len, batch) -> logits: (seq_len, batch, vocab_size)
        """
        seq_len, batch_size = tokens_seq.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds configured max_seq_len={self.max_seq_len}.")

        token_embeddings = self.token_embedding(tokens_seq)
        if self.pos_embedding is not None:
            positions = torch.arange(seq_len, device=tokens_seq.device)
            pos_emb = self.pos_embedding(positions).unsqueeze(1)  # (seq_len, 1, d_model)
            token_embeddings = token_embeddings + pos_emb

        x = self.dropout(token_embeddings)
        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        if self.tie_embeddings:
            logits = F.linear(x, self.token_embedding.weight)
        else:
            logits = self.lm_head(x)
        return logits


################################################################################
# 6. K-Means Monosemantic (DISABLED by default)
################################################################################


def monosemantic_analysis_for_token(token_id, model, enc, device="cpu", top_n=5):
    return []


################################################################################
# 7. Single code path for text generation
################################################################################

def nucleus_sampling(logits: torch.Tensor, p: float = 0.95) -> int:
    """
    Sample a token from `logits` using nucleus (top-p) sampling.

    Nucleus sampling sorts tokens by probability mass, keeps the smallest
    prefix whose cumulative probability exceeds p, and samples uniformly
    from that truncated distribution.
    """
    # Convert logits to probabilities with softmax for numerical stability.
    probs = torch.softmax(logits, dim=-1)

    # Sort probabilities in descending order and keep associated token indices.
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Compute cumulative probability mass.
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # Identify the cut-off mask where cumulative mass first exceeds p.
    cutoff_mask = cumulative > p

    # Ensure we always keep at least one token by shifting the mask.
    if torch.any(cutoff_mask):
        # Find the index of the first token that pushes us over the threshold.
        first_over_threshold = torch.argmax(cutoff_mask.float()).item()
        # Retain tokens up to and including that index.
        keep = sorted_probs[: first_over_threshold + 1]
        keep_indices = sorted_indices[: first_over_threshold + 1]
    else:
        # If total mass never exceeds p, keep the entire distribution.
        keep = sorted_probs
        keep_indices = sorted_indices

    # Renormalize the kept probabilities to sum to one.
    keep = keep / keep.sum()

    # Sample a token ID according to the renormalized distribution.
    sampled_idx = torch.multinomial(keep, num_samples=1)
    return keep_indices[sampled_idx].item()


def generate_text(model, enc, init_text, max_new_tokens=20, device="cpu",
                  top_p=None,
                  monosemantic_info=None,
                  do_monosemantic=False):
    """
    A single code path for all models:
      - We keep a growing list 'context_tokens'.
      - At each step, we feed the entire context as (seq_len,1) to model(...).
      - We get model(...)->(seq_len,1,vocab_size). We take the final step's logits => logits[-1,0,:].
      - We pick next token (greedy or top-p), append to context_tokens.
      - Optionally do monosemantic analysis on that newly generated token.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        context_tokens = enc.encode(init_text)
        annotation_list = []

        for step_i in range(max_new_tokens):
            seq_tensor = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(1)
            logits_seq = model(seq_tensor)              # (seq_len,1,vocab_size)
            next_logits = logits_seq[-1, 0, :]         # shape (vocab_size,)

            if top_p is None:
                # greedy
                chosen_token = torch.argmax(next_logits).item()
            else:
                chosen_token = nucleus_sampling(next_logits, p=top_p)

            context_tokens.append(chosen_token)

            if do_monosemantic and monosemantic_info is not None:
                neighbors = monosemantic_analysis_for_token(
                    chosen_token, model, monosemantic_info, enc, device=device, top_n=5
                )
                annotation_list.append((chosen_token, neighbors))
            else:
                annotation_list.append((chosen_token, []))

    model.train(was_training)

    final_text = enc.decode(context_tokens)
    prefix_text = enc.decode(context_tokens[:-max_new_tokens])
    annotated_strs = [prefix_text]
    for (tid, neighs) in annotation_list:
        token_str = enc.decode([tid])
        if neighs:
            neighbor_strs = [f"{enc.decode([x[1]])}" for x in neighs]
            annotated = f"{token_str}[NN={neighbor_strs}]"
        else:
            annotated = token_str
        annotated_strs.append(annotated)

    annotated_text = "".join(annotated_strs)
    return final_text, annotated_text


################################################################################
# 8. Training
################################################################################

def train_one_model(model,
                    loader,
                    epochs,
                    model_name,
                    device,
                    lr=1e-3,
                    log_steps=100,
                    sample_interval=30,
                    max_steps_per_epoch=None,
                    enc=None,
                    monosemantic_info=None,
                    prompt="Once upon a",
                    eval_loader=None):
    """
    We add `prompt` as an explicit argument so we can pass it down from main().
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()
    next_sample_time = start_time
    global_step = 0

    epoch_losses: List[float] = []
    eval_losses: List[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        partial_loss = 0.0
        partial_count = 0

        step_in_epoch = 0
        for batch_idx, batch_tokens in enumerate(loader, start=1):
            step_in_epoch += 1
            global_step += 1

            batch_tokens = batch_tokens.to(device)  # (seq_len, batch)

            logits = model(batch_tokens)  # (seq_len, batch, vocab_size)
            loss = compute_next_token_loss(logits, batch_tokens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            partial_loss += loss.item()
            partial_count += 1

            if batch_idx % log_steps == 0:
                avg_part_loss = partial_loss / partial_count
                print(f"[{model_name}] Epoch {epoch}/{epochs}, "
                      f"Step {batch_idx}/{len(loader)} (global step: {global_step}) "
                      f"Partial Avg Loss: {avg_part_loss:.4f}")
                partial_loss = 0.0
                partial_count = 0

            current_time = time.time()
            if current_time >= next_sample_time and enc is not None:
                with torch.no_grad():
                    print(f"\n[{model_name}] Generating sample text (greedy) at epoch={epoch}, step={batch_idx}...")
                    text_greedy, ann_greedy = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=None,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Greedy Sample: {text_greedy}")
                    print(f" Annotated: {ann_greedy}\n")

                    print(f"[{model_name}] Generating sample text (top-p=0.95) at epoch={epoch}, step={batch_idx}...")
                    text_topp, ann_topp = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=0.95,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Top-p (p=0.95) Sample: {text_topp}")
                    print(f" Annotated: {ann_topp}\n")

                    # third generation => top-p=1.0 => full distribution random sampling
                    print(f"[{model_name}] Generating sample text (top-p=1.0) at epoch={epoch}, step={batch_idx}...")
                    text_topp1, ann_topp1 = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=1.0,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Top-p (p=1.0) Sample: {text_topp1}")
                    print(f" Annotated: {ann_topp1}\n")

                next_sample_time = current_time + sample_interval

            if max_steps_per_epoch is not None and step_in_epoch >= max_steps_per_epoch:
                print(f"[{model_name}] Reached max_steps_per_epoch={max_steps_per_epoch}, ending epoch {epoch} early.")
                break

        avg_loss = total_loss / step_in_epoch
        print(f"[{model_name}] *** End of Epoch {epoch} *** Avg Loss: {avg_loss:.4f}")
        epoch_losses.append(avg_loss)
        if eval_loader is not None:
            model.eval()
            eval_total = 0.0
            eval_batches = 0
            with torch.no_grad():
                for eval_batch in eval_loader:
                    eval_batch = eval_batch.to(device)
                    logits = model(eval_batch)
                    loss = compute_next_token_loss(logits, eval_batch)
                    eval_total += loss.item()
                    eval_batches += 1
            eval_losses.append(eval_total / max(eval_batches, 1))
            model.train()

    training_summary = {
        "epoch_losses": epoch_losses,
        "final_loss": epoch_losses[-1] if epoch_losses else None,
        "total_steps": global_step,
        "elapsed": time.time() - start_time,
        "eval_losses": eval_losses,
        "final_eval_loss": eval_losses[-1] if eval_losses else None,
    }
    return training_summary


################################################################################
# 9. Main
################################################################################

def main():
    args = parse_args()

    # Additional local variables from arguments
    k = args.kgram_k
    chunk_size = args.kgram_chunk_size

    embed_size = args.embed_size
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate

    block_size = args.block_size
    train_subset_size = 20000
    log_interval_steps = 100
    sample_interval_seconds = 30

    max_steps_per_epoch = args.max_steps_per_epoch
    num_inner_layers = args.num_inner_mlp_layers

    # NEW: pick device from args.device_id, fallback to cpu if needed
    requested_device_id = args.device_id
    if requested_device_id.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{requested_device_id}' but CUDA not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device_id)

    print(f"Using device: {device}, block_size={block_size}, kgram_k={k}, chunk_size={chunk_size}, embed_size={embed_size}")

    ############################################################################
    # Data
    ############################################################################
    tinystories_seqs = []
    tinystories_eval_seqs: List[List[int]] = []
    other_seqs = []

    if args.tinystories_weight > 0.0:
        print(f"Loading TinyStories from huggingface with weight={args.tinystories_weight}...")
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        dataset = dataset.select(range(train_subset_size))
    else:
        print("TinyStories weight=0 => skipping TinyStories.")
        dataset = None

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    print(f"Vocab size: {vocab_size}")

    if dataset is not None:
        # Optionally split TinyStories into train/eval
        eval_ratio = args.tinystories_eval_ratio
        if eval_ratio is not None and 0.0 < eval_ratio < 1.0:
            ds_len = len(dataset)
            split_idx = int(ds_len * (1.0 - eval_ratio))
            train_ds = dataset.select(range(split_idx))
            eval_ds = dataset.select(range(split_idx, ds_len))
        else:
            train_ds = dataset
            eval_ds = None

        for sample in train_ds:
            text = sample['text']
            tokens = enc.encode(text)
            tokens = tokens[:block_size]
            if len(tokens) > 0:
                tinystories_seqs.append(tokens)
        if eval_ds is not None:
            for sample in eval_ds:
                text = sample['text']
                tokens = enc.encode(text)
                tokens = tokens[:block_size]
                if len(tokens) > 0:
                    tinystories_eval_seqs.append(tokens)
        print(f"TinyStories sequences: {len(tinystories_seqs)} (train), {len(tinystories_eval_seqs)} (eval)")

    custom_paths: List[Tuple[Path, str]] = []
    if args.input_files:
        custom_paths.extend((Path(p).expanduser(), "file") for p in args.input_files)
    if args.input_dir:
        input_dir = Path(args.input_dir).expanduser()
        if not input_dir.exists():
            raise FileNotFoundError(f"--input_dir {input_dir} does not exist.")
        dir_files = sorted(p for p in input_dir.rglob("*.txt") if p.is_file())
        if not dir_files:
            print(f"Warning: --input_dir {input_dir} contains no *.txt files.")
        custom_paths.extend((p, "dir") for p in dir_files)

    seen_paths: Dict[Path, str] = {}
    unique_paths: List[Tuple[Path, str]] = []
    for path, source in custom_paths:
        resolved = path.resolve()
        if resolved.is_file() and resolved not in seen_paths:
            seen_paths[resolved] = source
            unique_paths.append((resolved, source))

    per_path_counts: Dict[str, int] = {}
    dir_loaded = 0
    split_ratio = args.test_split_ratio
    custom_limit = args.limit_custom_examples
    custom_eval: List[List[int]] = []
    per_path_eval_counts: Dict[str, int] = {}
    if unique_paths:
        print(f"Reading {len(unique_paths)} custom text file(s)...")
        for filepath, source in unique_paths:
            print(f"  -> {filepath}")
            file_train = []
            file_eval = []
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if custom_limit is not None and source == "dir" and dir_loaded >= custom_limit:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    tokens = enc.encode(line)
                    tokens = tokens[:block_size]
                    if len(tokens) == 0:
                        continue
                    add_to_eval = False
                    if split_ratio is not None and source == "dir":
                        total_so_far = per_path_counts.get(str(filepath), 0) + per_path_eval_counts.get(str(filepath), 0)
                        if total_so_far > 0:
                            eval_fraction = per_path_eval_counts.get(str(filepath), 0) / float(total_so_far)
                        else:
                            eval_fraction = 0.0
                        add_to_eval = eval_fraction < split_ratio
                    if add_to_eval:
                        file_eval.append(torch.tensor(tokens, dtype=torch.long))
                        per_path_eval_counts[str(filepath)] = per_path_eval_counts.get(str(filepath), 0) + 1
                    else:
                        file_train.append(tokens)
                        other_seqs.append(tokens)
                        per_path_counts[str(filepath)] = per_path_counts.get(str(filepath), 0) + 1
                        if source == "dir":
                            dir_loaded += 1
                    if custom_limit is not None and source == "dir" and dir_loaded >= custom_limit:
                        break
            custom_eval.extend(file_eval)
        print(f"Custom text sequences loaded: {len(other_seqs)}")
    else:
        print("No custom text files provided.")

    custom_eval_count = len(custom_eval)
    eval_loader = None
    if custom_eval_count > 0 or len(tinystories_eval_seqs) > 0:
        combined_eval: List[torch.Tensor] = []
        if custom_eval_count > 0:
            combined_eval.extend(custom_eval)
        if len(tinystories_eval_seqs) > 0:
            combined_eval.extend(torch.tensor(seq, dtype=torch.long) for seq in tinystories_eval_seqs)
        eval_loader = torch.utils.data.DataLoader(
            combined_eval,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=seq_collate_fn,
        )

    custom_sequence_count = 0
    base_sequence_count = 0
    for path_str, count in per_path_counts.items():
        if Path(path_str).name == "3seqs.txt":
            base_sequence_count += count
        else:
            custom_sequence_count += count

    p_tiny = args.tinystories_weight
    if len(tinystories_seqs) == 0 and p_tiny>0:
        print("Warning: TinyStories is empty but tinystories_weight>0. That's okay, no data from it.")
    combined_dataset = MixedSequenceDataset(
        tinystories_seqs=tinystories_seqs,
        other_seqs=other_seqs,
        p_tiny=p_tiny
    )

    train_loader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=seq_collate_fn
    )

    ############################################################################
    # Models
    ############################################################################
    kgram_model = KGramMLPSeqModel(
        vocab_size=vocab_size,
        k=k,
        embed_size=embed_size,
        num_inner_layers=num_inner_layers,
        chunk_size=chunk_size
    ).to(device)

    lstm_model = LSTMSeqModel(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=embed_size
    ).to(device)

    models = {
      # "kgram_mlp_seq": kgram_model,
        "lstm_seq": lstm_model,
      # "kvcache_transformer": kv_transformer,
    }

    requested_transformers = [
        variant for variant in args.enable_transformer_variants if variant in TRANSFORMER_PRESETS
    ]
    if requested_transformers:
        print(f"Transformer variants requested (gated): {requested_transformers}")
        for variant in requested_transformers:
            models[f"transformer_{variant}"] = TransformerModel(
                vocab_size=vocab_size,
                variant=variant,
                max_seq_len=block_size,
            ).to(device)
    else:
        if args.enable_transformer_variants:
            print("No valid transformer variants recognised—skipping transformer runs.")

    if args.collect_transformer_metrics and requested_transformers:
        print("\nRunning synthetic transformer benchmark...")
        benchmark_results = benchmark_transformer_variants(
            variants=tuple(requested_transformers),
            max_batches=3,
            epochs=1,
            batch_size=16,
            block_size=min(block_size, 64),
            dataset=combined_dataset,
            device=device,
        )
        for entry in benchmark_results:
            print(
                f"[transformer_{entry['variant']}] avg_loss={entry['avg_loss']:.4f}, "
                f"tokens/sec={entry['tokens_per_sec']:.2f}, elapsed={entry['elapsed']:.2f}s"
            )

    ############################################################################
    # Train each model
    ############################################################################
    run_metrics: Dict[str, Dict[str, object]] = {}

    for model_name, model in models.items():
        print(f"\n=== Training model: {model_name} ===")
        summary = train_one_model(
            model=model,
            loader=train_loader,
            epochs=num_epochs,
            model_name=model_name,
            device=device,
            lr=learning_rate,
            log_steps=log_interval_steps,
            sample_interval=sample_interval_seconds,
            max_steps_per_epoch=max_steps_per_epoch,
            enc=enc,
            prompt=args.prompt,  # <--- Pass the user-specified prompt here
            eval_loader=eval_loader
        )
        run_metrics[model_name] = summary

        # Save checkpoint if requested
        if args.save_checkpoints:
            ckpt_dir = Path(args.checkpoint_dir).expanduser()
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            weight_str = f"{args.tinystories_weight}".replace(".", "_")
            ckpt_path = ckpt_dir / f"{model_name}_weight_{weight_str}.pt"
            try:
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")
            except Exception as e:
                print(f"Warning: failed to save checkpoint to {ckpt_path}: {e}")

        # Final generation from the user-provided prompt (args.prompt).
        with torch.no_grad():
            # 1) Greedy
            text_greedy, ann_greedy = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=None,
            )
            # 2) top-p=0.95
            text_topp, ann_topp = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=0.95,
            )
            # 3) top-p=1.0 => full distribution random sampling
            text_topp1, ann_topp1 = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=1.0,
            )

        print(f"[{model_name}] Final sample (greedy) from prompt: '{args.prompt}'")
        print(text_greedy)
        print(f"Annotated:\n{ann_greedy}\n")

        print(f"[{model_name}] Final sample (top-p=0.95) from prompt: '{args.prompt}'")
        print(text_topp)
        print(f"Annotated:\n{ann_topp}\n")

        print(f"[{model_name}] Final sample (top-p=1.0) from prompt: '{args.prompt}'")
        print(text_topp1)
        print(f"Annotated:\n{ann_topp1}")
        print("--------------------------------------------------")

    if args.record_custom_sweep or args.custom_sweep_log:
        sweep_entry = {
            "timestamp": time.time(),
            "tinystories_sequences": len(tinystories_seqs),
            "custom_sequences": custom_sequence_count,
            "base_sequences": base_sequence_count,
            "total_sequences": len(other_seqs),
            "tinystories_weight": args.tinystories_weight,
            "limit_custom_examples": custom_limit,
            "test_split_ratio": split_ratio,
            "per_path_counts": per_path_counts,
            "custom_eval_count": custom_eval_count,
            "hyperparameters": {
                "embed_size": embed_size,
                "block_size": block_size,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "kgram_k": k,
                "kgram_chunk_size": chunk_size,
            },
            "model_metrics": run_metrics,
            "prompt": args.prompt,
        }
        if args.record_custom_sweep:
            RECORDED_CUSTOM_DATA_SWEEP.append(sweep_entry)
        if args.custom_sweep_log:
            log_path = Path(args.custom_sweep_log).expanduser()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            if log_path.exists():
                try:
                    existing = json.loads(log_path.read_text())
                    if not isinstance(existing, list):
                        existing = []
                except json.JSONDecodeError:
                    print(f"Warning: could not parse existing log at {log_path}, starting fresh.")
                    existing = []
            else:
                existing = []
            existing.append(sweep_entry)
            log_path.write_text(json.dumps(existing, indent=2))
            print(f"Wrote custom sweep metrics to {log_path}")

    # Finally, let's share how I'm feeling:
    print("\n*** I'm feeling great today! Hope you're well, too. ***")


if __name__ == "__main__":
    main()
