import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from translator.constants import EOS, PAD
from translator.data import (
    Vocab,
    TranslationDataset,
    collate_fn,
    set_seed,
    tiny_parallel_corpus,
)
from translator.train import build_model


def make_args() -> argparse.Namespace:
    return argparse.Namespace(
        emb_dim=32,
        hidden_dim=64,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
        lr=1e-3,
        batch_size=4,
        seed=42,
    )


def build_training_objects():
    args = make_args()
    set_seed(args.seed)
    device = torch.device("cpu")

    pairs = tiny_parallel_corpus()
    src_vocab = Vocab.build([p[0] for p in pairs])
    tgt_vocab = Vocab.build([p[1] for p in pairs])

    dataset = TranslationDataset(pairs, src_vocab, tgt_vocab)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(
            batch, src_vocab.stoi[PAD], tgt_vocab.stoi[PAD]
        ),
    )

    model = build_model(args, src_vocab, tgt_vocab, device)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.stoi[PAD])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    return model, loader, criterion, optimizer, src_vocab, tgt_vocab, device, pairs


def batch_loss(model, criterion, batch, device):
    src, src_lens, tgt, _ = batch
    src = src.to(device)
    src_lens = src_lens.to(device)
    tgt = tgt.to(device)
    logits = model(src, src_lens, tgt)
    return criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))


def translation_match_count(model, pairs, src_vocab, tgt_vocab, device, n_samples=5):
    count = 0
    for src_text, tgt_text in pairs[:n_samples]:
        pred_ids = model.translate(
            src_vocab.encode(src_text),
            max_len=20,
            device=device,
            eos_idx=tgt_vocab.stoi[EOS],
        )
        if tgt_vocab.decode(pred_ids) == tgt_text:
            count += 1
    return count


def test_loss_decreases_over_updates():
    model, loader, criterion, optimizer, *_ = build_training_objects()
    first_batch = next(iter(loader))

    model.eval()
    with torch.no_grad():
        initial_loss = float(batch_loss(model, criterion, first_batch, torch.device("cpu")).item())

    model.train()
    for _ in range(40):
        for batch in loader:
            optimizer.zero_grad()
            loss = batch_loss(model, criterion, batch, torch.device("cpu"))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        final_loss = float(batch_loss(model, criterion, first_batch, torch.device("cpu")).item())

    assert final_loss < initial_loss


def test_translation_quality_improves_after_training():
    model, loader, criterion, optimizer, src_vocab, tgt_vocab, device, pairs = build_training_objects()

    model.eval()
    before = translation_match_count(model, pairs, src_vocab, tgt_vocab, device)

    model.train()
    for _ in range(60):
        for batch in loader:
            optimizer.zero_grad()
            loss = batch_loss(model, criterion, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    after = translation_match_count(model, pairs, src_vocab, tgt_vocab, device)

    assert after > before
