import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .constants import EOS, PAD, SOS
from .data import Vocab, TranslationDataset, collate_fn, set_seed, tiny_parallel_corpus
from .model import Decoder, Encoder, Seq2Seq


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = tiny_parallel_corpus()
    src_vocab = Vocab.build([p[0] for p in pairs])
    tgt_vocab = Vocab.build([p[1] for p in pairs])

    dataset = TranslationDataset(pairs, src_vocab, tgt_vocab)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(
            batch, src_vocab.stoi[PAD], tgt_vocab.stoi[PAD]
        ),
    )

    encoder = Encoder(src_vocab.size, args.emb_dim, args.hidden_dim, src_vocab.stoi[PAD])
    decoder = Decoder(
        tgt_vocab.size,
        args.emb_dim,
        args.hidden_dim,
        tgt_vocab.stoi[PAD],
        num_heads=args.num_heads,
    )
    model = Seq2Seq(
        encoder,
        decoder,
        tgt_sos_idx=tgt_vocab.stoi[SOS],
        src_pad_idx=src_vocab.stoi[PAD],
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.stoi[PAD])
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for src, src_lens, tgt, _ in loader:
            src = src.to(device)
            src_lens = src_lens.to(device)
            tgt = tgt.to(device)

            optim.zero_grad()
            logits = model(src, src_lens, tgt, teacher_forcing=args.teacher_forcing)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt[:, 1:].reshape(-1),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += float(loss.item())

        print(f"Epoch {epoch:03d} | loss={total_loss / len(loader):.4f}")

    model.eval()
    print("\nBeispiele:")
    for src_text, _ in pairs[:5]:
        src_ids = src_vocab.encode(src_text)
        pred_ids = model.translate(
            src_ids, max_len=20, device=device, eos_idx=tgt_vocab.stoi[EOS]
        )
        print(f"{src_text:20s} -> {tgt_vocab.decode(pred_ids)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimaler Encoder-Decoder Translator mit Dot-Product-Attention")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--emb-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--teacher-forcing", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    train(parse_args())
