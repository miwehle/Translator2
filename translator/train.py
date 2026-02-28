import argparse
from pathlib import Path
from typing import Any, Dict, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .attention import AttentionFactory, make_simple_sdp_attention_factory, make_torch_attention_factory
from .data import Tokenizer, TranslationDataset, collate_fn, set_seed, tiny_parallel_corpus
from .model import Seq2Seq


ATTENTION_CHOICES = ("torch", "simple_sdp")


def resolve_attention_factory(attention: str) -> AttentionFactory:
    if attention == "torch":
        return make_torch_attention_factory()
    if attention == "simple_sdp":
        return make_simple_sdp_attention_factory()
    raise ValueError(f"Unbekannte Attention: {attention!r}. Erlaubt: {ATTENTION_CHOICES}.")


def build_model(
    args: argparse.Namespace,
    src_tokenizer: Tokenizer,
    tgt_tokenizer: Tokenizer,
    device: torch.device,
    attention_factory: AttentionFactory | None = None,
) -> Seq2Seq:
    if attention_factory is None:
        attention_factory = resolve_attention_factory(getattr(args, "attention", "torch"))
    model = Seq2Seq(
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
        d_model=args.emb_dim,
        ff_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        src_pad_idx=src_tokenizer.pad_token_id,
        tgt_pad_idx=tgt_tokenizer.pad_token_id,
        tgt_sos_idx=tgt_tokenizer.sos_token_id,
        dropout=args.dropout,
        attention_factory=attention_factory,
    ).to(device)
    return model


def save_checkpoint(
    path: str,
    model: Seq2Seq,
    src_tokenizer: Tokenizer,
    tgt_tokenizer: Tokenizer,
    args: argparse.Namespace,
) -> None:
    checkpoint_dir = Path(path).parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "src_tokenizer": {"stoi": src_tokenizer.stoi, "itos": src_tokenizer.itos},
        "tgt_tokenizer": {"stoi": tgt_tokenizer.stoi, "itos": tgt_tokenizer.itos},
        "hparams": {
            "emb_dim": args.emb_dim,
            "hidden_dim": args.hidden_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "attention": getattr(args, "attention", "torch"),
        },
    }
    torch.save(payload, path)
    print(f"\nCheckpoint gespeichert: {path}")


def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device)
    return ckpt


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    
    def create_data_loader(pairs, src_tokenizer, tgt_tokenizer):
        dataset = TranslationDataset(pairs, src_tokenizer, tgt_tokenizer)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(
                batch, src_tokenizer.pad_token_id, tgt_tokenizer.pad_token_id
            ),
        )
        return loader

    def print_sample_translations(model, pairs, src_tokenizer, tgt_tokenizer):
        model.eval()
        print("\nBeispiele:")
        for src_text, _ in pairs[:5]:
            src_ids = src_tokenizer.encode(src_text)
            pred_ids = model.translate(
                src_ids, max_len=20, device=device, eos_idx=tgt_tokenizer.eos_token_id
            )
            print(f"{src_text:20s} -> {tgt_tokenizer.decode(pred_ids)}")

    pairs = tiny_parallel_corpus()
    src_tokenizer = Tokenizer.build([p[0] for p in pairs])
    tgt_tokenizer = Tokenizer.build([p[1] for p in pairs])

    loader = create_data_loader(pairs, src_tokenizer, tgt_tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        args,
        src_tokenizer,
        tgt_tokenizer,
        device,
        attention_factory=resolve_attention_factory(args.attention),
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_tokenizer.pad_token_id)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)

            optim.zero_grad()
            logits = model(src, tgt)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt[:, 1:].reshape(-1),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += float(loss.item())

        print(f"Epoch {epoch:03d} | loss={total_loss / len(loader):.4f}")

    print_sample_translations(model, pairs, src_tokenizer, tgt_tokenizer)

    save_checkpoint(args.checkpoint_path, model, src_tokenizer, tgt_tokenizer, args)


def run_translate(args: argparse.Namespace, interactive: bool) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.checkpoint_path, device)

    src_data = ckpt.get("src_tokenizer")
    tgt_data = ckpt.get("tgt_tokenizer")
    if not isinstance(src_data, dict) or not isinstance(tgt_data, dict):
        raise ValueError("Checkpoint enthaelt keine gueltigen Tokenizer-Daten.")

    src_stoi = cast(dict[str, int], src_data.get("stoi"))
    src_itos = cast(list[str], src_data.get("itos"))
    tgt_stoi = cast(dict[str, int], tgt_data.get("stoi"))
    tgt_itos = cast(list[str], tgt_data.get("itos"))
    if src_stoi is None or src_itos is None or tgt_stoi is None or tgt_itos is None:
        raise ValueError("Checkpoint enthaelt unvollstaendige Tokenizer-Daten.")

    src_tokenizer = Tokenizer(stoi=src_stoi, itos=src_itos)
    tgt_tokenizer = Tokenizer(stoi=tgt_stoi, itos=tgt_itos)
    hparams = ckpt["hparams"]

    trained_attention = hparams.get("attention", "torch")
    requested_attention = getattr(args, "attention", "torch")
    if requested_attention != trained_attention:
        raise ValueError(
            f"Attention-Mismatch: Checkpoint nutzt '{trained_attention}', "
            f"CLI fordert '{requested_attention}'."
        )

    model_args = argparse.Namespace(
        emb_dim=hparams["emb_dim"],
        hidden_dim=hparams["hidden_dim"],
        num_heads=hparams["num_heads"],
        num_layers=hparams.get("num_layers", 2),
        dropout=hparams.get("dropout", 0.1),
    )
    model = build_model(
        model_args,
        src_tokenizer,
        tgt_tokenizer,
        device,
        attention_factory=resolve_attention_factory(requested_attention),
    )
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint passt nicht zur aktuellen Modellarchitektur. "
            "Bitte mit der aktuellen Transformer-Version neu trainieren."
        ) from exc
    model.eval()

    def translate_text(text: str) -> str:
        src_ids = src_tokenizer.encode(text)
        pred_ids = model.translate(
            src_ids, max_len=args.max_len, device=device, eos_idx=tgt_tokenizer.eos_token_id
        )
        return tgt_tokenizer.decode(pred_ids)

    if interactive:
        print("Interaktiver Modus. Mit 'exit' beenden.")
        while True:
            try:
                text = input("de> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                continue
            if text.lower() == "exit":
                break
            print(f"en> {translate_text(text)}")
        return

    print(translate_text(args.translate))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimaler Transformer-Translator (Pre-Norm)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--emb-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-path", type=str, default="checkpoints/translator.pt")
    p.add_argument("--attention", type=str, choices=ATTENTION_CHOICES, default="torch")
    p.add_argument("--translate", type=str, default=None)
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--max-len", type=int, default=30)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.translate and args.interactive:
        raise ValueError("Nutze entweder --translate oder --interactive, nicht beides.")

    if args.translate or args.interactive:
        run_translate(args, interactive=args.interactive)
        return

    train(args)
