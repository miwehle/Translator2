import argparse
from pathlib import Path

import torch

from translator.constants import EOS, PAD, SOS
from translator.data import Vocab, tiny_parallel_corpus
from translator.model import Seq2Seq
from translator.train import (
    build_model,
    load_checkpoint,
    run_translate,
    save_checkpoint,
    train,
)


def make_args(checkpoint_path: Path, epochs: int = 1) -> argparse.Namespace:
    return argparse.Namespace(
        epochs=epochs,
        batch_size=4,
        emb_dim=32,
        hidden_dim=64,
        lr=1e-3,
        num_heads=4,
        num_layers=1,
        dropout=0.1,
        seed=42,
        checkpoint_path=str(checkpoint_path),
        translate=None,
        interactive=False,
        max_len=20,
    )


def make_vocabs():
    pairs = tiny_parallel_corpus()
    src_vocab = Vocab.build([p[0] for p in pairs])
    tgt_vocab = Vocab.build([p[1] for p in pairs])
    return src_vocab, tgt_vocab


def test_build_model_returns_seq2seq():
    src_vocab, tgt_vocab = make_vocabs()
    args = make_args(Path("dummy.pt"))
    device = torch.device("cpu")

    model = build_model(args, src_vocab, tgt_vocab, device)

    assert isinstance(model, Seq2Seq)


def test_save_and_load_checkpoint_roundtrip(tmp_path):
    src_vocab, tgt_vocab = make_vocabs()
    ckpt_path = tmp_path / "translator_test.pt"
    args = make_args(ckpt_path)
    model = build_model(args, src_vocab, tgt_vocab, torch.device("cpu"))

    save_checkpoint(str(ckpt_path), model, src_vocab, tgt_vocab, args)
    ckpt = load_checkpoint(str(ckpt_path), torch.device("cpu"))

    assert ckpt_path.exists()
    assert "model_state_dict" in ckpt
    assert "src_vocab" in ckpt
    assert "tgt_vocab" in ckpt
    assert ckpt["hparams"]["num_heads"] == 4
    assert ckpt["hparams"]["num_layers"] == 1


def test_train_writes_checkpoint(tmp_path):
    ckpt_path = tmp_path / "trained.pt"
    args = make_args(ckpt_path, epochs=1)

    train(args)

    assert ckpt_path.exists()
    assert ckpt_path.stat().st_size > 0


def test_run_translate_prints_output(monkeypatch, tmp_path, capsys):
    src_vocab, tgt_vocab = make_vocabs()
    ckpt_path = tmp_path / "translator_test.pt"
    args = make_args(ckpt_path)
    model = build_model(args, src_vocab, tgt_vocab, torch.device("cpu"))
    save_checkpoint(str(ckpt_path), model, src_vocab, tgt_vocab, args)

    def fake_translate(self, src_ids, max_len, device, eos_idx):
        return [self.tgt_sos_idx, self.tgt_sos_idx + 3, eos_idx]

    monkeypatch.setattr(Seq2Seq, "translate", fake_translate)

    infer_args = make_args(ckpt_path)
    infer_args.translate = "ich bin muede"
    run_translate(infer_args, interactive=False)
    out = capsys.readouterr().out.strip()

    assert out != ""


def test_run_translate_interactive_exit(monkeypatch, tmp_path, capsys):
    src_vocab, tgt_vocab = make_vocabs()
    ckpt_path = tmp_path / "translator_test.pt"
    args = make_args(ckpt_path)
    model = build_model(args, src_vocab, tgt_vocab, torch.device("cpu"))
    save_checkpoint(str(ckpt_path), model, src_vocab, tgt_vocab, args)

    inputs = iter(["exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    run_translate(args, interactive=True)
    out = capsys.readouterr().out

    assert "Interaktiver Modus" in out
