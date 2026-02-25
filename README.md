# Minimaler Encoder-Decoder-Translator (PyTorch)

Dieses Projekt enthaelt einen bewusst einfachen Seq2Seq-Translator mit:

- GRU-Encoder
- GRU-Decoder
- Multi-Head Scaled Dot-Product Cross-Attention mit lernbaren Projektionen (`W_q`, `W_k`, `W_v`, `W_o`)
- Teacher Forcing im Training

## Struktur

- `simple_attention_translator.py` (Entry-Point)
- `translator/constants.py` (Sondertokens)
- `translator/data.py` (Vokabular, Dataset, Collate, Toy-Korpus)
- `translator/model.py` (Encoder, Decoder, Attention, Seq2Seq)
- `translator/train.py` (Training, CLI, Beispiel-Inferenz)

## Start

```bash
python simple_attention_translator.py --epochs 120
```

## Checkpoint und Inferenz

Training speichert automatisch ein Checkpoint unter `checkpoints/translator.pt` (anpassbar mit `--checkpoint-path`).

Einzelsatz uebersetzen:

```bash
python simple_attention_translator.py --checkpoint-path checkpoints/translator.pt --translate "ich bin muede"
```

Interaktiv testen:

```bash
python simple_attention_translator.py --checkpoint-path checkpoints/translator.pt --interactive
```

## Wichtige Parameter

- `--epochs` (default: `200`)
- `--batch-size` (default: `4`)
- `--emb-dim` (default: `64`)
- `--hidden-dim` (default: `64`)
- `--lr` (default: `1e-3`)
- `--num-heads` (default: `4`)
- `--teacher-forcing` (default: `0.7`)
- `--checkpoint-path` (default: `checkpoints/translator.pt`)
- `--translate` (default: `None`)
- `--interactive` (Flag)
- `--max-len` (default: `30`, nur Inferenz)

## Hinweis

Der Datensatz ist absichtlich sehr klein (Toy-Parallelkorpus in der Datei), damit das Beispiel schnell verstaendlich und trainierbar bleibt.
