# Minimaler Encoder-Decoder-Translator (PyTorch)

Dieses Projekt enthaelt einen bewusst einfachen Seq2Seq-Translator mit:

- GRU-Encoder
- GRU-Decoder
- Dot-Product-Attention
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

## Wichtige Parameter

- `--epochs` (default: `200`)
- `--batch-size` (default: `4`)
- `--emb-dim` (default: `64`)
- `--hidden-dim` (default: `64`)
- `--lr` (default: `1e-3`)
- `--teacher-forcing` (default: `0.7`)

## Hinweis

Der Datensatz ist absichtlich sehr klein (Toy-Parallelkorpus in der Datei), damit das Beispiel schnell verstaendlich und trainierbar bleibt.
