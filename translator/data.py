import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import torch
from torch.utils.data import Dataset

from .constants import EOS, PAD, SOS, UNK


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

@dataclass
class Tokenizer:
    stoi: Dict[str, int]
    itos: List[str]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.lower().strip().split()

    @classmethod
    def build(cls, sentences: List[str]) -> "Tokenizer":
        tokens = [PAD, SOS, EOS, UNK]
        seen = set(tokens)
        for sent in sentences:
            for tok in cls._tokenize(sent):
                if tok not in seen:
                    seen.add(tok)
                    tokens.append(tok)
        stoi = {tok: i for i, tok in enumerate(tokens)}
        return cls(stoi=stoi, itos=tokens)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids: List[int] = []
        if add_special_tokens:
            ids.append(self.sos_token_id)
        ids.extend(self.stoi.get(tok, self.unk_token_id) for tok in self._tokenize(text))
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        words: List[str] = []
        for idx in ids:
            tok = self.itos[idx]
            if skip_special_tokens and tok in {SOS, PAD}:
                continue
            if tok == EOS and skip_special_tokens:
                break
            words.append(tok)
        return " ".join(words)

    def __call__(
        self,
        texts: Union[str, List[str]],
        padding: Union[bool, str] = False,
        truncation: bool = False,
        max_length: int = None,
        return_tensors: str = None,
        add_special_tokens: bool = True,
    ) -> Dict[str, Union[List[List[int]], torch.Tensor]]:
        if isinstance(texts, str):
            texts = [texts]

        input_ids = [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]

        if truncation and max_length is not None:
            input_ids = [ids[:max_length] for ids in input_ids]

        attention_mask = [[1] * len(ids) for ids in input_ids]

        needs_padding = padding is True or padding == "longest" or padding == "max_length"
        if needs_padding:
            if max_length is not None and padding == "max_length":
                target_len = max_length
            else:
                target_len = max(len(ids) for ids in input_ids)
            input_ids = [ids + [self.pad_token_id] * (target_len - len(ids)) for ids in input_ids]
            attention_mask = [mask + [0] * (target_len - len(mask)) for mask in attention_mask]

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def size(self) -> int:
        return self.vocab_size

    @property
    def pad_token_id(self) -> int:
        return self.stoi[PAD]

    @property
    def sos_token_id(self) -> int:
        return self.stoi[SOS]

    @property
    def eos_token_id(self) -> int:
        return self.stoi[EOS]

    @property
    def unk_token_id(self) -> int:
        return self.stoi[UNK]


class TranslationDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], src_tokenizer: Tokenizer, tgt_tokenizer: Tokenizer):
        self.data = [(src_tokenizer.encode(src), tgt_tokenizer.encode(tgt)) for src, tgt in pairs]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.data[idx]


def collate_fn(
    batch: List[Tuple[List[int], List[int]]], pad_idx_src: int, pad_idx_tgt: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    src_lens = torch.tensor([len(src) for src, _ in batch], dtype=torch.long)
    tgt_lens = torch.tensor([len(tgt) for _, tgt in batch], dtype=torch.long)

    max_src = int(src_lens.max().item())
    max_tgt = int(tgt_lens.max().item())
    bsz = len(batch)

    src_batch = torch.full((bsz, max_src), pad_idx_src, dtype=torch.long)
    tgt_batch = torch.full((bsz, max_tgt), pad_idx_tgt, dtype=torch.long)

    for i, (src, tgt) in enumerate(batch):
        src_batch[i, : len(src)] = torch.tensor(src, dtype=torch.long)
        tgt_batch[i, : len(tgt)] = torch.tensor(tgt, dtype=torch.long)
    return src_batch, src_lens, tgt_batch, tgt_lens


def tiny_parallel_corpus() -> List[Tuple[str, str]]:
    return [
        ("ich bin muede", "i am tired"),
        ("ich bin hungrig", "i am hungry"),
        ("ich liebe kaffee", "i love coffee"),
        ("ich trinke wasser", "i drink water"),
        ("guten morgen", "good morning"),
        ("gute nacht", "good night"),
        ("wie geht es dir", "how are you"),
        ("mir geht es gut", "i am fine"),
        ("danke", "thank you"),
        ("bis spaeter", "see you later"),
        ("wo ist der bahnhof", "where is the train station"),
        ("ich lerne deutsch", "i am learning german"),
    ]
