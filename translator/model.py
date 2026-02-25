import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, pad_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, src_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        enc_out_packed, hidden = self.gru(packed)
        enc_out, _ = nn.utils.rnn.pad_packed_sequence(enc_out_packed, batch_first=True)
        return enc_out, hidden


class DotAttention(nn.Module):
    def forward(
        self, query: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, pad_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim + hidden_dim, hidden_dim, batch_first=True)
        self.attn = DotAttention()
        self.out = nn.Linear(hidden_dim * 2, vocab_size)

    def forward_step(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        enc_out: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(input_token).unsqueeze(1)
        query = hidden[-1]
        context, _ = self.attn(query, enc_out, src_mask)
        gru_input = torch.cat([emb, context.unsqueeze(1)], dim=-1)
        dec_out, hidden = self.gru(gru_input, hidden)
        logits = self.out(torch.cat([dec_out.squeeze(1), context], dim=-1))
        return logits, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, tgt_sos_idx: int, src_pad_idx: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_sos_idx = tgt_sos_idx
        self.src_pad_idx = src_pad_idx

    def forward(
        self, src: torch.Tensor, src_lens: torch.Tensor, tgt: torch.Tensor, teacher_forcing: float = 0.5
    ) -> torch.Tensor:
        bsz, tgt_len = tgt.shape
        vocab_size = self.decoder.out.out_features
        outputs = torch.zeros(bsz, tgt_len - 1, vocab_size, device=src.device)

        enc_out, hidden = self.encoder(src, src_lens)
        src_mask = (src != self.src_pad_idx).to(src.device)

        dec_input = tgt[:, 0]
        for t in range(1, tgt_len):
            logits, hidden = self.decoder.forward_step(dec_input, hidden, enc_out, src_mask)
            outputs[:, t - 1, :] = logits

            use_tf = random.random() < teacher_forcing
            next_input = tgt[:, t] if use_tf else logits.argmax(dim=-1)
            dec_input = next_input
        return outputs

    @torch.no_grad()
    def translate(
        self, src_ids: List[int], max_len: int, device: torch.device, eos_idx: int
    ) -> List[int]:
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_lens = torch.tensor([len(src_ids)], dtype=torch.long, device=device)
        enc_out, hidden = self.encoder(src, src_lens)
        src_mask = (src != self.src_pad_idx).to(device)

        dec_input = torch.tensor([self.tgt_sos_idx], dtype=torch.long, device=device)
        out_ids: List[int] = [self.tgt_sos_idx]
        for _ in range(max_len):
            logits, hidden = self.decoder.forward_step(dec_input, hidden, enc_out, src_mask)
            dec_input = logits.argmax(dim=-1)
            token = int(dec_input.item())
            out_ids.append(token)
            if token == eos_idx:
                break
        return out_ids
