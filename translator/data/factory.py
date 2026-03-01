from typing import Callable, List, Protocol, runtime_checkable


@runtime_checkable
class TokenizerProtocol(Protocol):
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ...

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        ...

    @property
    def vocab_size(self) -> int:
        ...

    @property
    def pad_token_id(self) -> int | None:
        ...

    @property
    def bos_token_id(self) -> int | None:
        ...

    @property
    def eos_token_id(self) -> int | None:
        ...


TokenizerFactory = Callable[[List[str]], TokenizerProtocol]
TOKENIZER_CHOICES = ("custom", "hf")


def make_tokenizer_factory(tokenizer: str, hf_tokenizer_name: str) -> TokenizerFactory:
    from .tokenizer import HuggingFaceTokenizerAdapter, Tokenizer

    def make_custom_tokenizer_factory() -> TokenizerFactory:
        return lambda sentences: Tokenizer.build(sentences)

    def make_hf_tokenizer_factory() -> TokenizerFactory:
        return lambda _: HuggingFaceTokenizerAdapter.from_pretrained(hf_tokenizer_name)

    if tokenizer == "custom":
        return make_custom_tokenizer_factory()
    if tokenizer == "hf":
        return make_hf_tokenizer_factory()
    raise ValueError(f"Unknown tokenizer={tokenizer!r}. Allowed values: {TOKENIZER_CHOICES}.")
