# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from fairseq.data.encoders import register_tokenizer
from fairseq.dataclass import FairseqDataclass


@dataclass
class ZEJiebaTokenizerConfig(FairseqDataclass):
    source_lang: str = field(default="zh", metadata={"help": "source language"})
    target_lang: str = field(default="en", metadata={"help": "target language"})


@register_tokenizer("zejieba", dataclass=ZEJiebaTokenizerConfig)
class ZEJiebaTokenizer(object):
    def __init__(self, cfg: ZEJiebaTokenizerConfig):
        self.cfg = cfg

        try:
            import jieba
            from sacremoses import MosesDetokenizer

            self.tok = jieba
            self.detok = MosesDetokenizer(cfg.target_lang)
        except ImportError:
            raise ImportError(
                "Please install Moses and Jieba tokenizers with: pip install sacremoses and pip install jieba."
            )

    def encode(self, x: str) -> str:
        return ' '.join(self.tok.cut(x))

    def decode(self, x: str) -> str:
        return self.detok.detokenize(x.split())
