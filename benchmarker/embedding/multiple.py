from typing import Sequence

from torch import Tensor, nn

from benchmarker.embedding.base import ContextEmbeddings
from benchmarker.embedding.wrapper import ContextEmbeddingWrapper

import torch


class MultipleContextEmbeddings(ContextEmbeddings):
    """Container for multiple context embeddings.

    Args:
      context_embeddings: sequence of context embeddings
    """

    def __init__(self, context_embeddings: Sequence[ContextEmbeddings]) -> None:
        if len(context_embeddings) == 0:
            raise ValueError("Collection of context embeddings cannot be empty")
        dimensions = [ce.dimension for ce in context_embeddings]
        if any(map(lambda x: dimensions[0] != x, dimensions)):
            raise ValueError("Context embeddings have different dimensions")
        embedding_levels = [ce.embedding_level for ce in context_embeddings]
        if any(map(lambda x: embedding_levels[0] != x, embedding_levels)):
            raise ValueError("Context embeddings have different embedding levels")

        super().__init__(
            dimension=context_embeddings[0].dimension,
            embedding_level=context_embeddings[0].embedding_level,
        )
        self.context_embeddings = nn.ModuleList(context_embeddings)
        self.has_pre_encoder = self._has_pre_encoder()
        self.has_post_encoder = self._has_post_encoder()

    def __len__(self) -> int:
        return len(self.context_embeddings)  # type: ignore  # ModuleList annotated as Iterable, missing Sized

    def __getitem(self, i: int) -> ContextEmbeddings:
        return self.context_embeddings[i]  # type: ignore  # ModuleList annotated as Iterable, should be Sequence

    def get_required_segment_levels(self) -> Sequence[str]:
        return list(
            {
                seg_lvl
                for ce in self.context_embeddings  # type: ignore  # no __next__
                for seg_lvl in ce.get_required_segment_levels()
            }
        )

    def forward(  # type: ignore
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        seg_data: dict,
        pre_encoder: bool = True,
        **kwargs
    ) -> Tensor:
        desired_context_emb = [ce for ce in self.context_embeddings if self.are_pre_encoder(ce) == pre_encoder]

        if len(desired_context_emb) == 1:
            ce = desired_context_emb[0]
            return ce(input_ids=input_ids, attention_mask=attention_mask, seg_data=seg_data, **kwargs)

        return sum(  # type: ignore
            ce(input_ids=input_ids, attention_mask=attention_mask, seg_data=seg_data, **kwargs)
            for ce in desired_context_emb  # type: ignore  # no __next
        )

    def _has_pre_encoder(self) -> bool:
        for ce in self.context_embeddings:
            if self.are_pre_encoder(ce):
                return True
        return False

    def _has_post_encoder(self) -> bool:
        for ce in self.context_embeddings:
            if not self.are_pre_encoder(ce):
                return True
        return False

    @staticmethod
    def are_pre_encoder(ce) -> bool:
        if isinstance(ce, ContextEmbeddingWrapper):
            ce = ce.context_embeddings
        if not hasattr(ce, 'pre_encoder'):
            return True
        return ce.pre_encoder
