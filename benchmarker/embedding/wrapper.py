from typing import Sequence

import torch
from torch import Tensor, nn

from benchmarker.embedding.base import ContextEmbeddings


class ContextEmbeddingWrapper(ContextEmbeddings):
    """Context embeddings wrapper.

    Args:
      context_embeddings: context embeddings
      do_layer_norm: apply layer normalization on context embedding
      proxy_initializer_range: define if and how to initialize the proxy_layer,
        If 0, no proxy_layer is used,
        if >0, parameter define standard deviation of initialized weights.
        Proxy layer is changing the dimension of context embedding to match model embedding size.
        Low value of initializer_range also ensure small contribution to
        final embeddings in early stage of training
      embeddings_size: size of lm model embedding

    """

    _FINAL_EMBEDDING_LEVEL = 'tokens'

    def __init__(
        self,
        context_embeddings: ContextEmbeddings,
        do_layer_norm: bool = False,
        proxy_initializer_range: float = 0.0,
        embeddings_size: int = 768,
    ):
        # embeddings of ContextEmbeddingWrapper are always tokens because other levels are expanded
        super().__init__(embeddings_size, ContextEmbeddingWrapper._FINAL_EMBEDDING_LEVEL)
        self.context_embeddings = context_embeddings
        self.proxy_initializer_range = proxy_initializer_range
        self.do_layer_norm = do_layer_norm
        self.embeddings_size = embeddings_size

        if self.do_layer_norm:
            self.layer_norm = nn.LayerNorm(self.context_embeddings.dimension, eps=1e-7)

        if self.proxy_initializer_range:
            assert (
                0.1 > proxy_initializer_range > 0.0
            ), 'initializer range have to be within range (0,0.1)'
            self.proxy_layer = nn.Linear(self.context_embeddings.dimension, self.embeddings_size)
            self.init_linear_weights(self.proxy_layer, self.proxy_initializer_range)

    def get_required_segment_levels(self) -> Sequence[str]:
        return self.context_embeddings.get_required_segment_levels()

    def forward(  # type: ignore
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        seg_data: dict,
        **kwargs
    ) -> Tensor:

        context_embeddings: Tensor = self.context_embeddings(
            input_ids=input_ids, attention_mask=attention_mask, seg_data=seg_data, **kwargs
        )
        if self.do_layer_norm:
            context_embeddings = self.layer_norm(context_embeddings)

        if self.proxy_initializer_range:
            context_embeddings = self.proxy_layer(context_embeddings)

        if self.context_embeddings.embedding_level == 'tokens':
            return context_embeddings
        else:
            # if level is different than 'tokens' do expanding to 'tokens'
            segment_token_map = \
                seg_data[self.context_embeddings.embedding_level]['token_map'].clone()
            bs, seg_len, features_dim = context_embeddings.shape
            context_embeddings = torch.cat(
                [context_embeddings, torch.zeros_like(context_embeddings[:, 0:1, :])], dim=1
            )

            with torch.no_grad():
                segment_token_map[segment_token_map == -1] = seg_len
                expand_index = (
                    segment_token_map[:, :, None].expand(-1, -1, features_dim).to(torch.long)
                )

            context_token_embeddings = torch.gather(context_embeddings, 1, expand_index)

            return context_token_embeddings
