from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
from dalle_pytorch import OpenAIDiscreteVAE
from dalle_pytorch.vae import map_pixels
from torch import Tensor

from benchmarker.embedding.base import ContextEmbeddings
from benchmarker.embedding.relative.relative import (
    RelativePositionBiasAggregated,
    RelativePositionBiasHorizontal,
    RelativePositionBiasVertical,
)


class DiscreteEmbeddings(ContextEmbeddings, nn.Module):
    def __init__(
        self,
        dimension: int = 768,
        embedding_level: str = 'tokens',
        num_layers: int = 5,
        pretrained_path: Optional[str] = None,
        num_resnet_blocks: int = 0,
        use_position_bias: bool = False,
        model_config: Optional[Any] = None,
        use_position_embeddings: Optional[bool] = None,
        **kwargs,
    ):
        nn.Module.__init__(self)
        ContextEmbeddings.__init__(self, dimension, embedding_level=embedding_level)

        self.vae = OpenAIDiscreteVAE()
        self.dalee = True
        self.vae.dec = None
        h, w = 32, 32

        for param in self.vae.parameters():
            param.requires_grad = False

        # if pretrained_path is not None:
        # self.vae.load_state_dict(torch.load(pretrained_path, map_location='cpu'))

        # h, w = self.vae.get_grid_size()
        max_length = h * w  # depends on number of layers and image size
        self.use_position_bias = use_position_bias
        self.use_position_embeddings = (
            not use_position_bias if use_position_embeddings is None else use_position_embeddings
        )

        self.semantic_embeddings = nn.Embedding(self.vae.num_tokens, dimension)
        self.register_buffer('position_ids', torch.arange(max_length).expand((1, -1)))

        if self.use_position_bias:
            heads = model_config.num_attention_heads
            self.position_bias = RelativePositionBiasAggregated(
                [RelativePositionBiasVertical(num_heads=heads), RelativePositionBiasHorizontal(num_heads=heads)]
            )

        if self.use_position_embeddings:
            self.position_embeddings = nn.Embedding(max_length, dimension)

    def get_required_segment_levels(self) -> Sequence[str]:
        return []

    def append_embeddings(self, inputs_embeds, context_embeddings, attention_mask):
        attention_mask = attention_mask.clone().detach()

        input_len = inputs_embeds.shape[1]
        embeddings_len = context_embeddings.shape[1]
        # replace padded tokens with context
        lens = attention_mask.sum(1)
        bs = attention_mask.shape[0]
        for idx in range(bs):
            l = lens[idx]
            space_left = input_len - l
            if space_left <= embeddings_len:
                attention_mask[idx] = 1
                inputs_embeds[idx, l:] = context_embeddings[idx, :space_left]
            else:
                attention_mask[idx, l : l + embeddings_len] = 1
                inputs_embeds[idx, l : l + embeddings_len] = context_embeddings[idx]

        return inputs_embeds, attention_mask

    def produce_bias(self, input_ids, attention_mask, seg_data, position_bias, old_attention_mask):
        if not self.use_position_bias:
            return position_bias

        # Calculate pseudo-positions
        h, w = 32, 32
        # h, w = self.vae.get_grid_size()
        x = (torch.arange(0.5, w + 0.5) / w).repeat(h, 1).permute(1, 0).flatten()

        if w > h:
            # Ugly coordinate fix for transposed matrices.
            # Left: Look at me, I am the right now
            x = 1 - x

        y = (torch.arange(0.5, h + 0.5) / h).repeat(w, 1).flatten()
        coordinates = torch.stack([x, y, x, y], 1).to(input_ids.device)

        max_length = input_ids.shape[1]
        left = (input_ids == 0).sum(1)
        start_idx = max_length - left

        bs = input_ids.shape[0]
        for idx in range(bs):
            start = start_idx[idx]
            end = min(max_length, start + h * w)
            seg_data['tokens']['bboxes'][idx, start:end] = coordinates[: min(left[idx], w * h)]

        new_position_bias = self.position_bias(input_ids, attention_mask, seg_data)

        """
        old_interactions = (old_attention_mask.unsqueeze(1).transpose(2, 1) * old_attention_mask.unsqueeze(1))
        all_interactions = (attention_mask.unsqueeze(1).transpose(2, 1) * attention_mask.unsqueeze(1))
        new_interactions = all_interactions * ~old_interactions.bool()

        heads = position_bias.shape[1]

        # Zero bias for newely-created positions
        nidx = new_interactions.unsqueeze(1).repeat(1, heads, 1, 1).bool()
        # Zero bias for old positions
        oidx = old_interactions.unsqueeze(1).repeat(1, heads, 1, 1).bool()
        # What about masked? Now, there is sum of both for them
        zidx = ~all_interactions.unsqueeze(1).repeat(1, heads, 1, 1).bool()

        position_bias[nidx] = 0
        new_position_bias[oidx] = 0
        """

        both = position_bias + new_position_bias

        # Do not uncomment. A present which seems valuable but which in reality is a curse
        # both[zidx] = -10000

        return both

    def forward(  # type: ignore
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        seg_data: dict,
        text_embeddings: Tensor,
        position_bias: Tensor,
        **kwargs,
    ) -> Tensor:

        image_batch = seg_data['lazyimages']['img_lst']

        if len(image_batch.shape) == 3:
            image_batch = image_batch.unsqueeze(1).repeat(1, 3, 1, 1)
        image_batch = map_pixels(image_batch)

        # image_batch = image_batch.unsqueeze(1)

        codewords = self.vae.get_codebook_indices(image_batch)
        semantic = self.semantic_embeddings(codewords)

        seq_length = codewords.shape[1]
        position_ids = self.position_ids[:, :seq_length]

        position = self.position_embeddings(position_ids) if self.use_position_embeddings else 0.0

        context_embeddings = semantic + position

        inputs_embeds_, attention_mask_ = self.append_embeddings(text_embeddings, context_embeddings, attention_mask)

        position_bias_ = self.produce_bias(input_ids, attention_mask_, seg_data, position_bias, attention_mask)

        return (inputs_embeds_, attention_mask_, position_bias_)


def create_discrete_embeddings(
    pretrained_path: Optional[str] = None, dimension=768, num_layers: int = 5, **kwargs
) -> DiscreteEmbeddings:
    return DiscreteEmbeddings(dimension=dimension, num_layers=num_layers, pretrained_path=pretrained_path, **kwargs)
