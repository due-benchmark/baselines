from abc import ABC, abstractmethod
from typing import Sequence

from torch import Tensor, nn


class ContextEmbeddings(nn.Module, ABC):
    """Base class for context embeddings.

    Args:
      dimension: dimension of embedding vectors
      embedding_level: indicate level for which contex embedding will be computed,
        levels different that 'tokens' would need to be expanded before adding to token
        embeddings
    """

    def __init__(self, dimension: int, embedding_level: str):
        super().__init__()
        self.dimension = dimension
        self.embedding_level = embedding_level

    @staticmethod
    def init_linear_weights(module, init_range):
        module.weight.data.normal_(mean=0.0, std=init_range)
        if module.bias is not None:
            module.bias.data.zero_()

    @abstractmethod
    def get_required_segment_levels(self) -> Sequence[str]:
        """List segment levels required by the embedding.

        Each child class need to implement method which tells which segment
        levels need to be delivered by dataloader

        Returns:
          list of segment names

        """

    @abstractmethod
    def forward(self, input_ids: Tensor, attention_mask: Tensor,  # type: ignore
                seg_data: dict, **kwargs) -> Tensor:
        """Compute the additional context embedding.

        Its dimension has to equal the dimension of the original BERT
        embeddings. It has access to the attention mask (of the same shape as
        `input_ids`, taking values 1.0 for non-masked and 0.0 for masked
        tokens).

        Implementations are responsible for verifying whether correct extra inputs were passed in kwargs.

        Args:
          input_ids: input token indices
          attention_mask: bool mask indicating whether threre is a token at given position
          seg_data: dict of class specific parameters used to compute context embedding
          kwargs: additional parameters required by the specific implementation

        Returns:
          context embeddings of tokens or segments

        """
