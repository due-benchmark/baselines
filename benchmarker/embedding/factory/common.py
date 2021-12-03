from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping, Optional

from benchmarker.embedding.base import ContextEmbeddings
from benchmarker.embedding.wrapper import ContextEmbeddingWrapper


class EmbeddingsMetaFactory(ABC):
    @property
    @abstractmethod
    def registry(self) -> Mapping[str, Callable[..., ContextEmbeddings]]:
        """Mapping of context embedding name to associated factory functions."""

    def build(
        self,
        chosen_embeddings_type: str,
        embeddings_size: int,
        out_size: Optional[int] = None,
        proxy_initializer_range: float = 0.0,
        do_layer_norm: bool = False,
        **kwargs: Optional[Any],
    ) -> ContextEmbeddings:
        """Build embeddings.

        Args:
          chosen_embeddings_type: type of embeddings to initialize
          embeddings_size: size of language model embedding
          kwargs: dictionary of additional parameters used for context embedding initialization

        Returns:
          instance of ContextEmbeddings

        """
        if chosen_embeddings_type in self.registry:
            ce = self.registry[chosen_embeddings_type](
                context_out_size=out_size or embeddings_size,
                out_size=out_size,
                proxy_initializer_range=proxy_initializer_range,
                do_layer_norm=do_layer_norm,
                **kwargs,
            )  # type: ignore
        else:
            raise NotImplementedError(
                f"There is no embedding type implemented for given string {chosen_embeddings_type}"
            )

        if not (ce.dimension == embeddings_size or proxy_initializer_range):
            raise RuntimeError(
                "Context embedding size have to be equal to"
                " lm embedding size if no proxy layer is used"
            )

        # use wrapper only if it's needed
        embedding_level = getattr(ce, "embedding_level", "tokens")
        if embedding_level != "tokens" or proxy_initializer_range or do_layer_norm:
            return ContextEmbeddingWrapper(
                context_embeddings=ce,
                proxy_initializer_range=proxy_initializer_range,
                do_layer_norm=do_layer_norm,
                embeddings_size=embeddings_size,
            )
        else:
            return ce
