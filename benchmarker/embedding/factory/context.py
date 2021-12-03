import warnings
from typing import Optional

from benchmarker.config.benchmarker_config import BaseBenchmarkerConfig
from benchmarker.embedding.base import ContextEmbeddings
from benchmarker.embedding.factory.common import EmbeddingsMetaFactory
from benchmarker.embedding.multiple import MultipleContextEmbeddings
from benchmarker.embedding.image.discrete_vae.discrete_embeddings import create_discrete_embeddings


class ContextEmbeddingsFactory(EmbeddingsMetaFactory):
    @property
    def registry(self):
        return {
            "discrete_vae": create_discrete_embeddings,
        }

    def build_multiple(self, config: BaseBenchmarkerConfig) -> MultipleContextEmbeddings:
        """Create multiple context embeddings.

        Args:
          config: model's configuration

        Returns
          sequence with created embeddings

        """
        return MultipleContextEmbeddings(
            [
                self.build(
                    chosen_embeddings_type=context_embeddings["embedding_type"],
                    embeddings_size=config.hidden_size,
                    model_config=config,
                    **context_embeddings,
                )
                for context_embeddings in config.context_embeddings
            ]
        )

    def build_conditionally(
        self,
        config: BaseBenchmarkerConfig,
    ) -> Optional[ContextEmbeddings]:
        """Build if needed.

        Args:
          config: type of embeddings to initialize

        Returns:
          instance of ContextEmbeddings or None

        """
        if len(config.context_embeddings) > 0:
            return self.build_multiple(config)
        else:
            warnings.warn(
                "Config does not contain parameter which define type of context embeddings."
                "Layout information will not be used by the model."
            )
        return None
