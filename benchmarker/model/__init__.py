from transformers import (
    T5Config,
    T5TokenizerFast,
)


from benchmarker.config.benchmarker_config import (
    T5BenchmarkerConfig,
)
#from benchmarker.data.t5 import TrainingT5DataConverter

from benchmarker.model.t5 import T52dForConditionalGeneration

MODEL_CLASSES = {

    "t5": {
        "config": T5Config,
        "config2d": T5BenchmarkerConfig,
        "tokenizer": T5TokenizerFast,
        "model_attr_name": "t5",
        #"training_data_converter": TrainingT5DataConverter,
        "wordgap_data_converter": None,
        "pretraining": T52dForConditionalGeneration,
        "token_classification": T52dForConditionalGeneration,
    },
}

MODEL_CLASSES_REVERSE = {
    model_class: (model_type, model_kind)
    for model_type, model_dict in MODEL_CLASSES.items()
    for model_kind, model_class in model_dict.items()
    if isinstance(model_class, type)
}
