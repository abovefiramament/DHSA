"""Architecture bindings for official pyReFT; no intervention reimplementation."""


def register_pyreft_model(model):
    """Bind GPT-J's block output in pyvene releases without a GPT-J model card."""
    if getattr(model.config, "model_type", None) != "gptj":
        return
    from pyvene.models.constants import CONST_OUTPUT_HOOK
    from pyvene.models.intervenable_modelcard import (
        type_to_module_mapping, type_to_dimension_mapping,
    )

    # Both construction and saved-payload loading use the official model card.
    # Do not change mappings supplied by a newer pyvene release.
    type_to_module_mapping.setdefault(type(model), {}).setdefault(
        "block_output", ("transformer.h[%s]", CONST_OUTPUT_HOOK)
    )
    type_to_dimension_mapping.setdefault(type(model), {}).setdefault(
        "block_output", ("n_embd",)
    )
