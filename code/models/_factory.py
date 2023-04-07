from ._registry import model_entrypoint, split_model_name_tag


def create_model(
        model_name: str,
        pretrained: bool = False,
        **kwargs,
):
    """Create a model

    Lookup model's entrypoint function and pass relevant args to create a new model.

    **kwargs will be passed through entrypoint fn to timm.models.build_model_with_cfg()
    and then the model class __init__(). kwargs values set to None are pruned before passing.

    Args:
        model_name (str): name of model to instantiate
        pretrained (bool): load pretrained ImageNet-1k weights if true
        pretrained_cfg (Union[str, dict, PretrainedCfg]): pass in external pretrained_cfg for model
        pretrained_cfg_overlay (dict): replace key-values in base pretrained_cfg with these
        checkpoint_path (str): path of checkpoint to load _after_ the model is initialized
        scriptable (bool): set layer config so that model is jit scriptable (not working for all models yet)
        exportable (bool): set layer config so that model is traceable / ONNX exportable (not fully impl/obeyed yet)
        no_jit (bool): set layer config so that model doesn't utilize jit scripted layers (so far activations only)

    Keyword Args:
        drop_rate (float): dropout rate for training (default: 0.0)
        global_pool (str): global pool type (default: 'avg')
        **: other kwargs are consumed by builder or model __init__()
    """
    # Parameters that aren't supported by all models or are intended to only override model defaults if set
    # should default to None in command line args/cfg. Remove them if they are present and not set so that
    # non-supporting models don't break and default args remain in effect.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    model_name, pretrained_tag = split_model_name_tag(model_name)

    create_fn = model_entrypoint(model_name)

    model = create_fn(
        **kwargs,
    )   

    return model

