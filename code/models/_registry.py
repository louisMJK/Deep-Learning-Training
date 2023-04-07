import sys
from typing import Any, Callable, Dict, Set
from collections import defaultdict


_module_to_models: Dict[str, Set[str]] = defaultdict(set)  # dict of sets to check membership of model in module
_model_to_module: Dict[str, str] = {}  # mapping of model names to module names
_model_entrypoints: Dict[str, Callable[..., Any]] = {}  # mapping of model names to architecture entrypoint fns


def register_model(fn: Callable[..., Any]) -> Callable[..., Any]:
    # lookup containing module
    mod = sys.modules[fn.__module__]
    module_name_split = fn.__module__.split('.')
    module_name = module_name_split[-1] if len(module_name_split) else ''

    # add model to __all__ in module
    model_name = fn.__name__
    if hasattr(mod, '__all__'):
        mod.__all__.append(model_name)
    else:
        mod.__all__ = [model_name]  # type: ignore

    # add entries to registry dict/sets
    _model_entrypoints[model_name] = fn
    _model_to_module[model_name] = module_name
    _module_to_models[module_name].add(model_name)

    return fn


def split_model_name_tag(model_name: str, no_tag: str = ''):
    model_name, *tag_list = model_name.split('.', 1)
    tag = tag_list[0] if tag_list else no_tag
    return model_name, tag


def get_arch_name(model_name: str) -> str:
    return split_model_name_tag(model_name)[0]


def model_entrypoint(model_name: str, module_filter=None) -> Callable[..., Any]:
    """Fetch a model entrypoint for specified model name
    """
    arch_name = get_arch_name(model_name)
    if module_filter and arch_name not in _module_to_models.get(module_filter, {}):
        raise RuntimeError(f'Model ({model_name} not found in module {module_filter}.')
    return _model_entrypoints[arch_name]

