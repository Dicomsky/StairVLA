"""
Framework factory utilities.
Automatically builds registered framework implementations
based on configuration.

Each framework module (e.g., M1.py, QwenFast.py) should register itself:
    from starVLA.model.framework.framework_registry import FRAMEWORK_REGISTRY

    @FRAMEWORK_REGISTRY.register("InternVLA-M1")
    def build_model_framework(config):
        return InternVLA_M1(config=config)
"""

import pkgutil
import importlib
from starVLA.model.tools import FRAMEWORK_REGISTRY


try:
    pkg_path = __path__
except NameError:
    pkg_path = None

# 自动导入所有 framework 子模块 import，触发注册
if pkg_path is not None:
    for _, module_name, _ in pkgutil.iter_modules(pkg_path):
        importlib.import_module(f"{__name__}.{module_name}")

def build_framework(cfg):
    """
    Build a framework model from config.
    Args:
        cfg: Config object (OmegaConf / namespace) containing:
             cfg.framework.framework_py: Identifier string (e.g. "InternVLA-M1")
    Returns:
        nn.Module: Instantiated framework model.
    """
    if cfg.framework.framework_py == "InternVLA-M1":
        from starVLA.model.framework.M1 import InternVLA_M1
        return InternVLA_M1(cfg)
    elif cfg.framework.framework_py == "QwenOFT":
        from starVLA.model.framework.QwenOFT import Qwenvl_OFT
        return Qwenvl_OFT(cfg)
    elif cfg.framework.framework_py == "QwenFast":
        from starVLA.model.framework.QwenFast import Qwenvl_Fast
        return Qwenvl_Fast(cfg)
    elif cfg.framework.framework_py == "QwenFM":
        from starVLA.model.framework.QwenFM import Qwenvl_FMHead
        return Qwenvl_FMHead(cfg)
    
    # auto detect from registry
    framework_id = cfg.framework.framework_py
    if framework_id not in FRAMEWORK_REGISTRY._registry:
        raise NotImplementedError(f"Framework {cfg.framework.framework_py} is not implemented.")
    
    MODLE_CLASS = FRAMEWORK_REGISTRY[framework_id]
    return MODLE_CLASS(cfg)

__all__ = ["build_framework", "FRAMEWORK_REGISTRY"]
