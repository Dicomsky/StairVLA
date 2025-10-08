"""
Framework factory utilities.
Selects and instantiates a registered framework implementation based on config.
"""



def build_framework(cfg):
    """
    Build a framework model from config.

    Args:
        cfg: Config object (OmegaConf / namespace) containing:
             cfg.framework.framework_py: Identifier string (e.g. "InternVLA-M1")

    Returns:
        nn.Module: Instantiated framework model.

    Raises:
        NotImplementedError: If the specified framework id is unsupported.
    """
    if cfg.framework.framework_py == "InternVLA-M1":
        from starVLA.model.framework.M1 import build_model_framework
        return build_model_framework(cfg)
    elif cfg.framework.framework_py == "QwenOFT":
        from starVLA.model.framework.QwenOFT import build_model_framework
        return build_model_framework(cfg)
    elif cfg.framework.framework_py == "QwenFast":
        from starVLA.model.framework.QwenFast import build_model_framework
        return build_model_framework(cfg)

    raise NotImplementedError(f"Framework {cfg.framework.framework_py} is not implemented.")

