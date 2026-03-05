# Lazy imports to avoid loading heavy dependencies (e.g. open_clip) on module import
def __getattr__(name):
    if name == 'CLIPEncoder':
        from .encoder import CLIPEncoder
        return CLIPEncoder
    elif name == 'Reconstructor':
        from .reconstructor import Reconstructor
        return Reconstructor
    elif name == 'OcclusionSelector':
        from .selector import OcclusionSelector
        return OcclusionSelector
    elif name == 'MaskedPGD':
        from .perturber import MaskedPGD
        return MaskedPGD
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
