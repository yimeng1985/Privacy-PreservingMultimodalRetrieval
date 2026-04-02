# Lazy imports to avoid loading heavy dependencies (e.g. open_clip) on module import
def __getattr__(name):
    if name == 'CLIPEncoder':
        from .encoder import CLIPEncoder
        return CLIPEncoder
    elif name == 'Reconstructor':
        from .reconstructor import Reconstructor
        return Reconstructor
    elif name == 'ImprovedReconstructor':
        from .reconstructor import ImprovedReconstructor
        return ImprovedReconstructor
    elif name == 'OcclusionAnalyzer':
        from .selector import OcclusionAnalyzer
        return OcclusionAnalyzer
    elif name == 'VLMPrivacyJudge':
        from .vlm import VLMPrivacyJudge
        return VLMPrivacyJudge
    elif name == 'MockVLMJudge':
        from .vlm import MockVLMJudge
        return MockVLMJudge
    elif name == 'QwenVLMJudge':
        from .vlm import QwenVLMJudge
        return QwenVLMJudge
    elif name == 'ScoreFusion':
        from .fusion import ScoreFusion
        return ScoreFusion
    elif name == 'AdaptiveProtector':
        from .protector import AdaptiveProtector
        return AdaptiveProtector
    # Legacy support
    elif name == 'OcclusionSelector':
        from .selector import OcclusionAnalyzer as OcclusionSelector
        return OcclusionSelector
    elif name == 'MaskedPGD':
        from .perturber import MaskedPGD
        return MaskedPGD
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
