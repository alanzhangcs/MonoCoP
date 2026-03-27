from .monocop import build

def build_monocop(cfg):
    if cfg['type'] == 'monocop':
        return build(cfg)
    else:
        raise ValueError(f"Invalid model type: {cfg['type']}")