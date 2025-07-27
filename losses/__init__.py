from .simclr import simclr_loss as simCLR


CONTRASTIVE_LOSS = {
    'simclr': simCLR,
    
}