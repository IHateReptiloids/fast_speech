import torch


def seed_all(seed=3407):
    g = torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return g
