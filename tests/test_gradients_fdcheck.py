import torch

from ntops.inverse.gradients import cosine_similarity


def test_cosine_similarity_bounds():
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([1.0, 0.0])
    c = torch.tensor([0.0, 1.0])
    assert cosine_similarity(a, b) > 0.99
    assert cosine_similarity(a, c) < 0.1
