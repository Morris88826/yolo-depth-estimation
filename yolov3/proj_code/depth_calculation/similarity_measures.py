"""
This file constrains different similarity measures used to compare blocks
between two images
"""
import torch


def ssd_similarity_measure(patch1: torch.Tensor, patch2: torch.Tensor) -> torch.Tensor:
  assert patch1.shape == patch2.shape

  ssd_value = torch.sum((patch1-patch2)*(patch1-patch2))

  return ssd_value


def sad_similarity_measure(patch1: torch.Tensor, patch2: torch.Tensor) -> torch.Tensor:

  assert patch1.shape == patch2.shape


  sad_value = torch.sum(torch.abs(patch1-patch2))

  return sad_value

