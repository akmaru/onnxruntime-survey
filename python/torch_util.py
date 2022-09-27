from typing import Tuple, Union

import torch
import torchvision
from tqdm import tqdm


def subset_dataset(
    dataset: torchvision.datasets.VisionDataset, num_samples: int, seed: int = 0
) -> torchvision.datasets.VisionDataset:
    subset, _ = torch.utils.data.random_split(
        dataset,
        lengths=[num_samples, len(dataset) - num_samples],
        generator=torch.Generator().manual_seed(seed),
    )
    return subset


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int] = (1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def test_torch(
    model: Union[torch.nn.Module, torch.jit.ScriptModule],
    dataset: torchvision.datasets.VisionDataset,
    batch_size: int = 32,
):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    top1 = 0
    top5 = 0

    with tqdm(dataloader, desc=f"[Test {len(dataset)} img]") as pbar:
        for i, (images, labels) in enumerate(pbar):
            preds = model(images)
            acc1, acc5 = accuracy(preds, labels, topk=(1, 5))
            top1 += acc1[0]
            top5 += acc5[0]
            pbar.set_postfix(
                {
                    "Top1": top1 / (i + 1),
                    "Top5": top5 / (i + 1),
                }
            )
