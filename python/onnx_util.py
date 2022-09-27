import onnxruntime
import onnxruntime.quantization
import torch
import torchvision
from tqdm import tqdm

import torch_util


def test_onnx(
    session: onnxruntime.InferenceSession,
    dataset: torchvision.datasets.VisionDataset,
    batch_size: int = 1,
):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    top1 = 0
    top5 = 0

    with tqdm(dataloader, desc=f"[Test {len(dataset)} img]") as pbar:
        for i, (images, labels) in enumerate(pbar):
            preds = session.run([], {"input": images.numpy()})[0]
            acc1, acc5 = torch_util.accuracy(torch.from_numpy(preds), labels, topk=(1, 5))
            top1 += acc1[0]
            top5 += acc5[0]
            pbar.set_postfix(
                {
                    "Top1": top1 / (i + 1),
                    "Top5": top5 / (i + 1),
                }
            )


def calibration_method_from_str(str: str) -> onnxruntime.quantization.CalibrationMethod:
    try:
        return onnxruntime.quantization.CalibrationMethod[str]
    except KeyError:
        raise ValueError()
 