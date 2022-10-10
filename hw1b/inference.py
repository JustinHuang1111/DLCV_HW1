from model import EnsembledModel, FCN32
from dataset import InfDataset
import sys
import torch
import argparse
from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd
from torchvision.transforms import functional as TF


def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluation')
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='input data dir')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='output data dir')
    parser.add_argument('--model', '-m', nargs='+', type=str,
                        default=None, help='model path ( a list )')
    return parser.parse_args()


def inference(datapath, outpath, modelpath, batch_size=1):

    inference_set = InfDataset(datapath)
    inference_loader = DataLoader(
        inference_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # make output dir
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EnsembledModel(modelpath[0], modelpath[1], device)
    # model = FCN32().to(device)
    # model.load_state_dict(torch.load(modelpath[0]))
    model.eval()

    preds = []
    for imgs in inference_loader:
        logits = model(imgs.to(device))
        logits = torch.squeeze(logits)
        logits = TF.resize(logits, 512)
        pred = logits.permute(1, 2, 0).argmax(dim=-1)
        for _ in range(batch_size):
            output = torch.zeros((512, 512, 3))
            output[pred == 0] = torch.tensor(
                [0, 1, 1], dtype=torch.float32)
            output[pred == 1] = torch.tensor(
                [1, 1, 0], dtype=torch.float32)
            output[pred == 2] = torch.tensor(
                [1, 0, 1], dtype=torch.float32)
            output[pred == 3] = torch.tensor(
                [0, 1, 0], dtype=torch.float32)
            output[pred == 4] = torch.tensor(
                [0, 0, 1], dtype=torch.float32)
            output[pred == 5] = torch.tensor(
                [1, 1, 1], dtype=torch.float32)
            output[pred == 6] = torch.tensor(
                [0, 0, 0], dtype=torch.float32)
            pred_img = TF.to_pil_image(
                output.permute(2, 0, 1))
            preds.append(pred_img)

    for i, pred_img in enumerate(preds):
        output_path = os.path.join(
            args.output, "%s.png" % (inference_set.filenames[i][:-4]))
        pred_img.save(output_path)


if __name__ == "__main__":
    args = get_args()
    inference(args.input, args.output, args.model)
