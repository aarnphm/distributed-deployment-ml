import os

import spacy
import torch

from torch.train import model

deploy_dir = "deploy/onnx_service"
artifacts_dir = os.path.join(deploy_dir, "OnnxService")
deploy_dataset_dir = os.path.join(artifacts_dir, "../dataset", "imdb")

if not os.path.exists(deploy_dir):
    os.makedirs(deploy_dir, exist_ok=True)
if not os.path.exists(deploy_dataset_dir):
    os.makedirs(deploy_dataset_dir, exist_ok=True)

tokenizer = spacy.load('en_core_web_sm')

model.load_state_dict(torch.load("model/torchnet.pt"))
model.eval()
print("\nModel summary:\n")
print(model)

with torch.no_grad():
    inp = torch.randn(1, 3, 25000)