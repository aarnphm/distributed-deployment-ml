import os
from distutils.dir_util import copy_tree
# from distutils.file_util import copy_file

import torch
# import onnx

from bento_service import OnnxService
from helpers import TextClassificationModel, get_model_params, get_tokenizer_vocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

deploy_dir = "../bentoml_service/onnx_svc"
artifacts_dir = os.path.join(deploy_dir, "OnnxService")
onnx_model_path = "../model/onnx/pytorch_model.onnx"


# def tolong(tensor):
#     return tensor.detach().long().to(device) if tensor.requires_grad else tensor.long().to(device)


if not os.path.exists(deploy_dir):
    os.makedirs(deploy_dir, exist_ok=True)

tokenizer, vocab = get_tokenizer_vocab()
vocab_size, embedding_size, num_class = get_model_params(vocab)
model = TextClassificationModel(vocab_size, embedding_size, num_class).to(device)
model.load_state_dict(torch.load("../model/pytorch/pytorch_model.pt"))
model.eval()

print("\nExporting torch model to onnx...")
# inp = tolong(torch.rand(vocab_size, requires_grad=True).to(device))  # turn our input into a cuda tensor.
# inp = torch.rand(1,batch_size, 224,244).long().to(device)
inp = torch.rand(vocab_size).long().to(device)


torch.onnx.export(model, inp, onnx_model_path, export_params=True, opset_version=11, do_constant_folding=True,
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "vocab_size"}, "output": {0: "vocab_size"}})

# print("\n Loading model to check...")
# onnx_model = onnx.load(onnx_model_path)
# onnx.checker.check_model(onnx_model)

bento_svc = OnnxService()
bento_svc.pack("model", onnx_model_path)
bento_svc.pack("tokenizer", tokenizer)
bento_svc.pack("vocab", vocab)
saved_path = bento_svc.save()

copy_tree(saved_path, deploy_dir)
# copy_file("Dockerfile", deploy_dir + "/Dockerfile")

if __name__ == "__main__":
    print("\nExample run")
    print(bento_svc.predict({
        "text": "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
        enduring the season’s worst weather conditions on Sunday at The \
        Open on his way to a closing 75 at Royal Portrush, which \
        considering the wind and the rain was a respectable showing. \
        Thursday’s first round at the WGC-FedEx St. Jude Invitational \
        was another story. With temperatures in the mid-80s and hardly any \
        wind, the Spaniard was 13 strokes better in a flawless round. \
        Thanks to his best putting performance on the PGA Tour, Rahm \
        finished with an 8-under 62 for a three-stroke lead, which \
        was even more impressive considering he’d never played the \
        front nine at TPC Southwind."
    }))
    print("---")
    print(bento_svc.predict({
        "text": "Oil and Economy Cloud Stocks' Outlook (Reuters) Reuters - Soaring crude prices plus worries\\about "
                "the economy and the outlook for earnings are expected to\\hang over the stock market next week "
                "during the depth of the\\summer doldrums. "
    }))
    print("---")
    print("saved model path: %s" % saved_path)