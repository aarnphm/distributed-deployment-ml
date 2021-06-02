import os
from distutils.dir_util import copy_tree
from distutils.file_util import copy_file

import torch

from bento_service import PytorchService
from model import TextClassificationModel
from train import get_model_params, get_tokenizer_vocab

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trained_path = "../model/pytorch"
deploy_dir = "../bento_svc/torch_svc"
artifacts_dir = os.path.join(deploy_dir, "PytorchService")

if not os.path.exists(deploy_dir):
    os.makedirs(deploy_dir, exist_ok=True)

tokenizer, vocab = get_tokenizer_vocab()
vocab_size, emsize, num_class = get_model_params(vocab)
model = TextClassificationModel(vocab_size, emsize, num_class).to(device)
model.load_state_dict(torch.load("../model/pytorch/pytorch_model.pt"))
model.eval()

bento_svc = PytorchService()

bento_svc.pack("model", model)
bento_svc.pack("tokenizer", tokenizer)
bento_svc.pack("vocab", vocab)
saved_path = bento_svc.save()

copy_tree(saved_path, deploy_dir)
copy_file("Dockerfile", deploy_dir+"/Dockerfile")


if __name__ == '__main__':
    print("\nExample run")
    print(bento_svc.predict({
        'text': 'MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
        enduring the season’s worst weather conditions on Sunday at The \
        Open on his way to a closing 75 at Royal Portrush, which \
        considering the wind and the rain was a respectable showing. \
        Thursday’s first round at the WGC-FedEx St. Jude Invitational \
        was another story. With temperatures in the mid-80s and hardly any \
        wind, the Spaniard was 13 strokes better in a flawless round. \
        Thanks to his best putting performance on the PGA Tour, Rahm \
        finished with an 8-under 62 for a three-stroke lead, which \
        was even more impressive considering he’d never played the \
        front nine at TPC Southwind.'
    }))
    print("---")
    print(bento_svc.predict({
        'text': "Oil and Economy Cloud Stocks' Outlook (Reuters) Reuters - Soaring crude prices plus worries\\about "
                "the economy and the outlook for earnings are expected to\\hang over the stock market next week "
                "during the depth of the\\summer doldrums. "
    }))
    print("---")
    print("saved model path: %s" % saved_path)