import torch
import inspect
from pprint import pprint

from bento_service import PytorchService
from model import TextClassificationModel
from train import get_model_params, get_tokenizer_vocab

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


tokenizer, vocab = get_tokenizer_vocab()
vocab_size, emsize, num_class = get_model_params(vocab)
model = TextClassificationModel(vocab_size, emsize, num_class).to(device)
# pprint(inspect.getmembers(model, lambda a:not(inspect.isroutine(a))))
model.load_state_dict(torch.load("../model/pytorch/pytorch_model.pt"))
model.eval()

bento_svc = PytorchService()

bento_svc.pack("model", model)
bento_svc.pack("tokenizer", tokenizer)
bento_svc.pack("vocab", vocab)
saved_path = bento_svc.save()

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
