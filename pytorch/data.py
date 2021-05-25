from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab

import torch
from torchtext.datasets import AG_NEWS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")