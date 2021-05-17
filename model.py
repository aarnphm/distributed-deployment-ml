from typing import List

import torch
import torch.nn as nn
import torch.distributed as distributed
from torch.nn.parallel.distributed import DistributedDataParallel
from transformers import BertPreTrainedModel, BertModel, AutoTokenizer

from args import model_name_or_path


def wrap_ddp(model: nn.Module, rank: int, world_size: int):
    distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    ddp_model = DistributedDataParallel(model, device_ids=[rank])
    return ddp_model


class BertForSentimentClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # The classification layer that takes the [CLS] representation and outputs the logit
        self.cls_layer = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        """
        :param input_ids: Tensor of shape [B, T] containing token ids of sequences
        :param attention_mask: Tensor of shape [B, T] containing attention masks to be used to avoid contribution of
               PAD tokens.
        :return: logits
        """
        # Feed the input to Bert model to obtain outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Obtain the representations of [CLS] heads
        cls_reps = outputs.last_hidden_state[:, 0]
        cls_reps = self.dropout(cls_reps)
        logits = self.cls_layer(cls_reps)
        return logits


class WrappedBertForSentimentClassification:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.custom_bert = BertForSentimentClassification.from_pretrained(model_name_or_path)
        self.custom_bert.eval()
        self.custom_bert.to("cuda")

    def predict(self, batch: List[str]) -> List[str]:
        batch_inputs = []

        for text in batch:
            tokens = self.tokenizer.tokenize(text)
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            seq = torch.tensor(tokens_ids)
            seq = seq.unsqueeze(0)
            attn_mask = (seq != 0).long()
        with torch.no_grad():
            logit = self.custom_bert(seq, attn_mask)
            prob = torch.sigmoid(logit.unsqueeze(-1))
            prob = prob.item()
            soft_prob = prob > 0.5
            if soft_prob == 1:
                return 'positive', int(prob * 100)
            else:
                return 'negative', int(100 - prob * 100)

 