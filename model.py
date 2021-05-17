import torch.nn as nn
import torch.distributed as distributed
from torch.nn.parallel.distributed import DistributedDataParallel
from transformers import BertPreTrainedModel, BertModel

class BertForSentimentClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
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
        # cls_reps = self.dropout(cls_reps)
        logits = self.cls_layer(cls_reps)
        return logits


def wrap_ddp(model: nn.Module, rank: int, world_size: int):
    distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    ddp_model = DistributedDataParallel(model, device_ids=[rank])
    return ddp_model

