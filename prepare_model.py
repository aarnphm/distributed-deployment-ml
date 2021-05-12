import os
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoConfig

def get_bert_model(model_name="deepset/bert-base-cased-squad2", cache_dir="./cache"):
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name, cache_dir=cache_dir)
    return config, tokenizer, model

def main():
    model_name = "deepset/bert-case-cased-squad2"
    cache_dir="./cache"
    output_dir = "./saved_models"
    torch_model_name = "bert-base-cased-squad2_model.pt"
    torch_model_config_name = "bert-base-cased-squad2_config.pt"
    tokenizer_name = "bert-base-cased-squad2_tokenizer.pt"
    max_seq_length = 512

    config, tokenizer, model = get_bert_model(model_name=model_name, cache_dir=cache_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.save_pretrained(os.path.join(output_dir, torch_model_name))
    config.save_pretrained(os.path.join(output_dir, torch_model_config_name))
    tokenizer.save_pretrained(os.path.join(output_dir, tokenizer_name))

    inputs = {
        "input_ids": torch.zeros((1, max_seq_length),).long(),
        "attention_mask": torch.zeros((1, max_seq_length)).long(),
        "token_type_ids": torch.zeros((1, max_seq_length)).long(),
    }

    model.eval()

if __name__ == '__main__':
    main()
