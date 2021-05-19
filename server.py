import bentoml
from args import model_name_or_path
from model import TransformersBert

BERT = TransformersBert.from_pretrained(model_name_or_path)
