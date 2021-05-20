import os

import spacy
import torch
from distutils.dir_util import copy_tree

from bento_service_torch import PytorchService
from train_torch import model

deploy_dir = "deploy/pytorch_service"
artifacts_dir = os.path.join(deploy_dir, "PytorchService")
deploy_dataset_dir = os.path.join(artifacts_dir, "dataset", "imdb")

if not os.path.exists(deploy_dir):
    os.makedirs(deploy_dir, exist_ok=True)
if not os.path.exists(deploy_dataset_dir):
    os.makedirs(deploy_dataset_dir, exist_ok=True)

tokenizer = spacy.load('en_core_web_sm')

model.load_state_dict(torch.load("model/torchnet.pt"))
model.eval()

bento_svc = PytorchService()

bento_svc.pack("model", model)
bento_svc.pack("tokenizer", tokenizer)
saved_path = bento_svc.save()

print("\nExample run")
print(
    bento_svc.predict(
        {
            'text': "A wonderful little production.The filming technique is very unassuming- very "
                    "old-time-BBC fashion and gives a comforting, and sometimes discomforting, sense of realism to "
                    "the entire piece. The actors are extremely well chosen- Michael Sheen not only "
                    "has got all the polari but he has all the voices down pat too! You can truly see the seamless "
                    "editing guided by the references to Williams' diary entries, not only is it well worth the "
                    "watching but it is a terrificly written and performed piece. A masterful production about one of "
                    "the great master's of comedy and his life. The realism really comes home with the little things: "
                    "the fantasy of the guard which, rather than use the traditional 'dream' techniques remains solid "
                    "then disappears. It plays on our knowledge and our senses, particularly with the scenes "
                    "concerning Orton and Halliwell and the sets (particularly of their flat with Halliwell's murals "
                    "decorating every surface) are terribly well done."
        }
    )
)

print("_____")

print(
    bento_svc.predict(
        {
            'text': "This show was an amazing, fresh & innovative idea in the 70's when it first aired. The first 7 "
                    "or 8 years were brilliant, but things dropped off after that. By 1990, the show was not really "
                    "funny anymore, and it's continued its decline further to the complete waste of time it is "
                    "today.It's truly disgraceful how far this show has fallen. The writing is painfully "
                    "bad, the performances are almost as bad - if not for the mildly entertaining respite of the "
                    "guest-hosts, this show probably wouldn't still be on the air. I find it so hard to believe that "
                    "the same creator that hand-selected the original cast also chose the band of hacks that "
                    "followed. How can one recognize such brilliance and then see fit to replace it with such "
                    "mediocrity? I felt I must give 2 stars out of respect for the original cast that made this show "
                    "such a huge success. As it is now, the show is just awful. I can't believe it's still on the air. "
        }
    )
)

print("_____")
print("saved model path: %s" % saved_path)

print("\nCopy to deploy")
copy_tree(saved_path, deploy_dir)
copy_tree("dataset/imdb", deploy_dataset_dir)
copy_tree(".vector_cache", artifacts_dir+"/.vector_cache")
print("Finished.")