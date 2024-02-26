import cv2
import torch
from torch.functional import F
import numpy as np
from .datasets import detokenize
from PIL import Image

from . import config



@torch.no_grad
def generate(dataset, model: torch.nn.Module, image_encoder: torch.nn.Module, transform):
    model.eval()
    image_encoder.eval()
    
    image = Image.open("coco/coco2017/val2017" + "/000000001296.jpg").convert("RGB")
    image = transform(image).unsqueeze(0).to(config.device)
    idx = torch.tensor(dataset.vocab['<sos>']).unsqueeze(0).unsqueeze(0).to(config.device)


    while True:
        images_encoded = image_encoder(image)
        logits = model(images_encoded, idx)
        probs  = F.softmax(logits[:, -1, :], dim=-1)
        _, idx_next = torch.topk(probs, k=1, dim=-1) 
        idx = torch.cat((idx, idx_next), dim=1)
        
        print("Sentence so far:", detokenize(dataset.vocab, idx[0].numpy(force=True)))

        if idx_next == dataset.vocab['<eos>']:
            print("end of sentece")
            break

        if idx.numel() == config.max_len:
            break
        
    model.train()
    image_encoder.train()


if __name__ == "__main__":
    from .train import dataset, model, image_encoder, transform

    model.load_state_dict(torch.load("captions.pt"))
    image_encoder.load_state_dict(torch.load("image_encoder.pt"))
    generate(dataset, model, image_encoder, transform)