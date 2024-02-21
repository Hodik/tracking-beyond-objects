import os
import re
import csv
import json

import torch
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator, Vocab

from PIL import Image
from . import config


def yield_tokens(captions):
    for caption in captions:
        yield caption.strip().split()[1:-1]


def build_vocab(captions) -> Vocab:
    vocab = build_vocab_from_iterator(
        yield_tokens(captions),
        min_freq=1,
        specials=("<sos>", "<eos>", "<unk>", "<pad>"),
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab


def tokenize(vocab, text):
    return vocab.forward(text.split())


def detokenize(vocab, ids):
    return " ".join(vocab.lookup_tokens(ids))


class CaptionsDataset(Dataset):
    start_token = "<sos>"
    end_token = "<eos>"


    @property
    def vocab_size(self):
        return len(self.vocab)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name).convert("RGB")
        caption = self.captions[idx].split()[:config.max_len]
        num_padding = config.max_len - len(caption)

        # append padding tokens to the end of the caption
        caption = torch.tensor(
            tokenize(self.vocab, " ".join(caption))
            + [self.vocab["<pad>"]] * num_padding,
            dtype=torch.long,
        )

        if self.transform:
            image = self.transform(image)
        return image, caption


class FlickrDataset(CaptionsDataset):
    
    def __init__(self, images_dir, csv_file, transform=None, mode="train"):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = images_dir
        self.images = []
        self.captions = []
        with open(csv_file, "r") as f:
            reader = list(csv.reader(f, delimiter="|"))
            num_samples = len(reader) - 2
            for i, row in enumerate(reader[1:]):
                if len(row) != 3:
                    print("skipping", row)
                    continue

                if mode == "train" and float(i) / float(num_samples) > config.train_dataset_ratio:
                    break
                
                if mode == "val":
                    if i < float(num_samples) * config.train_dataset_ratio:
                        continue

                self.images.append(row[0].strip())
                self.captions.append(
                    f"{self.start_token} {row[2].strip()} {self.end_token}"
                )
        self.vocab = build_vocab(self.captions)
        self.transform = transform


class COCODataset(CaptionsDataset):
    
    def __init__(self, images_dir, captions_file, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = images_dir
        self.images = []
        self.captions = []
        with open(captions_file, "r") as f:
            data = json.load(f)
        for annotation in data['annotations']:
            self.images.append('%012d.jpg' % annotation['image_id'])
            caption = annotation['caption'].lower()
            caption = re.sub(r'[^\w\s]', '', caption)
            caption = re.sub('\s+', ' ', caption)
            caption = caption.strip()
            self.captions.append(f"{self.start_token} {caption} {self.end_token}")
        self.vocab = build_vocab(self.captions)
        self.transform = transform