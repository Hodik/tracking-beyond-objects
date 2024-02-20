import os
import csv
import time
from functools import cached_property

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator, Vocab
from torch.optim import Adam

from PIL import Image

from .cnn_encoder import ImageEncoder
from .transformer import Transformer


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
    return " ".join(vocab.forward(ids))


class CaptionsDataset(Dataset):
    start_token = "<sos>"
    end_token = "<eos>"

    def __init__(self, images_dir, csv_file, transform=None):
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
            reader = iter(csv.reader(f, delimiter="|"))
            next(reader)
            for row in reader:
                if len(row) != 3:
                    print("skipping", row)
                    continue
                self.images.append(row[0].strip())
                self.captions.append(
                    f"{self.start_token} {row[2].strip()} {self.end_token}"
                )
        self.vocab = build_vocab(self.captions)
        self.transform = transform

    @property
    def vocab_size(self):
        return len(self.vocab)

    @cached_property
    def max_len(self):
        return max(len(caption.split()) for caption in self.captions)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name).convert("RGB")
        num_padding = self.max_len - len(self.captions[idx].split())

        # append padding tokens to the end of the caption
        caption = torch.tensor(
            tokenize(self.vocab, self.captions[idx])
            + [self.vocab["<pad>"]] * num_padding,
            dtype=torch.long,
        )

        if self.transform:
            image = self.transform(image)
        return image, caption


# Define the path to your dataset directory and CSV file
images_dir = "flickr30k_images/flickr30k_images"
csv_file = "flickr30k_images/results.csv"

# Define transformations for your images
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize images to (224, 224)
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize images
    ]
)

# Create a custom dataset
dataset = CaptionsDataset(images_dir=images_dir, csv_file=csv_file, transform=transform)

device = "cpu"
max_steps = 1000
batch_size = 32
shuffle = True
num_workers = 4
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = Transformer(
    vocab_size=dataset.vocab_size,
    d_model=512,
    img_encode_size=196,
    enc_n_layers=2,
    dec_n_layers=8,
    enc_n_heads=8,
    dec_n_heads=8,
    dropout=0.1,
    max_len=dataset.max_len,
    pad_id=dataset.vocab["<pad>"],
)
image_encoder = ImageEncoder()

print(f"model #params: {sum(p.numel() for p in model.parameters())}")

optimizer = Adam(model.parameters(), lr=5e-4)

for step, (images, captions) in enumerate(dataloader):
    t0 = time.time()
    images_encoded = image_encoder(images)
    X, Y = images_encoded, captions
    logits, loss = model(X, captions[:, :-1], captions[:, 1:])
    model.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # wait for all CUDA work on the GPU to finish then calculate iteration time taken
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.time()

    # logging
    if step % 10 == 0:
        print(f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms")

    if step % 100 == 0:
        torch.save(model.state_dict(), "captions.pt")

    # # evaluate the model
    # if step > 0 and step % 100 == 0:
    #     train_loss = evaluate(model, train_dataset, batch_size=100, max_batches=10)
    #     test_loss = evaluate(model, test_dataset, batch_size=100, max_batches=10)
    #     writer.add_scalar("Loss/train", train_loss, step)
    #     writer.add_scalar("Loss/test", test_loss, step)
    #     writer.flush()
    #     print(f"step {step} train loss: {train_loss} test loss: {test_loss}")
    #     # save the model to disk if it has improved
    #     if best_loss is None or test_loss < best_loss:
    #         out_path = os.path.join(args.work_dir, "model.pt")
    #         print(
    #             f"test loss {test_loss} is the best so far, saving model to {out_path}"
    #         )
    #         torch.save(model.state_dict(), out_path)
    #         best_loss = test_loss

    # termination conditions

    if max_steps >= 0 and step >= max_steps:
        break
