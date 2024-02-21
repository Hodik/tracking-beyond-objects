import time

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.functional import F

from .cnn_encoder import ImageEncoder
from .transformer import Transformer

from .datasets import COCODataset, detokenize
from .embeddings import pretrained_weights
from . import config
from .inference import generate

@torch.inference_mode()
def evaluate(model, dataset, batch_size=50, max_batches=None):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(config.device) for t in batch]
        X, Y = batch
        X = image_encoder(X)
        logits = model(X, Y[:, :-1])
        loss = loss_fn(logits, Y[:, 1:], dataset.vocab["<pad>"])
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train() # reset model back to training mode
    return mean_loss

def loss_fn(logits: torch.Tensor, targets: torch.Tensor, pad_id) -> torch.Tensor:
    v_sz = logits.size()[-1]
    targets = targets.contiguous()
    return F.cross_entropy(logits.contiguous().view(-1, v_sz), targets.view(-1), ignore_index=pad_id).to(logits.device)

def save_model(model, filename):
    torch.save(model.state_dict(), filename)

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
# dataset = FlickrDataset(images_dir=config.images_dir, csv_file=config.csv_file, transform=transform)
# val_dataset = FlickrDataset(images_dir=config.images_dir, csv_file=config.csv_file, transform=transform, mode='val')
dataset = COCODataset(images_dir=config.images_dir, captions_file=config.csv_file, transform=transform)
val_dataset = COCODataset(images_dir=config.val_images_dir, captions_file=config.val_csv_file, transform=transform)

dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

model = Transformer(
    vocab_size=dataset.vocab_size,
    d_model=512,
    img_encode_size=196,
    enc_ff_dim=512,
    dec_ff_dim=2048,
    enc_n_layers=2,
    dec_n_layers=4,
    enc_n_heads=8,
    dec_n_heads=8,
    max_len=config.max_len - 1,
    dropout=config.dropout,
    pad_id=dataset.vocab["<pad>"],
)
image_encoder = ImageEncoder()
image_encoder.fine_tune(True)
model.to(device=config.device)


image_encoder.to(device=config.device)

def train():
    model.train()
    print("loading pretrained glove embeddings...")
    weights = pretrained_weights(dataset.vocab, 512)
    model.decoder.cptn_emb.from_pretrained(weights.to(config.device), freeze=True, padding_idx=dataset.vocab["<pad>"])
    list(model.decoder.cptn_emb.parameters())[0].requires_grad = False
    image_encoder.train()

    model_params = filter(lambda p: p.requires_grad, model.parameters())
    image_encoder_params = list(filter(lambda p: p.requires_grad, image_encoder.parameters()))
    print(f"train dataset {len(dataset)} max len {config.max_len}, vocab size {dataset.vocab_size}, padding idx {dataset.vocab['<pad>']}")
    print(f"val dataset {len(val_dataset)}")
    print(f"model #params: {sum(p.numel() for p in model.parameters())}")
    print(f"image encoder #params: {sum(p.numel() for p in image_encoder_params)}")
    print(f"using device {config.device}")

    optimizer = Adam(params=model_params, lr=1e-4)
    image_encoder_optim = Adam(params=image_encoder_params, lr=1e-4)

    lossi = []
    for step, (images, captions) in enumerate(dataloader):
        images, captions = images.to(config.device), captions.to(config.device)
        t0 = time.time()
        images_encoded = image_encoder(images)
        logits = model(images_encoded, captions[:, :-1])
        loss = loss_fn(logits, captions[:, 1:], dataset.vocab["<pad>"])

        model.zero_grad(set_to_none=True)
        image_encoder.zero_grad(set_to_none=True)

        loss.backward()

        image_encoder_optim.step()
        optimizer.step()

        # wait for all CUDA work on the GPU to finish then calculate iteration time taken
        if config.device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.time()

        # logging
        if step % 10 == 0:
            print(f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms")

        if step % 10 == 0:
            probs  = F.softmax(logits, dim=-1)
            _, idx = torch.topk(probs, k=1, dim=-1) 
            idx = idx.squeeze(-1)
            print("Generated logits on batch item 0", detokenize(dataset.vocab, idx[0].numpy(force=True)))

            train_loss = evaluate(model, dataset, batch_size=100, max_batches=10)
            test_loss  = evaluate(model, val_dataset, batch_size=100, max_batches=10)
            print(f"step {step} | train mean loss {train_loss:.4f} | step time {(t1-t0)*1000:.2f}ms")
            print(f"step {step} | val mean loss {test_loss:.4f} | step time {(t1-t0)*1000:.2f}ms")
            
        if step == 30:
            for p in model.decoder.cptn_emb.parameters():
                p.requires_grad = True
                optimizer.add_param_group({"params": p})
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

        lossi.append(loss.item())
        if config.max_steps >= 0 and step >= config.max_steps:
            save_model(model, "captions.pt")
            save_model(image_encoder, "image_encoder.pt")
            break


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("quitting")
        save_model(model, "captions.pt")
        save_model(image_encoder, "image_encoder.pt")
        raise