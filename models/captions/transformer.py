from copy import deepcopy

import torch
from torch import nn, Tensor
from torch.functional import F

from .transformer_encoder import EncoderBlock
from .transformer_decoder import DecoderBlock
from .positional_encoding import PositionalEncoding


class Encoder(nn.Module):
    """
    param:

    layer:      an instance of the EecoderLayer() class

    num_layers: the number of decoder-layers
                int
    """

    def __init__(self, layer: EncoderBlock, num_layers: int):
        super().__init__()
        # Make copies of the encoder layer
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(num_layers)])

    def forward(self, x: Tensor) -> Tensor:
        """
        param:
        x:  encoder input
            Tensor
            [encode_size^2, batch_size, image_embed_dim]

        outputs:
        x:  encoder output
            Tensor
            [encode_size^2, batch_size, model_embed_dim]
        """

        for layer in self.layers:
            x = layer(x)

        return x


class Decoder(nn.Module):
    """
    param:
    layer:          an instance of the EecoderLayer() class

    vocab_size:     the number of vocabulary
                    int

    d_model:        size of features in the transformer inputs
                    int

    num_layers:     the number of decoder-layers
                    int

    max_len:        maximum len pf target captions
                    int

    dropout:        dropout value
                    float

    """

    def __init__(
        self,
        layer: DecoderBlock,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        max_len: int,
        dropout: float,
        pad_id: int,
    ):
        super().__init__()
        self.pad_id = pad_id
        # Embedding layer + pos encoding
        self.cptn_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len)

        # Make copies of the decoder layer
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(num_layers)])

        self.dropout = nn.Dropout(p=dropout)

    def get_attn_subsequent_mask(self, sz: int) -> Tensor:
        """
        Generates an upper-triangular matrix of -inf, with zeros on diag.
        """
        return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)

    def forward(self, targets: Tensor, src_img: Tensor) -> Tensor:
        """
        param:
        targets:   Captions (Transformer target sequence)
                    Tensor
                    [batch_size, max_len-1]

        src_img:    Encoded images (Transformer source sequence)
                    Tensor
                    [encode_size^2, batch_size, image_embed_dim]

        outputs:
        output:     Decoder output
                    Tensor
                    [max_len, batch_size, model_embed_dim]
        """

        # create masks, then pass to decoder
        tgt_pad_mask = targets == self.pad_id
        tgt_mask = self.get_attn_subsequent_mask(targets.size()[1])
        tgt_mask = tgt_mask.to(targets.device)

        # encode captions + pos enc
        # (B, max_len) -> (B, max_len, d_model) -> (max_len, B, d_model)
        targets = self.cptn_emb(targets)  # type: Tensor
        targets = self.dropout(self.pos_emb(targets.permute(1, 0, 2)))

        for layer in self.layers:
            targets = layer(targets, src_img, tgt_mask, tgt_pad_mask)
        # [layer_num, batch_size, head_num, max_len, encode_size**2]

        return targets


class Transformer(nn.Module):
    """ """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        img_encode_size: int,
        enc_n_layers: int,
        dec_n_layers: int,
        enc_n_heads: int,
        dec_n_heads: int,
        max_len: int,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super(Transformer, self).__init__()
        encoder_layer = EncoderBlock(
            img_encode_size=img_encode_size,
            img_embed_dim=d_model,
            num_heads=enc_n_heads,
            dropout=dropout,
        )
        decoder_layer = DecoderBlock(
            d_model=d_model,
            num_heads=dec_n_heads,
            dropout=dropout,
        )
        self.encoder = Encoder(layer=encoder_layer, num_layers=enc_n_layers)
        self.decoder = Decoder(
            layer=decoder_layer,
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=dec_n_layers,
            max_len=max_len,
            dropout=dropout,
            pad_id=pad_id,
        )

        self.predictor = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self, images: Tensor, captions: Tensor, targets: Tensor | None = None
    ) -> Tensor:
        """
        param:
        image:      source images
                    [batch_size, encode_size^2=196, image_feature_size=512]

        captions:   target captions
                    [batch_size, max_len-1=51]

        outputs:
        predictions:    Decoder output
                        Tensor
                        [batch_size, max_len, vocab_size]
        """
        # encode, decode, predict
        images_encoded = self.encoder(images.permute(1, 0, 2))  # type: Tensor
        captions = self.decoder(captions, images_encoded)
        predictions = self.predictor(captions).permute(1, 0, 2)  # type: Tensor

        logits = F.softmax(predictions, dim=-1)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.contiguous().view(-1),
                ignore_index=-1,
            )

        return logits, loss


if __name__ == "__main__":

    src_img = torch.rand(10, 196, 512)  # B, encode, embed
    captions = torch.randint(0, 52, (10, 30), dtype=torch.long)
    m_test = Transformer(52, 512, 196, 2, 8, 8, 8, 30, 0.1, 0)
    print(captions[:, 1:].shape, captions[:, 1:])
    logits, loss = m_test(
        src_img, captions[:, :-1], captions[:, 1:]
    )  # B, max_len, vocab_size
    loss.backward()
    print(loss)
    for p in m_test.parameters():
        print(p.requires_grad, p.grad)
