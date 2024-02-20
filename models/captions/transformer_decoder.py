from typing import Tuple
from torch import nn, Tensor
from torch.nn import MultiheadAttention


class DecoderBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super(DecoderBlock, self).__init__()
        """
        param:
        d_model:    features size.
                    int

        num_heads:  number of heads in the multiheadattention model.
                    int

        dropout:    dropout value
                    float
        """

        self.dec_self_attn = MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.cross_attn = MultiheadAttention(d_model, num_heads, dropout=dropout)

        self.self_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.multihead_dropout = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(d_model * 4, d_model),
        )

        self.ff_norm = nn.LayerNorm(d_model)
        self.ff_dropout = nn.Dropout(dropout)

    def forward(
        self,
        dec_inputs: Tensor,
        enc_outputs: Tensor,
        tgt_mask: Tensor | None = None,
        tgt_pad_mask: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        param:
        dec_inputs:     Captions to decode
                        Tensor
                        [max_len, batch_size, embed_dim]

        enc_outputs:    Encoded image to decode
                        Tensor
                        [encode_size^2=196, batch_size, embed_dim]

        tgt_mask:       Mask to ensure that decoder doesn't look at future
                        tokens from a given subsequence
                        [max_len , max_len]

        tgt_pad_mask:   Mask to ensure that decoder doesn't look at padding tokens
                        [batch_size , max_len]

        outputs:
        output:         Decoder output
                        Tensor
                        [max_len, batch_size, embed_dim]
        """
        # self attention + resedual summation + norm
        output, _ = self.dec_self_attn(
            dec_inputs,
            dec_inputs,
            dec_inputs,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_pad_mask,
        )
        output = dec_inputs + self.self_attn_dropout(output)
        output = self.self_attn_norm(output)  # type: Tensor

        # # self attention + residual + norm + FF
        output2, _ = self.cross_attn(output, enc_outputs, enc_outputs)
        output = output + self.multihead_dropout(output2)
        output = self.cross_attn_norm(output)

        output2 = self.ff(output)  # type: Tensor
        output = self.ff_norm(output + self.ff_dropout(output2))

        return output


if __name__ == "__main__":
    import torch

    src_img = torch.rand(196, 10, 512)  # encode, B, embed
    captn = torch.rand(52, 10, 512)  # max_len, B, embed
    m_test = DecoderBlock(512, 8, 0.1)
    valus = m_test(captn, src_img, None)
    print(valus.size())
