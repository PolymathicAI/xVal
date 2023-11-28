import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import DatasetDict
from transformers import PreTrainedTokenizerFast
import torch.optim as optim


class Numformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=768,
        nhead=6,
        num_layers=6,
        dim_feedforward=3072,
        dropout=0.1,
        activation=nn.GELU(),
        layer_norm_eps=1e-05,
        batch_first=True,
        norm_first=True,
        transformer_bias=False,
        numhead_bias=True,
        context_length=1024,
        is_causal=False,
    ):
        super().__init__()
        encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            # bias=transformer_bias,
        )
        self.encoder_stack = nn.TransformerEncoder(
            encoder_layer=encoder, num_layers=num_layers, enable_nested_tensor=False
        )
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.position_embed = nn.Embedding(context_length, d_model)
        self.lm_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, bias=transformer_bias),
            nn.GELU(),
            nn.Linear(dim_feedforward, vocab_size, bias=transformer_bias),
        )
        self.num_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, bias=numhead_bias),
            nn.GELU(),
            nn.Linear(dim_feedforward, 1, bias=numhead_bias),
        )
        self.is_causal = is_causal

    def forward(self, x, x_num):
        x = self.token_embed(x) + self.position_embed.weight[: x.shape[1]].unsqueeze(0)
        x = x * x_num.unsqueeze(-1)
        x = self.encoder_stack(x, is_causal=self.is_causal)
        logit_preds = self.lm_head(x)
        num_preds = self.num_head(x)
        return logit_preds, num_preds


### Define collator and data loaders
def define_masked_num_collator(pad_token_id, mask_token_id, mlm_probability):
    def masked_num_collator(batch):
        x = [torch.tensor(sample["input_ids"]) for sample in batch]
        x_num = [torch.tensor(sample["numbers"]) for sample in batch]
        x = pad_sequence(x, batch_first=True, padding_value=pad_token_id)
        x_num = pad_sequence(x_num, batch_first=True, padding_value=1)
        probability_matrix = torch.full(x.shape, mlm_probability)
        mask = torch.bernoulli(probability_matrix).bool()
        y = x.clone()
        y_num = x_num.clone()
        y[~mask] = -100
        x[mask] = mask_token_id
        x_num[mask] = 1
        return {"x": x, "x_num": x_num, "y": y, "y_num": y_num, "mask": mask}

    return masked_num_collator
