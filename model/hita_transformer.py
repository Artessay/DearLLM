import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from datasets import SampleEHRDataset
from models import BaseModel

if __name__ == "__main__":
    from hita_transformer_layer import HitaTransformerLayer
else:
    from .hita_transformer_layer import HitaTransformerLayer

class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = torch.softmax(scores, dim=-1)
        if mask is not None:
            p_attn = p_attn.masked_fill(mask == 0, 0)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_model, d_model, bias=False) for _ in range(3)]
        )
        self.output_linear = nn.Linear(d_model, d_model, bias=False)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query, key, value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]

        if mask is not None:
            mask = mask.unsqueeze(1)
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, mask=None):
        x = self.w_2(self.dropout(self.activation(self.w_1(x))))
        if mask is not None:
            mask = mask.sum(dim=-1) > 0
            x[~mask] = 0
        return x


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class TransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden, d_ff=4 * hidden, dropout=dropout
        )
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        x = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, lambda _x: self.feed_forward(_x, mask=mask))
        return self.dropout(x)


class TransformerLayer(nn.Module):
    def __init__(self, feature_size, heads=1, dropout=0.5, num_layers=1):
        super(TransformerLayer, self).__init__()
        self.transformer = nn.ModuleList(
            [TransformerBlock(feature_size, heads, dropout) for _ in range(num_layers)]
        )

    def forward(
        self, x: torch.tensor, mask: Optional[torch.tensor] = None
    ) -> Tuple[torch.tensor, torch.tensor]:
        if mask is not None:
            mask = torch.einsum("ab,ac->abc", mask, mask)
        for transformer in self.transformer:
            x = transformer(x, mask)
        emb = x
        cls_emb = x[:, 0, :]
        return emb, cls_emb


class HitaTransformer(BaseModel):
    def __init__(
        self,
        dataset: SampleEHRDataset,
        feature_keys: List[str],
        label_key: str,
        mode: str,
        device,
        embedding_dim: int = 128,
        train_dropout_rate: float = 0.5,  
        num_layers: int = 1,
        num_heads: int = 4,
        **kwargs
    ):
        super(HitaTransformer, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        self.embedding_dim = embedding_dim
        if "feature_size" in kwargs:
            raise ValueError("feature_size is determined by embedding_dim")

        self.feat_tokenizers = {}
        self.label_tokenizer = self.get_label_tokenizer()
        self.embeddings = nn.ModuleDict()
        self.linear_layers = nn.ModuleDict()

        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            
            if input_info["type"] not in [str, float, int]:
                raise ValueError(
                    "Transformer only supports str code, float and int as input types"
                )
            elif (input_info["type"] == str) and (input_info["dim"] not in [2, 3]):
                raise ValueError(
                    "Transformer only supports 2-dim or 3-dim str code as input types"
                )
            elif (input_info["type"] in [float, int]) and (
                input_info["dim"] not in [2, 3]
            ):
                raise ValueError(
                    "Transformer only supports 2-dim or 3-dim float and int as input types"
                )
            self.add_feature_transform_layer(feature_key, input_info)

        self.transformer = nn.ModuleDict()

        self.options = {'n_diagnosis_codes':embedding_dim, 'layer':num_layers,
                        'dropout_rate':train_dropout_rate, 'use_gpu':True, 'num_heads':num_heads}
        for feature_key in feature_keys:
            self.transformer[feature_key] = HitaTransformerLayer(self.options)


    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        patient_emb = []
        patient_emb_all_step = []
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            dim_, type_ = input_info["dim"], input_info["type"]

            if (dim_ == 2) and (type_ == str):
                x = self.feat_tokenizers[feature_key].batch_encode_2d(
                    kwargs[feature_key]
                )
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                x = self.embeddings[feature_key](x)
                
                mask = torch.any(x !=0, dim=2)

            elif (dim_ == 3) and (type_ == str):
                x = self.feat_tokenizers[feature_key].batch_encode_3d(
                    kwargs[feature_key]
                )
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                x = self.embeddings[feature_key](x)
                mask = torch.any(x !=0, dim=2)

            
            elif (dim_ == 2) and (type_ in [float, int]):
                x, mask = self.padding2d(kwargs[feature_key])
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                x = self.linear_layers[feature_key](x)
                mask = mask.bool().to(self.device)

            
            elif (dim_ == 3) and (type_ in [float, int]):
                x, mask = self.padding3d(kwargs[feature_key])
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                x = torch.sum(x, dim=2)
                x = self.linear_layers[feature_key](x)
                mask = mask[:, :, 0]
                mask = mask.bool().to(self.device)

            else:
                raise NotImplementedError

            
            padding_delta_days = torch.ones_like(torch.sum(x, dim=2)).cpu()[:,:-1]
            for (i, batch_delta) in enumerate(kwargs['delta_days']):
                if len(batch_delta)==1: 
                    continue
                for (j, delta_day) in enumerate(batch_delta):
                    if delta_day != [1] and j <= padding_delta_days.shape[1]:
                        padding_delta_days[i][j-1][:] = torch.tensor([delta_day[0] for k in range(padding_delta_days.shape[-1])])

            _, x = self.transformer[feature_key](x, padding_delta_days,
                                                 self.prepare_labels(kwargs[self.label_key],
                                                 self.label_tokenizer), self.options, mask)
            patient_emb.append(x)
            patient_emb_all_step.append(_)

        patient_emb = torch.cat(patient_emb, dim=1)
        patient_emb_all_step = torch.cat(patient_emb_all_step, dim=-1)

        return patient_emb, patient_emb_all_step
