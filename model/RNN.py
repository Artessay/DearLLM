from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from datasets import SampleEHRDataset
from models import BaseModel

class RNNLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rnn_type: str = "LSTM",     
        num_layers: int = 1,
        dropout: float = 0.5,
        bidirectional: bool = False,
    ):
        super(RNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.dropout_layer = nn.Dropout(dropout)
        self.num_directions = 2 if bidirectional else 1
        rnn_module = getattr(nn, rnn_type)
        self.rnn = rnn_module(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        if bidirectional:
            self.down_projection = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
        self,
        x: torch.tensor,
        mask: Optional[torch.tensor] = None,
    ) -> Tuple[torch.tensor, torch.tensor]:
        x = self.dropout_layer(x)
        batch_size = x.size(0)
        if mask is None:
            lengths = torch.full(
                size=(batch_size,), fill_value=x.size(1), dtype=torch.int64
            )
        else:
            lengths = torch.sum(mask.int(), dim=-1).cpu()
        lengths = torch.where(lengths<=0, torch.ones_like(lengths), lengths)
        x = rnn_utils.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        outputs, _ = self.rnn(x)
        outputs, _ = rnn_utils.pad_packed_sequence(outputs, batch_first=True)
        if not self.bidirectional:
            last_outputs = outputs[torch.arange(batch_size), (lengths - 1), :]
            return outputs, last_outputs
        else:
            outputs = outputs.view(batch_size, outputs.shape[1], 2, -1)
            f_last_outputs = outputs[torch.arange(batch_size), (lengths - 1), 0, :]
            b_last_outputs = outputs[:, 0, 1, :]
            last_outputs = torch.cat([f_last_outputs, b_last_outputs], dim=-1)
            outputs = outputs.view(batch_size, outputs.shape[1], -1)
            last_outputs = self.down_projection(last_outputs)
            outputs = self.down_projection(outputs)
            return outputs, last_outputs


class RNN(BaseModel):
    def __init__(
        self,
        dataset: SampleEHRDataset,
        feature_keys: List[str],
        label_key: str,
        mode: str,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        train_dropout_rate: float=0.5, 
        rnn_type: str="LSTM",
        num_layers: int= 1,
        **kwargs
    ):
        super(RNN, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type
        self.num_layers = num_layers

        if "input_size" in kwargs:
            raise ValueError("input_size is determined by embedding_dim")
        if "hidden_size" in kwargs:
            raise ValueError("hidden_size is determined by hidden_dim")

        self.feat_tokenizers = {}
        self.label_tokenizer = self.get_label_tokenizer()
        self.embeddings = nn.ModuleDict()
        
        self.linear_layers = nn.ModuleDict()
        self.dropout = nn.Dropout(train_dropout_rate)

        
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            if input_info["type"] not in [str, float, int]:
                raise ValueError(
                    "RNN only supports str code, float and int as input types"
                )
            elif (input_info["type"] == str) and (input_info["dim"] not in [2, 3]):
                raise ValueError(
                    "RNN only supports 2-dim or 3-dim str code as input types"
                )
            elif (input_info["type"] in [float, int]) and (
                input_info["dim"] not in [2, 3]
            ):
                raise ValueError(
                    "RNN only supports 2-dim or 3-dim float and int as input types"
                )

            self.add_feature_transform_layer(feature_key, input_info)

        self.rnn = nn.ModuleDict()
        for feature_key in feature_keys:
            self.rnn[feature_key] = RNNLayer(
                input_size=embedding_dim, hidden_size=hidden_dim, rnn_type = rnn_type, num_layers = num_layers, dropout= train_dropout_rate,
                **kwargs
            )

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
                x = torch.sum(x, dim=2)
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

            _, x = self.rnn[feature_key](x, mask)
            patient_emb.append(x)
            patient_emb_all_step.append(_)

        patient_emb = torch.cat(patient_emb, dim=1)
        patient_emb_all_step = torch.cat(patient_emb_all_step, dim = -1)
        return patient_emb, patient_emb_all_step