import torch.nn as nn
import torch.nn.functional as F
from model.RNN import RNN
import math
import torch
from model.GraphEncoder import GraphEncoder
from model.StageNet import StageNet
from model.hita_transformer import HitaTransformer

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.xavier_uniform_(module.weight)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

class DearLLM(nn.Module):
    def __init__(self, args,
                 **kwargs):
        super(DearLLM, self).__init__()
        self.device = torch.device(args.cuda_choice if torch.cuda.is_available() else "cpu")
        self.dataset = kwargs["dataset"]
        self.feature_keys=kwargs["feature_keys"]
        self.label_key=kwargs["label_key"]
        self.mode=kwargs["mode"]
        self.modeltype = args.modeltype
        self.train_dropout_rate = args.train_dropout_rate
        self.hidden_dim = args.hidden_dim
        self.embed_dim = args.embed_dim

        if args.modeltype == "GRU":
            self.model = RNN(
                dataset=self.dataset,
                feature_keys=self.feature_keys,
                label_key=self.label_key,
                mode=self.mode,
                embedding_dim = self.embed_dim,
                hidden_dim = self.hidden_dim,
                train_dropout_rate = self.train_dropout_rate,
                rnn_type = "GRU",
                num_layers= args.encoder_layer
            )
            self.patient_dim = len(self.model.feature_keys) * self.model.hidden_dim

        elif args.modeltype == "StageNet":
            self.model = StageNet(
                dataset=self.dataset,
                feature_keys=self.feature_keys,
                label_key=self.label_key,
                mode=self.mode,
                embedding_dim = self.embed_dim,
                train_dropout_rate = self.train_dropout_rate,
                chunk_size = args.chunk_size,
                levels = args.levels
            )
            self.patient_dim = len(self.model.feature_keys) * args.chunk_size * args.levels

        elif args.modeltype == "HiTANet":
            self.model = HitaTransformer(
                dataset=self.dataset,
                feature_keys=self.feature_keys,
                label_key=self.label_key,
                mode=self.mode,
                embedding_dim = self.embed_dim,
                train_dropout_rate = self.train_dropout_rate,
                num_layers= args.encoder_layer,
                num_heads = args.encoder_head,
                device = self.device
            )
            self.patient_dim = len(self.model.feature_keys) * self.model.embedding_dim

        
        self.node_num = kwargs["node_num"]
        self.gencoder_dim_list = [args.node_dim] + eval(args.gencoder_dim_list)
        self.n_layers_gencoder = len(eval(args.gencoder_dim_list))
        self.node_embedding = nn.Embedding(self.node_num, args.node_dim)
        self.graph_model = GraphEncoder(self.gencoder_dim_list, self.n_layers_gencoder)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(self.train_dropout_rate)

        output_size = self.model.get_output_size(self.model.label_tokenizer)
        assert len(self.gencoder_dim_list) > 0
        self.classfier_fc = nn.Linear(self.patient_dim + self.gencoder_dim_list[-1], output_size)

    def _init_weight(self):
        self.model.apply(init_weights)
        self.classfier_fc.apply(init_weights())
        nn.init.xavier_uniform_(self.node_embedding)

    def forward(self, graph_batch, **data):
        patient_embed, patient_emb_all_step = self.model(**data)
        patient_y_true = self.model.prepare_labels(data[self.model.label_key], self.model.label_tokenizer)
        graph_embed = self.graph_model(graph_batch, self.node_embedding)
        
        all_embed = torch.cat((patient_embed, graph_embed), dim=1)

        all_embed = self.dropout(all_embed)
        cls_logits = self.classfier_fc(all_embed)
        patient_y_prob = self.model.prepare_y_prob(cls_logits)
        loss_cls = F.binary_cross_entropy_with_logits(cls_logits, patient_y_true)

        return loss_cls, cls_logits, patient_y_true, patient_y_prob, patient_embed