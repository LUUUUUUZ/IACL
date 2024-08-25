import torch as th
from torch import nn

from srs.layers.seframe import SEFrame
from srs.layers.gat import GatLayer
from srs.utils.data.collate import CollateFnGNNContrativeLearning
from srs.utils.data.load import BatchSampler
from srs.utils.Dict import Dict
from srs.utils.prepare_batch import prepare_batch_factory_recursive
from srs.utils.data.transform import seq_to_weighted_graph
import copy
import math


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(th.ones(hidden_size))
        self.bias = nn.Parameter(th.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / th.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + th.erf(x / math.sqrt(2.0)))

class FMLP(nn.Module):
    def __init__(
            self,
            embedding_dim,
            max_seq_length,
            feat_drop=0.0,
    ):
        super(FMLP, self).__init__()
        # FilterLayer:
        self.max_seq_length = max_seq_length
        self.complex_weight = nn.Parameter(th.randn(1, self.max_seq_length//2 + 1, embedding_dim, 2, dtype=th.float32) * 0.02)
        self.out_dropout = nn.Dropout(feat_drop)
        self.LayerNorm = LayerNorm(embedding_dim, eps=1e-12)

        # Intermedia:
        self.dense_1 = nn.Linear(embedding_dim, embedding_dim * 4)
        self.intermediate_act_fn = gelu
        self.dense_2 = nn.Linear(4 * embedding_dim, embedding_dim)
        self.LayerNorm = LayerNorm(embedding_dim, eps=1e-12)
        self.dropout = nn.Dropout(feat_drop)
    
    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        #sequence_emb_fft = torch.rfft(input_tensor, 2, onesided=False)  # [:, :, :, 0]
        #sequence_emb_fft = torch.fft(sequence_emb_fft.transpose(1, 2), 2)[:, :, :, 0].transpose(1, 2)
        # import pdb;pdb.set_trace()
        batch, seq_len, hidden = input_tensor.shape
        x = th.fft.rfft(input_tensor, dim=1, norm='ortho')
        weight = th.view_as_complex(self.complex_weight)
        x = x * weight
        sequence_emb_fft = th.fft.irfft(x, n=seq_len, dim=1, norm='ortho')
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        hidden_states = self.dense_1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SNARM(SEFrame):
    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim,
        knowledge_graph,
        num_layers,
        num_fmlp_layers,
        relu=False,
        batch_norm=True,
        feat_drop=0.0,
        **kwargs
    ):
        super().__init__(
            num_users,
            num_items,
            embedding_dim,
            knowledge_graph,
            num_layers,
            batch_norm=batch_norm,
            feat_drop=feat_drop,
            **kwargs,
        )
        self.pad_embedding = nn.Embedding(1, embedding_dim, max_norm=1)
        self.pad_indices = nn.Parameter(th.arange(1, dtype=th.long), requires_grad=False)
        self.pos_embedding = nn.Embedding(50, embedding_dim, max_norm=1)

        self.fc_i = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.fc_u = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.PSE_layer = GatLayer(
            embedding_dim,
            num_steps=1,
            batch_norm=batch_norm,
            feat_drop=feat_drop,
            relu=relu,
        )
        self.fc_sr = nn.Linear(5 * embedding_dim, embedding_dim, bias=False)
        self.fc_sr1 = nn.Linear(7 * embedding_dim, embedding_dim, bias=False)

        layer = FMLP(2 * embedding_dim, 50, feat_drop)
        self.fmlp_layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(num_fmlp_layers)])


        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs, extra_inputs=None):
        # import pdb;pdb.set_trace() 
        KG_embeddings = super().forward(extra_inputs)
        KG_embeddings["item"] = th.cat([KG_embeddings["item"], self.pad_embedding(self.pad_indices)], dim=0)

        
        uids, padded_seqs, pos, g = inputs
        
        # graph 
        iid = g.ndata['iid']  # (num_nodes,)
        feat_i = KG_embeddings['item'][iid]
        feat_u = KG_embeddings['user'][uids]

        ct_l, ct_g = self.PSE_layer(g, feat_i, feat_u)

        emb_seqs = KG_embeddings["item"][padded_seqs]
        pos_emb = self.pos_embedding(pos)

        feat = th.cat(
            [emb_seqs, pos_emb.unsqueeze(0).expand(emb_seqs.shape)], dim=-1
        )

        query = feat
        if self.dropout is not None:
            query = self.dropout(query)

        hidden_states = query
        for layer_module in self.fmlp_layer:
            hidden_states = layer_module(hidden_states)

        sr = th.cat([ct_l,ct_g, feat_u, hidden_states[:,-1,:]], dim=1)

        sequence_output = self.fc_sr(sr)
        logits = sequence_output @ self.item_embedding(self.item_indices).t()
        return sequence_output, logits

seq_to_graph_fns = [seq_to_weighted_graph]
config = Dict({
    'Model': SNARM,
    'CollateFn': CollateFnGNNContrativeLearning,
    'seq_to_graph_fns': seq_to_graph_fns,
    'BatchSampler': BatchSampler,
    'prepare_batch_factory': prepare_batch_factory_recursive,
})


