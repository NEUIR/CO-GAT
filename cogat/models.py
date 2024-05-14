import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ReLU, CrossEntropyLoss
from transformers import AutoModel


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class DPTModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(DPTModel, self).__init__()
        self.args = args
        self.num_labels = args.project_dim
        self.model = AutoModel.from_pretrained(args.model_name_or_path)
        self.evi_num = args.evi_num
        self.hidden_size = args.hidden_size
        self.loss_fct = CrossEntropyLoss()
        self.pre_cls = nn.Linear(self.hidden_size, 2)
        self.classfication = nn.Linear(self.hidden_size, self.num_labels)
        self.aggregator = nn.Linear(self.hidden_size, 1)
        self.num_head = self.hidden_size // 64
        self.attentions = MultiHeadAttention(self.num_head, self.hidden_size, 64, 64)

    def reason_graph(self, cls_hidden_state, attentions):
        ## self-attention
        bsz = cls_hidden_state.size(0)
        cls_hidden_state = cls_hidden_state.contiguous().view(bsz, -1, self.hidden_size)
        v, att = attentions(cls_hidden_state, cls_hidden_state, cls_hidden_state)
        sz_v, len_v, hidden_dim = v.size(0), v.size(1), v.size(2)
        v = v.view(sz_v, len_v, hidden_dim)
        agg_atten = self.aggregator(v)
        agg_atten = F.softmax(agg_atten, dim=1)
        output = torch.sum(agg_atten * v, 1)
        return output

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, ground_truth=None, labels=None,
                return_dict=None):
        bsz, max_len = attention_mask.size()
        if self.args.roberta:
            outputs = self.model(input_ids, attention_mask=attention_mask,
                                 return_dict=return_dict, )
        else:
            outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,return_dict=return_dict, )
        raw_logits = outputs[0]
        cls_hidden_state = raw_logits.view(-1, self.evi_num, max_len, self.hidden_size)
        cls_claim = cls_hidden_state[:, 0, 0, :].unsqueeze(1)
        cls_claim = cls_claim.repeat(1, self.evi_num - 1, 1)
        cls_evidence = cls_hidden_state[:, 1:, 0, :]
        confin = self.pre_cls(cls_evidence)
        evi_logit = F.log_softmax(confin, -1).view(-1, confin.size(2))
        confin = F.softmax(confin, dim=-1)
        hidden_state = torch.cat((cls_claim, cls_evidence), -1).view(cls_evidence.size(0), cls_evidence.size(1), -1,
                                                                     self.hidden_size)
        hidden_state = torch.sum(torch.mul(confin.unsqueeze(-1), hidden_state), 2)
        graph= self.reason_graph(hidden_state, self.attentions)
        logits = self.classfication(graph)
        logits = F.log_softmax(logits, -1)

        if labels is not None:
            ground_truth = ground_truth.view(-1, 1).squeeze(-1)
            evi_loss = F.nll_loss(evi_logit, ground_truth.to(evi_logit.device), reduction='mean')
            fact_loss = F.nll_loss(logits, labels, reduction='mean')
            loss = fact_loss + evi_loss
            if self.args.ablation:
                loss = fact_loss
            return loss
        else:
            return logits
