import torch
import json
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class BERT_SCL(nn.Module):
    def __init__(self,  opt,bert,):
        super(BERT_SCL, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.num_labels)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        #print(type( inputs[0]))
        #print(inputs[0].shape)
        #print(inputs[1].shape)
        #_, pooled_output = self.bert(text_bert_indices, token_type_ids=bert_segments_ids)
        #print(type(pooled_output))
        
        output = self.bert(text_bert_indices, token_type_ids=bert_segments_ids)
        pooled_output = output.pooler_output
        #pooled_output=torch.tensor(pooled_output)
        pooled_output = pooled_output.clone().detach()
        pooled_output = self.dropout(pooled_output)

        feature = pooled_output.unsqueeze(1)
        feature = F.normalize(feature, dim=2)

        logits = self.dense(pooled_output)

        return feature, logits
