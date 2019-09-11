from pytorch_transformers import BertModel, BertPreTrainedModel
from pytorch_transformers.modeling_bert import BertEncoder, BertOutput, BertAttention, BertIntermediate, BertLayer
import torch
from torch import nn
from torch.nn import CrossEntropyLoss


class BertConcatForStatefulSearch(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """
    def __init__(self, config):
        super(BertConcatForStatefulSearch, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

    
class HierBertConcatForStatefulSearch(BertConcatForStatefulSearch):
    def __init__(self, config):
        super(HierBertConcatForStatefulSearch, self).__init__(config)
        self.bert = HierBertModel(config)
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, hier_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask, hier_mask=hier_mask)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class HierBertModel(BertModel):
    def __init__(self, config):
        super(HierBertModel, self).__init__(config)
        self.encoder = HierBertEncoder(config)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, 
                head_mask=None, hier_mask=None):
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask,
                                       hier_mask=hier_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)
        
        
class HierBertEncoder(BertEncoder):
    def __init__(self, config):
        super(HierBertEncoder, self).__init__(config)
        self.layer = nn.ModuleList([HierBertLayer(config) for _ in range(config.num_hidden_layers)])
        
    def forward(self, hidden_states, attention_mask, head_mask=None, hier_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], hier_mask)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

        
class HierBertLayer(nn.Module):
    def __init__(self, config):
        super(HierBertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = HierBertOutput(config)
        
        self.hier = HierAttentionLayer(config)
        # self.hier.att.attention.load_state_dict(self.attention.state_dict())
        # self.hier.att.intermediate.load_state_dict(self.intermediate.state_dict())
        # self.hier.att.output.load_state_dict(self.output.state_dict())

    def forward(self, hidden_states, attention_mask, head_mask=None, hier_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        
        hier_output = self.hier(hidden_states, hier_mask)
                
        layer_output = self.output(intermediate_output, attention_output, hier_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
                
        return outputs
    
class HierBertOutput(BertOutput):
    def __init__(self, config):
        super(HierBertOutput, self).__init__(config)

    def forward(self, hidden_states, input_tensor, hier_output):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor + hier_output)
        return hidden_states
    
    
class HierAttentionLayer(nn.Module):
    def __init__(self, config):
        super(HierAttentionLayer, self).__init__()
        self.att = BertLayer(config)
        
    def forward(self, hidden_states, hier_mask):
        
        # hidden_states: batch_size, seq_len, hidden_size
        # hier_mask:     batch_size, seq_len
        
        batch_size = hidden_states.size(0)
        
        attention_mask_list = []
        hidden_states_list = []
        target_ids = [-1] + list(range(1, hier_mask.max() + 1))

        for target_id in target_ids:
            mask = torch.zeros_like(hier_mask, device=hier_mask.device, dtype=torch.float)
            mask[hier_mask == target_id] = 1.0
            attention_mask_list.append(mask)
            hidden_states_list.append(hidden_states)

        
        attention_mask = torch.cat(attention_mask_list, dim=0)
        all_hidden_states = torch.cat(hidden_states_list, dim=0)
        
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
                        
        layer_outputs = self.att(all_hidden_states, extended_attention_mask)
        outputs = layer_outputs[0]
        
        splits = outputs.split(batch_size, dim=0)
        hier_output = torch.zeros_like(hidden_states, device=hidden_states.device)
        for split, mask in zip(splits, attention_mask_list):
            hier_output += split * mask.unsqueeze(-1)
            
        return hier_output
        
        
        
        
        
        
        
    
    
    
    
    