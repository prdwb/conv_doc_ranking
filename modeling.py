import torch
from pytorch_transformers import BertModel, BertPreTrainedModel
from pytorch_transformers.modeling_bert import (BertEncoder, BertOutput, BertAttention, 
                                                BertIntermediate, BertLayer, BertEmbeddings,
                                                BertPooler, BertLayerNorm)
from torch import nn
from torch.nn import CrossEntropyLoss
from copy import deepcopy


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
        # print('input_ids', input_ids)
        # print('hier_mask', hier_mask)
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
        self.output = BertOutput(config)
        
        self.hier = HierAttentionLayer(config)
        # self.dense = nn.Linear(2 * config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, attention_mask, head_mask=None, hier_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]                
        hier_output = self.hier(hidden_states, hier_mask)
        combined_output = attention_output + hier_output
        
        intermediate_output = self.intermediate(combined_output)
        # attention_hier_concat = torch.cat((attention_output, hier_output), dim=2)
        # intermediate_output = self.intermediate(self.dense(attention_hier_concat))
                
        layer_output = self.output(intermediate_output, combined_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
                
        return outputs
    
# class HierBertOutput(BertOutput):
#     def __init__(self, config):
#         super(HierBertOutput, self).__init__(config)

#     def forward(self, hidden_states, input_tensor, hier_output):
#         print('hidden_states in bert output', hidden_states.size())
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states + input_tensor)
#         return hidden_states
    
    
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
        # target_ids = [-1] + list(range(1, hier_mask.max() + 1))
        target_ids = list(range(1, hier_mask.max() + 1))
        # print('target_ids', target_ids)
    
        for target_id in target_ids:
            mask = torch.zeros_like(hier_mask, device=hier_mask.device, dtype=torch.float)
            mask[hier_mask == target_id] = 1.0
            attention_mask_list.append(deepcopy(mask))
            hidden_states_list.append(hidden_states)
        # print('attention_mask_list', attention_mask_list)
        
        attention_mask = torch.cat(attention_mask_list, dim=0)
        all_hidden_states = torch.cat(hidden_states_list, dim=0)
        # print('attention_mask size', attention_mask.size())
        # print('all_hidden_states size', all_hidden_states.size())
        
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
                        
        layer_outputs = self.att(all_hidden_states, extended_attention_mask)
        outputs = layer_outputs[0]
        
        splits = outputs.split(batch_size, dim=0)
        hier_output = torch.zeros_like(hidden_states, device=hidden_states.device, dtype=torch.float)
        for split, mask in zip(splits, attention_mask_list):
            # print('split size', split.size())
            # print('mask size', mask.size())
            hier_output += split * mask.unsqueeze(-1)
            
        return hier_output
        
               
class BehaviorAwareBertConcatForStatefulSearch(BertPreTrainedModel):

    def __init__(self, config):
        super(BehaviorAwareBertConcatForStatefulSearch, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BehaviorAwareBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, hier_mask=None, 
                behavior_rel_pos_mask=None, behavior_type_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask, hier_mask=hier_mask, 
                            behavior_rel_pos_mask=behavior_rel_pos_mask, behavior_type_mask=behavior_type_mask)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)        
        
    
class BehaviorAwareBertModel(BertModel):

    def __init__(self, config):
        super(BehaviorAwareBertModel, self).__init__(config)

        self.embeddings = BehaviorAwareBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()    
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, 
                hier_mask=None, behavior_rel_pos_mask=None, behavior_type_mask=None):
        
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

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids, 
                                           hier_mask=hier_mask, behavior_rel_pos_mask=behavior_rel_pos_mask, 
                                           behavior_type_mask=behavior_type_mask)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

        
class BehaviorAwareBertEmbeddings(BertEmbeddings):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BehaviorAwareBertEmbeddings, self).__init__(config)
        
        self.behavior_rel_pos_embeddings = nn.Embedding(4, config.hidden_size)
        self.behavior_type_embeddings = nn.Embedding(4, config.hidden_size)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, 
                hier_mask=None, behavior_rel_pos_mask=None, behavior_type_mask=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        behavior_rel_pos_embeddings = self.behavior_rel_pos_embeddings(behavior_rel_pos_mask)
        behavior_type_embeddings = self.behavior_type_embeddings(behavior_type_mask)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings + \
                     behavior_rel_pos_embeddings + behavior_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings      

class HierAttBertConcatForStatefulSearch(BertPreTrainedModel):
    """ Use BertConcat as usual, then obtain a rep for each turn with multi-head att on the tokens in this turn.
    Finally do another multi-head att on all rep: [CLS] [rep for current turn] ... [rep for first turn]
    """

    def __init__(self, config):
        super(HierAttBertConcatForStatefulSearch, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        
        self.turn_att = BertLayer(config)
        self.sess_att = BertLayer(config)
        
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = BertPooler(config)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, hier_mask=None, 
                behavior_rel_pos_mask=None, behavior_type_mask=None):
        
        # run bert concat and get contextual rep for every token
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]
        
        behavior_rel_pos_mask_plus = behavior_rel_pos_mask + 1
        behavior_rel_pos_mask_plus[attention_mask == 0] = 0
        # print('behavior_rel_pos_mask_plus', behavior_rel_pos_mask_plus)
        
        # isolate each turn and run a bert layer to get a rep for each turn
        batch_size, max_seq_len, hidden_size = sequence_output.size()        
        within_turn_attention_mask_list = []
        hidden_states_list = []
        target_ids = list(range(1, behavior_rel_pos_mask_plus.max() + 1))
        # print('target_ids', target_ids)
    
        for target_id in target_ids:
            mask = torch.zeros_like(behavior_rel_pos_mask_plus, device=behavior_rel_pos_mask_plus.device)
            mask[behavior_rel_pos_mask_plus == target_id] = 1
            mask[:, 0] = 1
            within_turn_attention_mask_list.append(deepcopy(mask))
            hidden_states_list.append(sequence_output)
        
        within_turn_attention_mask = torch.cat(within_turn_attention_mask_list, dim=0)
        # print('within_turn_attention_mask', within_turn_attention_mask)
        all_hidden_states = torch.cat(hidden_states_list, dim=0)
        # print('attention_mask size', attention_mask.size())
        # print('all_hidden_states size', all_hidden_states.size())
        
        extended_attention_mask = within_turn_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # print('all_hidden_states', all_hidden_states.tolist())
        layer_outputs = self.turn_att(all_hidden_states, extended_attention_mask)
        outputs = layer_outputs[0]
        
        splits = outputs.split(batch_size, dim=0)
        turn_reps = [sequence_output[:, 0]]
        for split, mask in zip(splits, within_turn_attention_mask_list):
            mask = mask.float()
            turn_rep = split * mask.unsqueeze(-1)
            turn_rep_sum = turn_rep.sum(dim=1)
            turn_rep = turn_rep_sum / mask.sum(dim=1, keepdim=True)
            turn_reps.append(turn_rep)
        turn_reps = torch.stack(turn_reps, dim=1)
        
        # run a bert layer on all turn reps: [CLS] [rep for current turn] ... [rep for first turn]
        turn_mask_len = behavior_rel_pos_mask_plus.max(dim=1).values + 1
        # max_len = turn_mask_len.max() + 1
        max_len = 5
        turn_mask = torch.arange(max_len, device=behavior_rel_pos_mask_plus.device,
                       dtype=behavior_rel_pos_mask_plus.dtype).expand(len(turn_mask_len), max_len) < turn_mask_len.unsqueeze(1)
        turn_mask = torch.as_tensor(turn_mask, dtype=behavior_rel_pos_mask_plus.dtype, 
                                               device=behavior_rel_pos_mask_plus.device)
        # print('turn_mask', turn_mask)
                
        extended_attention_mask = turn_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # print('turn_reps', turn_reps.tolist())
        
        # pad turn_repsclasscls  
        turn_reps_padding = torch.ones((batch_size, max_len - turn_reps.size(1), hidden_size), 
                                       dtype=turn_reps.dtype, device=turn_reps.device)
        turn_reps = torch.cat((turn_reps, turn_reps_padding), dim=1)
        
        # the position_ids is actually the reverse turn ids
        position_ids = torch.arange(max_len, dtype=torch.long, device=turn_reps.device)
        position_ids = position_ids.unsqueeze(0).expand_as(turn_mask)        
        position_embeddings = self.bert.embeddings.position_embeddings(position_ids)

        turn_reps += position_embeddings
        turn_reps = self.LayerNorm(turn_reps)
        turn_reps = self.dropout(turn_reps)
        
        layer_outputs = self.sess_att(turn_reps, extended_attention_mask)
        sess_att_output = layer_outputs[0]
        pooled_output = self.pooler(sess_att_output)
        
        pooled_output = self.dropout(pooled_output)      
        logits = self.classifier(pooled_output)

        outputs = (logits,) + layer_outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
    
    
    
    
    
    
    
    
    
    
    
    