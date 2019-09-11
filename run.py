#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="3"


# In[2]:


from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import json

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from pytorch_transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer)
from modeling import BertConcatForStatefulSearch, HierBertConcatForStatefulSearch

from pytorch_transformers import AdamW, WarmupLinearSchedule

from utils import (compute_metrics, convert_examples_to_features, output_modes, ConcatModelDataset)
from scipy.special import softmax


# In[3]:


logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertConcatForStatefulSearch, BertTokenizer),
    'hier': (BertConfig, HierBertConcatForStatefulSearch, BertTokenizer)
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# In[4]:


def train(args, train_dataset, eval_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                  batch_size=args.train_batch_size, num_workers=args.num_workers)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_eval_mrr = 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            guids = batch['guid']
            batch = {k: v.to(args.device) for k, v in batch.items() if k != 'guid'}
            inputs = {'input_ids':      batch['input_ids'],
                      'attention_mask': batch['input_mask'],
                      'token_type_ids': batch['segment_ids'],
                      'labels':         batch['ranker_label_ids']}
            if args.model_type == 'hier':
                inputs['hier_mask'] = batch['hier_mask']
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics                   
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint if it outperforms previous models
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results, eval_output = evaluate(args, eval_dataset, model, 
                                                        tokenizer, args.per_gpu_eval_batch_size)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    
                    if results['mrr'] > best_eval_mrr:
                        best_eval_mrr = results['mrr']
                        output_dir = os.path.join(args.output_dir, 'checkpoint')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)
                        
                        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                        with open(output_eval_file, "w") as writer:
                            logger.info("***** Best eval results so far *****")
                            for key in sorted(results.keys()):
                                logger.info("  %s = %s", key, str(results[key]))
                                writer.write("%s = %s\n" % (key, str(results[key])))
                                
                        output_eval_preds_file = os.path.join(args.output_dir, "eval_preds.txt")
                        with open(output_eval_preds_file, 'w') as writer:
                            json.dump(eval_output, writer)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


# In[5]:


def evaluate(args, eval_dataset, model, tokenizer, batch_size, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task = args.task_name
    eval_output_dir = args.output_dir

    results = {}
    # eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, 
                                 batch_size=args.eval_batch_size, num_workers=args.num_workers)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    all_eval_guids = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        eval_guids = batch['guid']
        all_eval_guids.extend(eval_guids)
        # batch = tuple(t.to(args.device) for t in batch)
        batch = {k: v.to(args.device) for k, v in batch.items() if k != 'guid'}
        with torch.no_grad():
            inputs = {'input_ids':      batch['input_ids'],
                      'attention_mask': batch['input_mask'],
                      'token_type_ids': batch['segment_ids'],
                      'labels':         batch['ranker_label_ids']}
            if args.model_type == 'hier':
                inputs['hier_mask'] = batch['hier_mask']
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    if args.output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif args.output_mode == "regression":
        preds = np.squeeze(preds)
    elif args.output_mode == "ranking":
        preds = softmax(preds, axis=1)
        preds = np.squeeze(preds[:, 1])    

    result, qrels, run = compute_metrics(eval_task, preds, out_label_ids, guids=all_eval_guids)
    results.update(result)
    eval_output = {'qrels': qrels,
                  'run': run,
                  'ranker_test_all_label_ids': out_label_ids.tolist(),
                  'guids': all_eval_guids,
                  'preds': preds.tolist()}

    # output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    # with open(output_eval_file, "w") as writer:
    logger.info("***** Eval results {} *****".format(prefix))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
        # writer.write("%s = %s\n" % (key, str(result[key])))

    return results, eval_output


# In[6]:


# def main():
parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--data_dir", default='/mnt/scratch/chenqu/aol/preprocessed/', type=str, required=False,
                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--model_type", default='hier', type=str, required=False,
                    help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
parser.add_argument("--model_name_or_path", default='/mnt/scratch/chenqu/huggingface/', type=str, required=False,
                    help="Path to pre-trained model or shortcut name")
parser.add_argument("--task_name", default='stateful_search', type=str, required=False,
                    help="The name of the task to train")
parser.add_argument("--output_dir", default='/mnt/scratch/chenqu/stateful_search/20000/', type=str, required=False,
                    help="The output directory where the model predictions and checkpoints will be written.")

## Other parameters
parser.add_argument("--config_name", default="", type=str,
                    help="Pretrained config name or path if not the same as model_name")
parser.add_argument("--tokenizer_name", default="", type=str,
                    help="Pretrained tokenizer name or path if not the same as model_name")
parser.add_argument("--cache_dir", default="", type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")
parser.add_argument("--max_seq_length", default=128, type=int,
                    help="The maximum total input sequence length after tokenization. Sequences longer "
                         "than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--do_train", default=True, type=str2bool,
                    help="Whether to run training.")
parser.add_argument("--do_eval", default=True, type=str2bool,
                    help="Whether to run eval on the dev set.")
parser.add_argument("--evaluate_during_training", default=True, type=str2bool,
                    help="Run evaluation during training at each logging step.")
parser.add_argument("--do_lower_case", default=True, type=str2bool,
                    help="Set this flag if you are using an uncased model.")

parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--per_gpu_eval_batch_size", default=2, type=int,
                    help="Batch size per GPU/CPU for evaluation.")
parser.add_argument("--per_gpu_test_batch_size", default=2, type=int,
                    help="Batch size per GPU/CPU for testing.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--learning_rate", default=3e-5, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument("--num_train_epochs", default=3.0, type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--max_steps", default=-1, type=int,
                    help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
parser.add_argument("--warmup_steps", default=0, type=int,
                    help="Linear warmup over warmup_steps.")

parser.add_argument('--logging_steps', type=int, default=5,
                    help="Log and save checkpoint every X updates steps.")
parser.add_argument('--save_steps', type=int, default=5000,
                    help="Save checkpoint every X updates steps, this is disabled in our code")
parser.add_argument("--eval_all_checkpoints", default=False, type=str2bool,
                    help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
parser.add_argument("--no_cuda", default=False, type=str2bool,
                    help="Avoid using CUDA when available")
parser.add_argument('--overwrite_output_dir', default=True, type=str2bool,
                    help="Overwrite the content of the output directory")
parser.add_argument('--overwrite_cache', action='store_true',
                    help="Overwrite the cached training and evaluation sets")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")

parser.add_argument('--fp16', default=False, type=str2bool,
                    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
parser.add_argument('--fp16_opt_level', type=str, default='O1',
                    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                         "See details at https://nvidia.github.io/apex/amp.html")
parser.add_argument("--local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")
parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

# parameters we added
parser.add_argument("--include_skipped", default=True, type=str2bool, required=False,
                    help="whether to include the skipped doc from prev turn")
parser.add_argument("--enable_turn_id_embeddings", default=True, type=str2bool, required=False,
                    help="whether to enable turn id embeddings")
parser.add_argument("--enable_component_embeddings", default=True, type=str2bool, required=False,
                    help="whether to enable component embeddings")
parser.add_argument("--load_small", default=False, type=str2bool, required=False,
                    help="whether to just a small portion of data during development")
parser.add_argument("--dataset", default='aol', type=str, required=False,
                    help="aol or msmarco. For bing data, we do not use the first query in a session")
parser.add_argument("--history_num", default=2, type=int, required=False,
                    help="number of history turns to concat")
parser.add_argument("--num_workers", default=2, type=int, required=False,
                    help="number of workers for dataloader")

# args = parser.parse_args()
args, unknown = parser.parse_known_args()

if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
    raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

# Setup distant debugging if needed
if args.server_ip and args.server_port:
    # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
    import ptvsd
    print("Waiting for debugger attach")
    ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
    ptvsd.wait_for_attach()

# Setup CUDA, GPU & distributed training
if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend='nccl')
    args.n_gpu = 1
args.device = device

# Setup logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

# Set seed
set_seed(args)

args.task_name = args.task_name.lower()
# if args.task_name not in processors:
#     raise ValueError("Task not found: %s" % (args.task_name))
# processor = processors[args.task_name]()
args.output_mode = output_modes[args.task_name]
label_list = ["False", "True"]
num_labels = len(label_list)

# Load pretrained model and tokenizer
if args.local_rank not in [-1, 0]:
    torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

args.model_type = args.model_type.lower()
config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
tokenizer.add_tokens(['[EMPTY_QUERY]', '[EMPTY_TITLE]'])
model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

if args.local_rank == 0:
    torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

model.to(args.device)

if args.model_type == 'hier':
    for layer in model.bert.encoder.layer:
        layer.hier.att.attention.load_state_dict(layer.attention.state_dict())
        layer.hier.att.intermediate.load_state_dict(layer.intermediate.state_dict())
        layer.hier.att.output.load_state_dict(layer.output.state_dict())


logger.info("Training/evaluation parameters %s", args)


# Training
if args.do_train:
    # train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
    train_dataset = ConcatModelDataset(os.path.join(args.data_dir, "session_train.txt"), args.include_skipped,
                               args.max_seq_length, tokenizer, args.output_mode, args.load_small, args.dataset,
                                 args.history_num)
    eval_dataset = ConcatModelDataset(os.path.join(args.data_dir, "session_dev_small.txt"), args.include_skipped, 
                                args.max_seq_length, tokenizer, args.output_mode, args.load_small, args.dataset,
                                 args.history_num)
    test_dataset = ConcatModelDataset(os.path.join(args.data_dir, "session_test.txt"), args.include_skipped, 
                                args.max_seq_length, tokenizer, args.output_mode, args.load_small, args.dataset,
                                 args.history_num)
    global_step, tr_loss = train(args, train_dataset, eval_dataset, model, tokenizer)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
#     if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
#         # Create output directory if needed
#         if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
#             os.makedirs(args.output_dir)

#         logger.info("Saving model checkpoint to %s", args.output_dir)
#         # Save a trained model, configuration and tokenizer using `save_pretrained()`.
#         # They can then be reloaded using `from_pretrained()`
#         model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
#         model_to_save.save_pretrained(args.output_dir)
#         tokenizer.save_pretrained(args.output_dir)

#         # Good practice: save your training arguments together with the trained model
#         torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

#         # Load a trained model and vocabulary that you have fine-tuned
#         model = model_class.from_pretrained(args.output_dir)
#         tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
#         model.to(args.device)


# Evaluation on test set
results = {}
if args.do_eval and args.local_rank in [-1, 0]:
    tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    logger.info("Testing")
    model = model_class.from_pretrained(os.path.join(args.output_dir, 'checkpoint'))
    model.to(args.device)
    result, test_output = evaluate(args, test_dataset, model, 
                                   tokenizer, args.per_gpu_test_batch_size, prefix='test')
    result = dict((k + '_{}'.format('test'), v) for k, v in result.items())
    results.update(result)

    output_eval_file = os.path.join(args.output_dir, "test_results.txt")
    with open(output_eval_file, "w") as writer:
        for key in sorted(results.keys()):
            writer.write("%s = %s\n" % (key, str(results[key])))

    output_test_preds_file = os.path.join(args.output_dir, "test_preds.txt")
    with open(output_test_preds_file, 'w') as writer:
        json.dump(test_output, writer)

    # return results


# In[ ]:


# if __name__ == "__main__":
#     main()


# In[ ]:




