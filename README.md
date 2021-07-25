# Contextual Re-Ranking with Behavior Aware Transformers

Code for our paper [Contextual Re-Ranking with Behavior Aware Transformers](http://ciir-publications.cs.umass.edu/getpdf.php?id=1383).

## Usage
* `unzip pytrec_eval.zip` in the code directory. 
* Follow the sample command:  
```
python3 -u -m torch.distributed.launch --nproc_per_node 8 run.py \
  --data_dir=DATA_DIR \
  --model_name_or_path=PATH_TO_HUGGINGFACE_MODELS (OPTIONAL) \
  --output_dir=OUTPUT_DIR \
  --max_seq_length=128 \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=16 \
  --per_gpu_test_batch_size=16 \
  --learning_rate=1e-4 \
  --num_train_epochs=10 \
  --logging_steps=5 \
  --save_steps=5000 \
  --include_skipped=False \
  --include_clicked=True \
  --load_small=False \
  --dataset=aol \
  --history_num=3 \
  --num_workers=2 \
  --model_type=ba_hier_att \
  --fp16=True \
  --overwrite_output_dir=False \
  --warmup_portion=0.1 \
  --do_train=True \
  --gradient_accumulation_steps=4 \
  --enable_behavior_rel_pos_embeddings=True  \
  --enable_regular_pos_embeddings_in_sess_att=False \
  --enable_behavior_type_embeddings=True \
  --intra_att=True \
  --num_inter_att_layers=1
```
* Sample output is available in `slurm-6568727.out`.

## Input format
The input files are in json lines.
```
{
  "query": QUERY,
  "title": DOCUMENT_TITLE,
  "history": 
    [
      [HISTORY_QUERY_1, HISTORY_CLICK_1, HISTORY_SKIP_1], 
      [HISTORY_QUERY_2, HISTORY_CLICK_2, HISTORY_SKIP_2],
      ...
    ],
  "guid": INSTANCE_ID,
  "label": true or false
}
```
See `utils.py` for more details.

## Dependencies
* pytorch-transformers 1.2

## Acknowledgement
* Our code is based on [Pytorch-transformers](https://huggingface.co/transformers/v1.2.0/installation.html), [BERT](https://github.com/google-research/bert), and [pytrec_eval](https://github.com/cvangysel/pytrec_eval). We thank the authors for building these libararies.  
* Our data came from [Context Attentive Document Ranking and Query Suggestion](https://arxiv.org/pdf/1906.02329.pdf). We thank the authors for sharing their data.  
