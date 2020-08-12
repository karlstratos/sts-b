# Semantic Textual Similarity Benchmark (STS-B)

This is a self-contained codebase for STS-B evaluation. 

## Setup

```
pip install -r requirements.txt
```

## Highlights

1. It can roughly replicate reported single-task single-model fine-tuning results. For instance, 
    ```
    python finetune.py --model_path scratch/finetune_bert-large_best --train --model_type bert-large --seed 45847 --optimize bert --lr 5e-05 --num_workers 4 --gpu 2  # 87.7 
    python finetune.py --model_path scratch/finetune_electra-large_best --train --batch_size 16 --model_type electra-large --seed 84899 --lr 3e-05 --num_workers 4 --gpu 0  # 91.5
    ```
    are on par with BERT (87.6) on the [GLUE leaderboard](https://gluebenchmark.com/leaderboard) and ELECTRA (91.7) in Table 8 of the [paper](https://arxiv.org/pdf/2003.10555.pdf). But most reported STS results use model ensembling and "intermediate-task" training (fine-tuning from an MNLI checkpoint) and thus not directly comparable (e.g., we can get up to 91.5 with RoBERTa which is still a bit behind the reported 92.2 on the leaderboard).  
    
2. It can train a generic encoder (for now just a BiLSTM-based dual encoder following GLUE) on top of frozen representations of the data to assess their quality. The representations are lists of word embeddings in a pickle file: see the `dump` function in `evaluate_word_embeddings.py` and `finetune.py`. For instance, `python evaluate_word_embeddings.py --word_embeddings [glove_path] --dump_path scratch/dump_word_embeddings_glove` dumps GloVe embeddings and `python finetune.py --dump_path scratch/dump_bert --model_type bert --batch_size 256 --gpu 0` dumps output embeddings of BERT. Then we can fit a regressor on top of these embeddings by 
    ```
    python frozen.py --model_path scratch/frozen_glove --train --dump_path scratch/dump_word_embeddings_glove --verbose --num_runs 100 --num_workers 4 --gpu 5  # 72.8
    python frozen.py --model_path scratch/frozen_bert --train --dump_path scratch/dump_bert --verbose --num_runs 100 --num_workers 4 --gpu 4  # 81.9    
    python frozen.py --model_path scratch/frozen_bert_disjoint --train --dump_path scratch/dump_bert_disjoint --verbose --num_runs 100 --num_workers 4 --gpu 3  # 72.2
    ```
    The last command uses BERT embeddings without cross attention between sentences (can be done by passing `--disjoint`). It is clear that cross attention is crucial. 
    
3. It offers various unsupervised/lightly supervised options for evaluation. Word embeddings can be evaluated by simple mean/max pooling, optionally with smooth inverse frequency (SIF) reweighting and subspace cleaning described in [Arora et al. (2017)](https://openreview.net/pdf?id=SyK00v5xx). For instance with GloVe
    ```
    python evaluate_word_embeddings.py --word_embeddings [glove_path]  # 36.3
    python evaluate_word_embeddings.py --word_embeddings [glove_path] --dim_subspace 17 --pca --freq [freq_file]  # 69.6    
    ```
    More generally, we can apply the same unsupervised/lightly supervised evaluation to any frozen representations of the data using `frozen_raw.py` (e.g., `python frozen_raw.py --dump_path scratch/dump_roberta`). [Sent2vec](https://github.com/epfml/sent2vec), which seems to be a strongest unsupervised baseline, can be evaluated by `python evaluate_sent2vec.py --model ../sent2vec/twitter_unigrams.bin` (75.5).   
