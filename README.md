# Semantic Textual Similarity Benchmark

```
python finetune.py --model_path scratch/finetune_bert_drop_focused_best --train --batch_size 32 --test_batch_size 64 --epochs 3 --num_gradient_accumulations 1 --check_interval 1000 --num_bad_epochs 6 --num_workers 8 --seed 55482 --data_path STS-B --model_type bert --joint --pooling cls --use_projection --lr 5e-05 --drop 0.1 --clip 1 --gpu 7


109483009 params
hparams:  --train --batch_size 32 --test_batch_size 64 --epochs 3 --num_gradient_accumulations 1 --check_interval 1000 --num_bad_epochs 6 --num_workers 8 --seed 55482 --data_path STS-B --model_type bert --joint --pooling cls --use_projection --lr 5e-05 --drop 0.1 --clip 1.0
End of epoch   1 | loss     1.56 | val perf    85.43		*Best model so far, deep copying*
End of epoch   2 | loss     1.05 | val perf    87.12		*Best model so far, deep copying*
End of epoch   3 | loss     0.80 | val perf    87.79		*Best model so far, deep copying*
----New best    87.79, saving
Time: 0:01:56
Val:     87.81
Test:    84.02
```

```
python finetune.py --model_path scratch/finetune_roberta_drop_focused_best --train --batch_size 32 --test_batch_size 64 --epochs 3 --num_gradient_accumulations 1 --check_interval 1000 --num_bad_epochs 6 --num_workers 8 --seed 67294 --data_path STS-B --model_type roberta --joint --pooling avg --use_projection --lr 5e-05 --drop 0.1 --clip 1 --gpu 6

124646401 params
hparams:  --train --batch_size 32 --test_batch_size 64 --epochs 3 --num_gradient_accumulations 1 --check_interval 1000 --num_bad_epochs 6 --num_workers 8 --seed 67294 --data_path STS-B --model_type roberta --joint --pooling avg --use_projection --lr 5e-05 --drop 0.1 --clip 1.0
End of epoch   1 | loss     1.38 | val perf    88.54		*Best model so far, deep copying*
End of epoch   2 | loss     0.95 | val perf    88.36		Bad epoch 1
End of epoch   3 | loss     0.74 | val perf    90.43		*Best model so far, deep copying*
----New best    90.43, saving
Time: 0:01:55
Val:     90.42
Test:    87.63
```

```
python finetune.py --train --data_path data_toy --model_type bert --num_runs 3

python finetune.py --train --data_path data_toy --batch_size 3 --joint --use_projection --lr 3e-5 --num_bad_epochs 100 --epochs 100

pip install -r requirements.txt
```
