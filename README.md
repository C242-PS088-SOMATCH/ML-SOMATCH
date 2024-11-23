### How to run

```
python train.py `
    --input_file_pattern="Your-Path/data/tf_records/train-no-dup-*" `
    --inception_checkpoint_file="checkpoints/inception_v3/inception_v3.ckpt" `
    --train_dir="checkpoints/polyvore_model" `
    --train_inception `
    --number_of_steps=2000 `
    --log_every_n_steps=10
```
