# Training, Evaluation and Inference

## Dataset generation
First we need to generate the dataset, please follow this guide:

[How to generate the dataset](dataset.md)

## Model training
```bash
$ python train.py  --train_data .cache/train.pkl \
                   --val_data .cache/val.pkl \
                   --model_name cnn \
                   --model_dir checkpoints \
                   --initial_learning_rate 0.0003 \
                   --epochs 150 \
                   --folds 1 \
                   --dropout 0.00001 \
                   --verified_only \
                   --balance_class \
                   --binary_class \
                   --early_stopping \
                   --early_stopping_patience 10 \
                   --reduce_lr_plateau \
                   --reduce_lr_plateau_factor 0.75 \
                   --reduce_lr_plateau_patience 2
```
`model_name` may also have a prefix of `resnet` or `mobilenet`.
We may also specify `arch_index` to use pre-defined seed design in `Covidnet_assistant/models/utils.py`.
For more options and information, `python train.py --help`

## Evaluation
```bash
$ python eval.py --model_path checkpoints/cnn-0.h5 \
                 --test_data .cache/test.pkl \
                 --batch_size 64 \
                 --binary_class
```
For more options and information, `python eval.py --help`

## Inference
```bash
$python inference.py --model checkpoints/cnn-0.h5 \
                     --audio audio.mp3 \
                     --binary_class \
                     --model_type h5
```
For more options and information, `python inference.py --help`
