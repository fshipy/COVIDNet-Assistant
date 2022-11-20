# Covid19-Cough Dataset

## Description
The dataset came from https://github.com/covid19-cough/dataset

This Covid19-Cough dataset consists of 1324 raw audio samples with 682 cases labeled positive, and 382 of them are confirmed by PCR test. For the negative labeled audio pieces, no verification information is provided. The audio were collected through a call center or telegram. We excluded two broken audio files in our study, leaving us with 1322 pieces of audio, with 681 labeled positive and 381 positive samples verified.

Please refer to [the repository](https://github.com/covid19-cough/dataset) for detailed description

## Generate dataset
### Step 1: Download dataset
```bash
$ git clone https://github.com/covid19-cough/dataset.git
```

### Step 2: Delete bad files

These two pieces are malformed.
```bash
$ rm dataset/raw/098d66e5-bda6-4e99-b787-ab890046c44b.mp3
$ rm dataset/raw/a9ecaf03-40a5-4b43-aaf3-f076f84a69aa.mp3
```

### Step 3: Data preprocessing

We use the dataset in 2 settings, the all data setting and the verified only setting.

1. All data

Use every audio piece except the above two bad audio pieces
```bash
$ python data/process.py --split_file split.json \
                         --cache_dir  .cache \
                         --data_dir ./dataset \
                         --parallelism 16 \
                         --train_size 0.6 \
                         --val_size 0.2 \
                         --seed 42 \
                         --output_train .cache/train.pkl \
                         --output_val .cache/val.pkl \
                         --output_test .cache/test.pkl
```

2. Verified only

In verified only setting, we only use the audio pieces associated with a verified
covid test result OR negative test result. \
The audio pieces associated with negative results are included becasue
in the database, all verified pieces are associated with a positive result.
```bash
$ python data/process.py --split_file split_verified.json \
                         --cache_dir  .cache \
                         --data_dir ./dataset \
                         --parallelism 16 \
                         --train_size 0.8 \
                         --seed 42 \
                         --output_train .cache/train.pkl \
                         --output_val .cache/val.pkl \
                         --output_test .cache/test.pkl \
                         --verified_only
```

3. Augmentation

To generate a prcoessed dataset with augmentation, add the following options to the execution command.
In our experiemnts, we use `aug_strength=50` and `aug_per_example=5`
```bash
--use_augmentation \
--augments_per_example <aug_per_example> \
--augment_strength <aug_strength> \
```
