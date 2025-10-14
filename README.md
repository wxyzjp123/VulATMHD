# VulATMHD
This paper has been submitted in the journal of Information and Software Technology (Revision).

## Environment Setup
To successfully run the project, the following Python packages need to be installed:
```
torch==2.1.1+rocm5.6
transformers==4.35.2
tokenizers==0.15.0
numpy==1.24.1
tqdm==4.65.0
scikit-learn==1.7.0
pandas==2.2.3
```

## Execution Steps

The CodeBERT model fine-tuned with VulATMHD can be downloaded from the following link:
```
https://drive.google.com/file/d/1hL7qmmM6rkDObj4SeySi192aa53RR8yn/view?usp=sharing
```
Once downloaded, place the model in the `saved_models/checkpoint-best-acc` directory and execute the following command for model inference:
```
python codebert_main.py \
    --output_dir=./saved_models \
    --model_name=soft_distil_model_07.bin \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --train_data_file=../data/big_vul/train.csv \
    --eval_data_file=../data/big_vul/val.csv \
    --test_data_file=../data/big_vul/test.csv \
    --do_test \
    --block_size 512 \
    --epochs 50 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee test.log
```
To retrain the model, you only need to execute the following two steps:

**Step 1: Feature Learning**

    python feature_main.py \
        --output_dir=./saved_models \
        --model_name=cnnteacher.bin \
        --tokenizer_name=microsoft/codebert-base \
        --model_name_or_path=microsoft/codebert-base \
        --train_data_file=../data/big_vul/train.csv \
        --eval_data_file=../data/big_vul/val.csv \
        --test_data_file=../data/big_vul/test.csv \
        --do_train \
        --do_test \
        --block_size 512 \
        --epochs 50 \
        --train_batch_size 128 \
        --eval_batch_size 128 \
        --learning_rate 5e-3 \
        --max_grad_norm 1.0 \
        --evaluate_during_training \
        --seed 123456  2>&1 | tee train_feature.log

**Step 2: Hybrid Knowledge Distillation**

    python codebert_main.py \
        --output_dir=./saved_models \
        --model_name=model.bin \
        --tokenizer_name=microsoft/codebert-base \
        --model_name_or_path=microsoft/codebert-base \
        --train_data_file=../data/big_vul/train.csv \
        --eval_data_file=../data/big_vul/val.csv \
        --test_data_file=../data/big_vul/test.csv \
        --do_train \
        --do_test \
        --block_size 512 \
        --epochs 50 \
        --train_batch_size 8 \
        --eval_batch_size 8 \
        --learning_rate 2e-5 \
        --max_grad_norm 1.0 \
        --evaluate_during_training \
        --seed 123456  2>&1 | tee train_codebert.log
