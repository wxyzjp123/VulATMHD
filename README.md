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

**Step 1: Feature Learning**

    python feature_main.py \
        --output_dir=./saved_models \
        --model_name=cnnteacher.bin \
        --tokenizer_name=codebert \
        --model_name_or_path=codebert \
        --train_data_file=train.csv \
        --eval_data_file=val.csv \
        --test_data_file=test.csv \
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
        --alpha 0.7 \
        --output_dir=./saved_models \
        --model_name=model.bin \
        --tokenizer_name=codebert \
        --model_name_or_path=codebert \
        --train_data_file=train.csv \
        --eval_data_file=val.csv \
        --test_data_file=test.csv \
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
