# VulATMHD
This paper has been submitted in the journal of Information and Software Technology (Revision).

## Environment Setup
To successfully run the project, the following Python packages need to be installed in a `Python 3.8` environment:
```
torch==2.1.1+rocm5.6
transformers==4.35.2
tokenizers==0.15.0
numpy==1.24.1
tqdm==4.65.0
scikit-learn==1.7.0
pandas==2.2.3
```

## Datasets

Our experiments are conducted on the following three publicly available datasets, for which we provide the corresponding download links or source data in the `data/` directory:

-   **BigVul**: Jiahao Fan, Yi Li, Shaohua Wang, and Tien N Nguyen. A C/C++ code vulnerability dataset with code changes and CVE summaries. In Proceedings of the 17th international conference on mining software repositories, pages 508–512, 2020.
-   **Devign**: Yaqin Zhou, Shangqing Liu, Jingkai Siow, Xiaoning Du, and Yang Liu. Devign: Effective vulnerability identification by learning comprehensive program semantics via graph neural networks. Advances in neural information processing systems, 32, 2019.
-   **Reveal**: Saikat Chakraborty, Rahul Krishna, Yangruibo Ding, and Baishakhi Ray. Deep learning based vulnerability detection: Are we there yet? IEEE Transactions on Software Engineering, 48(9):3280–3296, 2021.

## Execution Steps

The CodeBERT model fine-tuned with VulATMHD can be downloaded from the following link:
```
https://drive.google.com/file/d/1hL7qmmM6rkDObj4SeySi192aa53RR8yn/view?usp=sharing
```
Once downloaded, place the model in the `saved_models/checkpoint-best-acc/` directory and execute the following command for model inference:
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
        
## Contact
If you have any problem about our code, feel free to contact

* 202310188991@mail.scut.edu.cn or describe your problem in Issues.
