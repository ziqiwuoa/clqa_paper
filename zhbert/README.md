# zhbert
BERT model in Chinese

The project is based on the blog [https://huggingface.co/blog/how-to-train](https://huggingface.co/blog/how-to-train). We used
the latest code base from huggingface which is `run_mlm.py` from [transformers examples](https://github.com/huggingface/transformers/tree/master/examples/pytorch/language-modeling). Huggingface transformers version
4.13.0 dev and above is required. We couldn't manage to install it by using `pip install transformers` and used the script
below instead. 

Huggingface accelerate config file location: `/home/zwan499/.cache/huggingface/accelerate/default_config.yaml`

Installing huggingface transformers 4.13.0 dev0:
```shell
pip install git+https://github.com/huggingface/transformers
```

Training RoBERTa from scratch using ultimate translation corpus
```shell
CUDA_VISIBLE_DEVICES=6 python run_mlm.py \
--output_dir ./models/zhberta-v1 \
--validation_file ./data/valid.raw.zh.txt \
--train_file ./data/train.raw.zh.txt \
--config_name ./models/roberta-base \
--tokenizer_name ./tokenizers/bpe52k-zh \
--model_type roberta \
--do_train \
--do_eval \
--num_train_epochs 8 \
--save_steps 10000 \
--per_device_train_batch_size 32 \
--learning_rate 1e-4 \
--line_by_line \
--seed 42
```

Evaluation script using run_c3.py
```shell
export MODEL_NAME=c3-ft-chinese_roberta_L-12_H-768_notrainer_lr3e05_ep20/best_model
export OUTPUT_NAME=test_c3-ft-chinese_roberta_L-12_H-768_notrainer_lr3e05_ep20
CUDA_VISIBLE_DEVICES=2,3,4,6 python run_c3.py \
--output_dir ./models/$OUTPUT_NAME \
--validation_file ./data/c3/c3-all-hf-test-zh.json \
--train_file ./data/c3/c3-all-hf-train-zh.json \
--model_name_or_path ./models/$MODEL_NAME \
--do_eval \
--save_steps 10000 \
--per_device_eval_batch_size=3 \
--learning_rate 1e-5 \
--seed 42 | tee -a ./models/$OUTPUT_NAME.log
```


Evaluation using run_eval.py
- using "--uncased" argument for language models that only supports lowercase inputs
```shell
export MODEL_NAME=bert-base-uncased/dnr_c3_v6_ultimate_lr1e05_ep30
CUDA_VISIBLE_DEVICES="6" python run_eval.py \
--validation_file ./data/c3/c3-all-hf-test-v6-ultimate_v2-bpe20k-en.json \
--model_name_or_path ./models-ft/$MODEL_NAME \
--max_length 512 \
--per_device_eval_batch_size=12 \
--uncased \
--task_name race
```


Evaluation using run_eval.py
- using "--uncased" argument for language models that only supports lowercase inputs
```shell
for SPLIT in v5-iwslt17 v5-um-education v5-um-laws v5-um-news v5-um-science v5-um-spoken v5-um-thesis v5-un-bpe20k v5-ultimate_v2-bpe20k v5-umeln-bpe20k v6-un-bpe20k v6-ultimate_v2-bpe20k v6-umeln-bpe20k
do
MODEL_NAME=bert-base-uncased/dream_lr2e05_ep30
CUDA_VISIBLE_DEVICES="2" python run_eval.py \
--validation_file ./data/c3/c3-all-hf-test-$SPLIT-en.json \
--model_name_or_path ./models-ft/$MODEL_NAME \
--max_length 512 \
--per_device_eval_batch_size=12 \
--uncased \
--task_name race | tee -a ./models-ft/eval_log_20220413.log
echo "@@ Evaluation done for [$MODEL_NAME] with split [$SPLIT]" | tee -a ./models-ft/eval_log_20220413.log
done

for SPLIT in 0p5 1 2 3 4 5 10 20 40 60 80 100
do
MODEL_NAME=bert-base-uncased/dream_lr2e05_ep30
CUDA_VISIBLE_DEVICES="2" python run_eval.py \
--validation_file ./data/c3/c3-all-hf-test-umeln-$SPLIT-en.json \
--model_name_or_path ./models-ft/$MODEL_NAME \
--max_length 512 \
--per_device_eval_batch_size=12 \
--uncased \
--task_name race | tee -a ./models-ft/eval_log_20220413.log
echo "@@ Evaluation done for [$MODEL_NAME] with split [$SPLIT]" | tee -a ./models-ft/eval_log_20220413.log
done
```


Evaluation using run_eval.py
- using "--uncased" argument for language models that only supports lowercase inputs
```shell
export MODEL_NAME=bert-base-chinese/lr2e05_ep20/best_model
export OUTPUT_NAME=bert-base-chinese_lr2e05_ep20
for SEED in 42 54 94 183 283 87 291 549 12 9
do
echo "@@ Evaluating [$MODEL_NAME] with seed: $SEED" | tee -a ./models-ft/$MODEL_NAME/$OUTPUT_NAME.log
CUDA_VISIBLE_DEVICES="5" python run_eval.py \
--validation_file ./data/c3/c3-all-hf-test-zh.json \
--model_name_or_path ./models-ft/$MODEL_NAME \
--max_length 512 \
--per_device_eval_batch_size=3 \
--task_name c3 \
--seed $SEED | tee -a ./models-ft/$MODEL_NAME/$OUTPUT_NAME.log
done
```


Convert RACE dataset to Huggingface version json file
```shell
python3 utils/race_to_hfjson.py \
--input-dir ./data/RACE \
--output-dir ./data/RACE/hfjson
```


Fine-tuning RoBERTa chinese_roberta_L-12_H-768 on c3 dataset
** Fine-tuning RoBERTa takes 2 times GPU's memory comparing to BERT, 12 samples occupy around 40gbs of RAM
accuracy: 44.16
```shell
CUDA_VISIBLE_DEVICES=5,6 python run_c3.py \
--output_dir ./models/c3-ft-chinese_roberta_L-12_H-768 \
--validation_file ./data/c3/c3-all-hf-dev-zh.json \
--train_file ./data/c3/c3-all-hf-train-zh.json \
--model_name_or_path ./models/chinese_roberta_L-12_H-768 \
--do_train \
--do_eval \
--num_train_epochs 8 \
--save_steps 10000 \
--per_device_train_batch_size 12 \
--learning_rate 2e-5 \
--seed 42 | tee -a ./models/c3-ft-chinese_roberta_L-12_H-768_train.log
```


20220109: fine-tune model with lr=1e-5 (from RoBERTa paper para5.1), with 24 batch size, also with increased 
evaluation size from 16 to 24 (same as training batch size)
accuracy: 45.13%
```shell
CUDA_VISIBLE_DEVICES=1,2,3,4 python run_c3.py \
--output_dir ./models/c3-ft-chinese_roberta_L-12_H-768_lr1e05 \
--validation_file ./data/c3/c3-all-hf-dev-zh.json \
--train_file ./data/c3/c3-all-hf-train-zh.json \
--model_name_or_path ./models/chinese_roberta_L-12_H-768 \
--do_train \
--do_eval \
--num_train_epochs 8 \
--save_steps 10000 \
--per_device_train_batch_size 3 \
--per_device_eval_batch_size=3 \
--gradient_accumulation_steps 2 \
--learning_rate 1e-5 \
--seed 42 | tee -a ./models/c3-ft-chinese_roberta_L-12_H-768_train_lr1e05.log
```


20220110-01: fine-tune model with lr=3e-5 (from RoBERTa paper para5.1), with 24 batch size, also with increased 
evaluation size from 16 to 24 (same as training batch size)
accuracy: 46.46%
```shell
CUDA_VISIBLE_DEVICES=1,2,3,4 python run_c3.py \
--output_dir ./models/c3-ft-chinese_roberta_L-12_H-768_lr3e05 \
--validation_file ./data/c3/c3-all-hf-dev-zh.json \
--train_file ./data/c3/c3-all-hf-train-zh.json \
--model_name_or_path ./models/chinese_roberta_L-12_H-768 \
--do_train \
--do_eval \
--num_train_epochs 8 \
--save_steps 10000 \
--per_device_train_batch_size 3 \
--per_device_eval_batch_size=3 \
--gradient_accumulation_steps 2 \
--learning_rate 3e-5 \
--seed 42 | tee -a ./models/c3-ft-chinese_roberta_L-12_H-768_train_lr3e05.log
```


20220110-02: fine-tune model with lr=1.5e-5 (from RoBERTa paper table 10 in the appendix), with 24 batch size, also with increased 
evaluation size from 16 to 24 (same as training batch size)
accuracy: 45%
```shell
CUDA_VISIBLE_DEVICES=1,2,3,4 python run_c3.py \
--output_dir ./models/c3-ft-chinese_roberta_L-12_H-768_lr1pt5e05 \
--validation_file ./data/c3/c3-all-hf-dev-zh.json \
--train_file ./data/c3/c3-all-hf-train-zh.json \
--model_name_or_path ./models/chinese_roberta_L-12_H-768 \
--do_train \
--do_eval \
--num_train_epochs 8 \
--save_steps 10000 \
--per_device_train_batch_size 3 \
--per_device_eval_batch_size=3 \
--gradient_accumulation_steps 2 \
--learning_rate 1.5e-5 \
--seed 42 | tee -a ./models/c3-ft-chinese_roberta_L-12_H-768_train_lr1pt5e05.log
```


20220110: Evaluation only (UER RoBERTa model without fine-tuning) accuracy: 26%
```shell
export MODEL_NAME=c3-ft-chinese_roberta_L-12_H-768_lr3e05_ep20_bz72_warmup_wdecay/checkpoint-1000
export OUTPUT_NAME=c3-ft-chinese_roberta_L-12_H-768_eval_only_lr3e05_ep20_bz72_warmup_wdecay_cpt1000
CUDA_VISIBLE_DEVICES=1,2,3,4 python run_c3.py \
--output_dir ./models/$OUTPUT_NAME \
--validation_file ./data/c3/c3-all-hf-dev-zh.json \
--train_file ./data/c3/c3-all-hf-train-zh.json \
--model_name_or_path ./models/$MODEL_NAME \
--do_eval \
--save_steps 10000 \
--per_device_eval_batch_size=3 \
--learning_rate 1e-5 \
--seed 42 | tee -a ./models/$OUTPUT_NAME.log
```


20220111-01: fine-tune model with lr=1e-4 (from RoBERTa paper table 10 in the appendix), with 24 batch size, also with increased 
evaluation size from 16 to 24 (same as training batch size)
accuracy: 23.22%
```shell
CUDA_VISIBLE_DEVICES=1,2,3,4 python run_c3.py \
--output_dir ./models/c3-ft-chinese_roberta_L-12_H-768_lr1e04 \
--validation_file ./data/c3/c3-all-hf-dev-zh.json \
--train_file ./data/c3/c3-all-hf-train-zh.json \
--model_name_or_path ./models/chinese_roberta_L-12_H-768 \
--do_train \
--do_eval \
--num_train_epochs 8 \
--save_steps 10000 \
--per_device_train_batch_size 3 \
--per_device_eval_batch_size=3 \
--gradient_accumulation_steps 2 \
--learning_rate 1e-4 \
--seed 42 | tee -a ./models/c3-ft-chinese_roberta_L-12_H-768_train_lr1e04.log
```


20220111-02: learning rate of 3e-05 seems converging faster than other two learning rates, so we want to fine-tune the model
with this learning rate with more epochs, and see when the model fully converges.

we want to train with 24 epochs first, which takes a night, because we don't have the mechanism to save the best model yet.

accuracy was: 48.06%
```shell
CUDA_VISIBLE_DEVICES=1,2,3,4 python run_c3.py \
--output_dir ./models/c3-ft-chinese_roberta_L-12_H-768_lr3e05_ep25 \
--validation_file ./data/c3/c3-all-hf-dev-zh.json \
--train_file ./data/c3/c3-all-hf-train-zh.json \
--model_name_or_path ./models/chinese_roberta_L-12_H-768 \
--do_train \
--do_eval \
--num_train_epochs 25 \
--save_steps 10000 \
--per_device_train_batch_size 3 \
--per_device_eval_batch_size=3 \
--gradient_accumulation_steps 2 \
--learning_rate 3e-5 \
--seed 42 | tee -a ./models/c3-ft-chinese_roberta_L-12_H-768_train_lr3e05_ep25.log
```


20220112-01: learning rate of 3e-05 seems converging faster than other two learning rates, so we want to fine-tune the model
with this learning rate with more epochs, and see when the model fully converges.

we have modified the code, the best model will be saved if it achieves the best accuracy.
we also extracted a partition of the training set for evaluating and recording the training accuracy along the way 
as an alternative metrics for deciding if the model is fully converged.

** we need to run evaluation with the true evaluation set as we pass in a partition of training set for training.

accuracy was: 39.15%, using training set evaluation accuracy as the metric for best model is not a good idea. 
```shell
export MODEL_NAME=c3-ft-chinese_roberta_L-12_H-768_lr3e05_ep50
CUDA_VISIBLE_DEVICES=1,2,3,4 python run_c3.py \
--output_dir ./models/$MODEL_NAME \
--validation_file ./data/c3/c3-all-hf-train-dev-zh.json \
--train_file ./data/c3/c3-all-hf-train-zh.json \
--model_name_or_path ./models/chinese_roberta_L-12_H-768 \
--do_train \
--do_eval \
--num_train_epochs 1 \
--save_steps 10000 \
--per_device_train_batch_size 3 \
--per_device_eval_batch_size=3 \
--gradient_accumulation_steps 2 \
--learning_rate 3e-5 \
--ziqi_num_epochs 50 \
--ziqi_best_output_dir ./models/$MODEL_NAME/best_model \
--seed 42 | tee -a ./models/$MODEL_NAME.log
```


20220113-01: learning rate of 3e-05 seems converging faster than other two learning rates, so we want to fine-tune the model
with this learning rate with more epochs, and see when the model fully converges.

we have modified the code, the best model will be saved if it achieves the best accuracy.
we use the original dev set this time for recording the accuracy during training, and save the best model based on it

** we need to run evaluation with the true evaluation set as we pass in a partition of training set for training.

accuracy was: 52.78% 
```shell
export MODEL_NAME=c3-ft-chinese_roberta_L-12_H-768_lr3e05_ep50_dev
CUDA_VISIBLE_DEVICES=1,2,3,4 python run_c3.py \
--output_dir ./models/$MODEL_NAME \
--validation_file ./data/c3/c3-all-hf-dev-zh.json \
--train_file ./data/c3/c3-all-hf-train-zh.json \
--model_name_or_path ./models/chinese_roberta_L-12_H-768 \
--do_train \
--do_eval \
--num_train_epochs 1 \
--save_steps 10000 \
--per_device_train_batch_size 3 \
--per_device_eval_batch_size=3 \
--gradient_accumulation_steps 2 \
--learning_rate 3e-5 \
--ziqi_num_epochs 50 \
--ziqi_best_output_dir ./models/$MODEL_NAME/best_model \
--seed 42 | tee -a ./models/$MODEL_NAME.log
```


20220113-01: learning rate of 3e-05 seems converging faster than other two learning rates, so we want to fine-tune the model
with this learning rate with more epochs, and see when the model fully converges.

we have modified the code, the best model will be saved if it achieves the best accuracy.
we use the original dev set this time for recording the accuracy during training, and save the best model based on it

** we need to run evaluation with the true evaluation set as we pass in a partition of training set for training.

** we updated the converted dataset structure to the original one, and the accuracy improved a bit, we want to try a 
different learning rate and see if it can get better.

accuracy was: 52.78% 
```shell
export MODEL_NAME=c3-ft-chinese_roberta_L-12_H-768_lr3e05_ep50_dev
CUDA_VISIBLE_DEVICES=1,2,3,4 python run_c3.py \
--output_dir ./models/$MODEL_NAME \
--validation_file ./data/c3/c3-all-hf-dev-zh.json \
--train_file ./data/c3/c3-all-hf-train-zh.json \
--model_name_or_path ./models/chinese_roberta_L-12_H-768 \
--do_train \
--do_eval \
--num_train_epochs 1 \
--save_steps 10000 \
--per_device_train_batch_size 3 \
--per_device_eval_batch_size=3 \
--gradient_accumulation_steps 2 \
--learning_rate 3e-5 \
--ziqi_num_epochs 50 \
--ziqi_best_output_dir ./models/$MODEL_NAME/best_model \
--seed 42 | tee -a ./models/$MODEL_NAME.log
```


20220114-01: we use 1e-05 this time to see if the accuracy can get better.

accuracy was: ~52% 
```shell
export MODEL_NAME=c3-ft-chinese_roberta_L-12_H-768_lr1e05_ep50_dev
CUDA_VISIBLE_DEVICES=1,2,3,4 python run_c3.py \
--output_dir ./models/$MODEL_NAME \
--validation_file ./data/c3/c3-all-hf-dev-zh.json \
--train_file ./data/c3/c3-all-hf-train-zh.json \
--model_name_or_path ./models/chinese_roberta_L-12_H-768 \
--do_train \
--do_eval \
--num_train_epochs 1 \
--save_steps 10000 \
--per_device_train_batch_size 3 \
--per_device_eval_batch_size=3 \
--gradient_accumulation_steps 2 \
--learning_rate 1e-5 \
--ziqi_num_epochs 50 \
--ziqi_best_output_dir ./models/$MODEL_NAME/best_model \
--seed 42 | tee -a ./models/$MODEL_NAME.log
```


20220116-01: Facebook example fine-tune RoBERTa with ' ' between paragraphs instead of '\n', we want to try it with our
current UER RoBERTa model and see if it makes any difference.

** Result: we only fine-tuned with 10 epochs, but the accuracy is pretty much the same

accuracy was: ~50% 
```shell
export MODEL_NAME=c3-ft-chinese_roberta_L-12_H-768_lr1e05_ep10_emptyline
CUDA_VISIBLE_DEVICES=1,2,3,4 python run_c3.py \
--output_dir ./models/$MODEL_NAME \
--validation_file ./data/c3/c3-all-hf-dev-zh.json \
--train_file ./data/c3/c3-all-hf-train-zh.json \
--model_name_or_path ./models/chinese_roberta_L-12_H-768 \
--do_train \
--do_eval \
--num_train_epochs 1 \
--save_steps 10000 \
--per_device_train_batch_size 3 \
--per_device_eval_batch_size=3 \
--gradient_accumulation_steps 2 \
--learning_rate 1e-5 \
--ziqi_num_epochs 10 \
--ziqi_best_output_dir ./models/$MODEL_NAME/best_model \
--seed 42 | tee -a ./models/$MODEL_NAME.log
```


20220117-01: We just realise that the BERT version used batch size of 24, but gradient accumulated for 3 steps, which is
equal to 72, so we want to try the same batch size for RoBERTa.

** Result: The accuracy didn't grow to 60% as expected.

accuracy was: ~49.68% 
```shell
export MODEL_NAME=c3-ft-chinese_roberta_L-12_H-768_lr1e05_ep10_bz72
CUDA_VISIBLE_DEVICES=1,2,3,4 python run_c3.py \
--output_dir ./models/$MODEL_NAME \
--validation_file ./data/c3/c3-all-hf-dev-zh.json \
--train_file ./data/c3/c3-all-hf-train-zh.json \
--model_name_or_path ./models/chinese_roberta_L-12_H-768 \
--do_train \
--do_eval \
--num_train_epochs 1 \
--save_steps 10000 \
--per_device_train_batch_size 3 \
--per_device_eval_batch_size=3 \
--gradient_accumulation_steps 6 \
--learning_rate 1e-5 \
--ziqi_num_epochs 10 \
--ziqi_best_output_dir ./models/$MODEL_NAME/best_model \
--seed 42 | tee -a ./models/$MODEL_NAME.log
```


20220117-02: The performance didn't grow to 60% as expected, try remove our customised loop and put in "warmup_ratio"
and "weight_decay". Changed learning rate to 3e-05.

** Result: Last saved model had the best accuracy of 58.54%, but we want to save more model and changed the learning 
rate to see if we can achieve better performance.

accuracy was: 58.54%
```shell
export MODEL_NAME=c3-ft-chinese_roberta_L-12_H-768_lr3e05_ep20_bz72_warmup_wdecay
CUDA_VISIBLE_DEVICES=1,2,3,4 python run_c3.py \
--output_dir ./models/$MODEL_NAME \
--validation_file ./data/c3/c3-all-hf-dev-zh.json \
--train_file ./data/c3/c3-all-hf-train-zh.json \
--model_name_or_path ./models/chinese_roberta_L-12_H-768 \
--do_train \
--do_eval \
--num_train_epochs 20 \
--save_steps 1000 \
--per_device_train_batch_size 3 \
--per_device_eval_batch_size=3 \
--gradient_accumulation_steps 6 \
--learning_rate 3e-5 \
--warmup_ratio 0.1 \
--weight_decay 0.01 \
--seed 42 | tee -a ./models/$MODEL_NAME.log
```


20220118-01: The performance by introducing weight_decay and warmup boost the accuracy to ~58%, but still significant 
worse than 65%. we want to try a different learning rate with more saved checkpoints.

** Result: 

accuracy was: ??% 
```shell
export MODEL_NAME=c3-ft-chinese_roberta_L-12_H-768_lr2e05_ep20_bz72_warmup_wdecay
CUDA_VISIBLE_DEVICES=1,2,3,4 python run_c3.py \
--output_dir ./models/$MODEL_NAME \
--validation_file ./data/c3/c3-all-hf-dev-zh.json \
--train_file ./data/c3/c3-all-hf-train-zh.json \
--model_name_or_path ./models/chinese_roberta_L-12_H-768 \
--do_train \
--do_eval \
--num_train_epochs 20 \
--save_steps 200 \
--per_device_train_batch_size 3 \
--per_device_eval_batch_size=3 \
--gradient_accumulation_steps 6 \
--learning_rate 2e-5 \
--warmup_ratio 0.1 \
--weight_decay 0.01 \
--seed 42 | tee -a ./models/$MODEL_NAME.log
```


20220118-02: run classifier without using Huggingface trainer, this enalbes us to customise the traning process as we 
needed. The script down here if for debugging.
```shell
export MODEL_NAME=c3-ft-chinese_roberta_L-12_H-768_lr2e05_notrainer_debug
CUDA_VISIBLE_DEVICES="2,3,4,6" accelerate launch run_c3_no_trainer.py \
--output_dir ./models/$MODEL_NAME \
--validation_file ./data/c3/c3-all-hf-dev-zh.json \
--train_file ./data/c3/c3-all-hf-train-zh.json \
--model_name_or_path ./models/chinese_roberta_L-12_H-768 \
--num_train_epochs 20 \
--per_device_train_batch_size 3 \
--per_device_eval_batch_size=3 \
--gradient_accumulation_steps 6 \
--learning_rate 2e-5 \
--warmup_ratio 0.1 \
--weight_decay 0.01 \
--debug \
--seed 42 | tee -a ./models/$MODEL_NAME.log
```


20220119-01: run classifier without using Huggingface notrainer, this enalbes us to customise the traning process as we 
needed. 
- Evaluate training accuracy after each epoch
- Save the best model according to evaluation accuracy
* Result: eval-55.5%, test-55.91% @ around 10 epochs
```shell
export MODEL_NAME=c3-ft-chinese_roberta_L-12_H-768_notrainer_lr2e05_ep100
CUDA_VISIBLE_DEVICES="2,3,4,6" accelerate launch run_c3_no_trainer.py \
--output_dir ./models/$MODEL_NAME \
--best_output_dir ./models/$MODEL_NAME/best_model \
--validation_file ./data/c3/c3-all-hf-dev-zh.json \
--train_file ./data/c3/c3-all-hf-train-zh.json \
--model_name_or_path ./models/chinese_roberta_L-12_H-768 \
--num_train_epochs 100 \
--per_device_train_batch_size 3 \
--per_device_eval_batch_size=3 \
--gradient_accumulation_steps 6 \
--learning_rate 2e-5 \
--warmup_ratio 0.1 \
--weight_decay 0.01 \
--eval_train \
--seed 42 | tee -a ./models/$MODEL_NAME.log
```


20220119-02: run classifier without using Huggingface trainer, this enalbes us to customise the traning process as we 
needed. 
- Change hyper-parameters and see if we can improve the accuracy
- learning rate: {1e-05, 2e-05, 3e-05}
- warmup_ratio: 0.06
- weight_decay: 0.01
- epochs: 20
* Result 1e-05, max_length: 128: eval-55.29%, test-55.68% @ around 10 epochs
* Result 2e-05, max_length: 128: eval-56.00%, test-54.93% @ around 18 epochs
* Result 2e-05, max_length: 512: eval-59.75%, test-58.25% @ around 17 epochs
* Result 3e-05, max_length: 128: eval-55.06%, test-55.29% @ around 20 epochs
```shell
export MODEL_NAME=chinese_roberta_L-12_H-768
export FT_MODEL_NAME=lr2e05_ep20
CUDA_VISIBLE_DEVICES="1,2,3,6" accelerate launch run_c3_no_trainer.py \
--output_dir ./models-ft/$MODEL_NAME/$FT_MODEL_NAME \
--best_output_dir ./models-ft/$MODEL_NAME/$FT_MODEL_NAME/best_model \
--validation_file ./data/c3/c3-all-hf-dev-zh.json \
--train_file ./data/c3/c3-all-hf-train-zh.json \
--model_name_or_path ./models/$MODEL_NAME \
--num_train_epochs 20 \
--per_device_train_batch_size 3 \
--per_device_eval_batch_size 3 \
--gradient_accumulation_steps 6 \
--max_length 512 \
--learning_rate 2e-5 \
--warmup_ratio 0.06 \
--weight_decay 0.01 \
--eval_train \
--seed 42 | tee -a ./models-ft/$MODEL_NAME/$FT_MODEL_NAME.log
```

20220120-01: fine-tune with BERT base Chinese in Huggingface with default hyper-parameters from c3 dataset
- Change hyper-parameters and see if we can improve the accuracy
- learning rate: {1e-05, 2e-05, 3e-05}
- warmup_ratio: 0.06
- weight_decay: 0.01
- epochs: 20
* Result 1e-05, max_length: 128: eval-57.57%, test-??% @ around 7 epochs
* Result 2e-05, max_length: 128: eval-58.28%, test-??% @ around 6 epochs
* Result 2e-05, max_length: 512: eval-65.00%, test-64.52% @ around 15 epochs
* Result 3e-05, max_length: 128: eval-57.36%, test-??% @ around 5 epochs
training the sample with max-seq of 512, we want to re-train all models if the accuracy is improved significantly
```shell
export MODEL_NAME=bert-base-chinese
export FT_MODEL_NAME=lr3e05_ep20
CUDA_VISIBLE_DEVICES="1,2,3,6" accelerate launch run_c3_no_trainer.py \
--output_dir ./models-ft/$MODEL_NAME/$FT_MODEL_NAME \
--best_output_dir ./models-ft/$MODEL_NAME/$FT_MODEL_NAME/best_model \
--validation_file ./data/c3/c3-all-hf-dev-zh.json \
--train_file ./data/c3/c3-all-hf-train-zh.json \
--model_name_or_path ./models/$MODEL_NAME \
--num_train_epochs 20 \
--per_device_train_batch_size 3 \
--per_device_eval_batch_size 3 \
--gradient_accumulation_steps 6 \
--learning_rate 3e-5 \
--warmup_ratio 0.06 \
--weight_decay 0.01 \
--eval_train \
--seed 42 | tee -a ./models-ft/$MODEL_NAME/$FT_MODEL_NAME.log
```


20220121-01: fine-tune with BERT base uncased in Huggingface on RACE dataset with default hyper-parameters from c3 dataset
- learning rate: {1e-05, 2e-05, 3e-05}
- warmup_ratio: 0.06
- weight_decay: 0.01
- epochs: 30
* Result 1e-05, max_length: 512: eval-66.81%, test_middle-71.45%, test_high-63.83% @ around 24 epochs
* Result 2e-05, max_length: 512: eval-68.30%, test_middle-73.12%, test_high-64.37% @ around 25 epochs
* Result 3e-05, max_length: 512: eval-68.01%, test_middle-72.98%, test_high-63.84% @ around 30 epochs
training the sample with max-seq of 512, we want to re-train all models if the accuracy is improved significantly
```shell
# CUDA_VISIBLE_DEVICES="5" accelerate launch run_race_no_trainer.py \
export MODEL_NAME=bert-base-uncased
export FT_MODEL_NAME=race_lr3e05_ep30
CUDA_VISIBLE_DEVICES="2,3,4,6" accelerate launch --config_file 4gpu_config.yaml run_race_no_trainer.py \
--output_dir ./models-ft/$MODEL_NAME/$FT_MODEL_NAME \
--best_output_dir ./models-ft/$MODEL_NAME/$FT_MODEL_NAME/best_model \
--validation_file ./data/RACE/hfjson/dev.json \
--train_file ./data/RACE/hfjson/train.json \
--model_name_or_path ./models/$MODEL_NAME \
--num_train_epochs 30 \
--per_device_train_batch_size 3 \
--per_device_eval_batch_size 3 \
--gradient_accumulation_steps 6 \
--learning_rate 3e-5 \
--max_length 512 \
--uncased \
--warmup_ratio 0.06 \
--weight_decay 0.01 \
--eval_train \
--seed 42 | tee -a ./models-ft/$MODEL_NAME/$FT_MODEL_NAME.log
```


20220316-01: fine-tune with BERT base uncased in Huggingface on DREAM_RACE dataset with default hyper-parameters from c3 dataset
- learning rate: {1e-05, 2e-05, 3e-05}
- warmup_ratio: 0.06
- weight_decay: 0.01
- epochs: 30
* Result 1e-05, max_length: 512: eval-67.52%, test_middle-71.73%, test_high-64.55%, test_dream-68.84% @ around 21 epochs
* Result 2e-05, max_length: 512: eval-67.98%, test_middle-73.19%, test_high-64.07%, test_dream-69.77% @ around 29 epochs
* Result 3e-05, max_length: 512: eval-68.56%, test_middle-71.94%, test_high-64.92%, test_dream-68.45% @ around 25 epochs
```shell
# CUDA_VISIBLE_DEVICES="5" accelerate launch run_race_no_trainer.py \
export MODEL_NAME=bert-base-uncased
export FT_MODEL_NAME=dream_race_lr3e05_ep30
CUDA_VISIBLE_DEVICES="0,3,4,5" accelerate launch --config_file 4gpu_config.yaml run_race_no_trainer.py \
--output_dir ./models-ft/$MODEL_NAME/$FT_MODEL_NAME \
--best_output_dir ./models-ft/$MODEL_NAME/$FT_MODEL_NAME/best_model \
--validation_file ./data/DREAM_RACE/race_dream-hf-dev-en.json \
--train_file ./data/DREAM_RACE/race_dream-hf-train-en.json \
--model_name_or_path ./models/$MODEL_NAME \
--num_train_epochs 30 \
--per_device_train_batch_size 6 \
--per_device_eval_batch_size 6 \
--gradient_accumulation_steps 6 \
--learning_rate 3e-5 \
--max_length 512 \
--uncased \
--warmup_ratio 0.06 \
--weight_decay 0.01 \
--eval_train \
--seed 42 | tee -a ./models-ft/$MODEL_NAME/$FT_MODEL_NAME.log
```


20220323-01: fine-tune with BERT base uncased in Huggingface on DREAM dataset with default hyper-parameters from c3 dataset
- learning rate: {1e-05, 2e-05, 3e-05}
- warmup_ratio: 0.06
- weight_decay: 0.01
- epochs: 30
- 4min/epoch, 27G/gpu * 4
* Result 1e-05, max_length: 512: eval-59.31%, test-60.85% @ around 27 epochs
* Result 2e-05, max_length: 512: eval-61.37%, test-60.31% @ around 11 epochs
* Result 3e-05, max_length: 512: eval-61.28%, test-61.64% @ around 26 epochs
```shell
# CUDA_VISIBLE_DEVICES="5" accelerate launch run_race_no_trainer.py \
export MODEL_NAME=bert-base-uncased
export FT_MODEL_NAME=dream_lr3e05_ep30
CUDA_VISIBLE_DEVICES="2,3,4,5" accelerate launch --config_file 4gpu_config.yaml run_race_no_trainer.py \
--output_dir ./models-ft/$MODEL_NAME/$FT_MODEL_NAME \
--best_output_dir ./models-ft/$MODEL_NAME/$FT_MODEL_NAME/best_model \
--validation_file ./data/DREAM/dream-hf-dev-en.json \
--train_file ./data/DREAM/dream-hf-train-en.json \
--model_name_or_path ./models/$MODEL_NAME \
--num_train_epochs 30 \
--per_device_train_batch_size 6 \
--per_device_eval_batch_size 6 \
--gradient_accumulation_steps 3 \
--learning_rate 3e-5 \
--max_length 512 \
--uncased \
--warmup_ratio 0.06 \
--weight_decay 0.01 \
--eval_train \
--seed 42 | tee -a ./models-ft/$MODEL_NAME/$FT_MODEL_NAME.log
```


20220414-01: fine-tune with BERT base uncased in Huggingface on v6-ultimate translated c3 dataset with default hyper-parameters from c3 dataset
- learning rate: {1e-05, 2e-05, 3e-05}
- warmup_ratio: 0.06
- weight_decay: 0.01
- epochs: 30
* Result 1e-05, max_length: 512: eval-40.38%, test-39.88% @ around 4 epochs
* Result 2e-05, max_length: 512: eval-41.3%, test-40.39% @ around 4 epochs
* Result 3e-05, max_length: 512: eval-41.25%, test-40.23% @ around 3 epochs
training the sample with max-seq of 512, we want to re-train all models if the accuracy is improved significantly
```shell
# CUDA_VISIBLE_DEVICES="5" accelerate launch run_race_no_trainer.py \
for LRATE in 1e-5 2e-5 3e-5
do
export MODEL_NAME=bert-base-uncased
export FT_MODEL_NAME=c3_v6_ultimate_lr$LRATE\_ep30
CUDA_VISIBLE_DEVICES="1,2,3,4" accelerate launch --config_file 4gpu_config.yaml run_race_no_trainer.py \
--output_dir ./models-ft/$MODEL_NAME/$FT_MODEL_NAME \
--best_output_dir ./models-ft/$MODEL_NAME/$FT_MODEL_NAME/best_model \
--validation_file ./data/c3/c3-all-hf-dev-v6-ultimate_v2-bpe20k-en.json \
--train_file ./data/c3/c3-all-hf-train-v6-ultimate_v2-bpe20k-en.json \
--model_name_or_path ./models/$MODEL_NAME \
--num_train_epochs 30 \
--per_device_train_batch_size 6 \
--per_device_eval_batch_size 6 \
--gradient_accumulation_steps 3 \
--learning_rate $LRATE \
--max_length 512 \
--uncased \
--warmup_ratio 0.06 \
--weight_decay 0.01 \
--eval_train \
--seed 42 | tee -a ./models-ft/$MODEL_NAME/$FT_MODEL_NAME.log
done
```


20220415-01: fine-tune with RACE&DREAM BERT model on v6-ultimate translated c3 dataset with default hyper-parameters from c3 dataset
- learning rate: {1e-05, 2e-05, 3e-05}
- warmup_ratio: 0.06
- weight_decay: 0.01
- epochs: 30
* Result 1e-05, max_length: 512: eval-44.92%, test-43.53% @ around 9 epochs
* Result 2e-05, max_length: 512: eval-44.63%, test-44.17% @ around 5 epochs
* Result 3e-05, max_length: 512: eval-44.86%, test-44.5% @ around 5 epochs
training the sample with max-seq of 512, we want to re-train all models if the accuracy is improved significantly
```shell
# CUDA_VISIBLE_DEVICES="5" accelerate launch run_race_no_trainer.py \
for LRATE in 1e-5 2e-5 3e-5
do
export MODEL_NAME=bert-base-uncased
export FT_MODEL_NAME=dnr_c3_v6_ultimate_lr$LRATE\_ep30
CUDA_VISIBLE_DEVICES="1,2,3,4" accelerate launch --config_file 4gpu_config.yaml run_race_no_trainer.py \
--output_dir ./models-ft/$MODEL_NAME/$FT_MODEL_NAME \
--best_output_dir ./models-ft/$MODEL_NAME/$FT_MODEL_NAME/best_model \
--validation_file ./data/c3/c3-all-hf-dev-v6-ultimate_v2-bpe20k-en.json \
--train_file ./data/c3/c3-all-hf-train-v6-ultimate_v2-bpe20k-en.json \
--model_name_or_path ./models-ft/$MODEL_NAME/dream_race_lr3e05_ep30 \
--num_train_epochs 30 \
--per_device_train_batch_size 6 \
--per_device_eval_batch_size 6 \
--gradient_accumulation_steps 3 \
--learning_rate $LRATE \
--max_length 512 \
--uncased \
--warmup_ratio 0.06 \
--weight_decay 0.01 \
--eval_train \
--seed 42 | tee -a ./models-ft/$MODEL_NAME/$FT_MODEL_NAME.log
done
```




Run inference using run_inference.py
- using "--uncased" argument for language models that only supports lowercase inputs
```shell
export MODEL_NAME=bert-base-uncased/dream_race_lr2e05_ep30
CUDA_VISIBLE_DEVICES="3" python run_inference_bk.py \
--validation_file ./data/infer-hf-test.json \
--model_name_or_path ./models-ft/$MODEL_NAME \
--max_length 512 \
--per_device_eval_batch_size=1 \
--uncased \
--task_name race

# details can be found in Fairseq project's CLQA demo
CUDA_VISIBLE_DEVICES="3" python run_inference.py
```


```
Package            Version    
------------------ -----------
accelerate         0.5.1      
aiohttp            3.8.1      
aiosignal          1.2.0      
async-timeout      4.0.1      
attrs              21.2.0     
certifi            2021.5.30  
charset-normalizer 2.0.6      
click              8.0.1      
datasets           1.16.1     
dill               0.3.4      
filelock           3.3.0      
frozenlist         1.2.0      
fsspec             2021.11.1  
huggingface-hub    0.1.2      
idna               3.2        
joblib             1.0.1      
multidict          5.2.0      
multiprocess       0.70.12.2  
numpy              1.21.2     
packaging          21.0       
pandas             1.3.4      
pip                20.0.2     
pkg-resources      0.0.0      
pyarrow            6.0.1      
pyparsing          2.4.7      
python-dateutil    2.8.2      
pytz               2021.3     
PyYAML             5.4.1      
regex              2021.9.30  
requests           2.26.0     
ruamel.yaml        0.17.16    
ruamel.yaml.clib   0.2.6      
sacremoses         0.0.46     
scikit-learn       1.0.2      
scipy              1.7.3      
setuptools         44.0.0     
six                1.16.0     
sklearn            0.0        
threadpoolctl      3.0.0      
tokenizers         0.10.3     
torch              1.9.1      
tqdm               4.62.3     
transformers       4.13.0.dev0
typing-extensions  3.10.0.2   
urllib3            1.26.7     
xxhash             2.0.2      
yarl               1.7.2 
```
