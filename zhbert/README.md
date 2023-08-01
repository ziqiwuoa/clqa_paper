# zhbert
The QA models are trained based on the code base from huggingface, for more information please see [transformers examples](https://github.com/huggingface/transformers/tree/master/examples/pytorch/language-modeling). The code is in the `zhbert` folder.

## Requirements and Installation
Huggingface transformers version 4.13.0 dev and above is required.

Installing huggingface transformers:
```shell
pip install git+https://github.com/huggingface/transformers
``` 

## Datasets
The publicly accessible QA datasets are listed below:
- DREAM [https://dataset.org/dream/](https://dataset.org/dream/)
- RACE [https://github.com/qizhex/RACE_AR_baselines](https://github.com/qizhex/RACE_AR_baselines)
- C3 [https://github.com/nlpdata/c3](https://github.com/nlpdata/c3)
- EXAMS (We only used AR/FR/EN) [https://github.com/mhardalov/exams-qa](https://github.com/mhardalov/exams-qa)

The C3 and EXAMS datasets are translated to English by all the NMT models trained in this paper. The EXAMS dataset is also translated to English using Google Translate. We release all the translated data which which can be downloaded [here](https://drive.google.com/file/d/1X2VLGqFMk0GyX_lN2d84CHtwZ5UkgHOx/view).

## Pre-trained Language Model
We fine-tune the QA model based on BERT BASE UNCASED model [https://huggingface.co/bert-base-uncased](https://huggingface.co/bert-base-uncased).

## Find-tune a QA Model
```shell
export MODEL_NAME=bert-base-uncased
export FT_MODEL_NAME=xxx
accelerate launch run_race_no_trainer.py \
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

## Evaluation
```shell
export MODEL_NAME=xxx
python run_eval.py \
--validation_file ./data/c3/c3-all-hf-test-v6-ultimate_v2-bpe20k-en.json \
--model_name_or_path ./models-ft/$MODEL_NAME \
--max_length 512 \
--per_device_eval_batch_size=12 \
--uncased \
--task_name race
```

## Installed Packages:
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
