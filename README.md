#Introduction
This repository contains the code and corresponding data for *Multiple-Choice Cross-Lingual Question Answering through Neural
Machine Translation: An Empirical Study*

<br>

#Neural Machine Translation
We train the machine translation models using [Fairseq](https://github.com/facebookresearch/fairseq) toolkit.

##Requirements and Installation
- PyTorch version == 1.0.0/1.1.0
- Python version >= 3.5
- Clone repository to local machine
- Enter the "fairseq" folder and run the following script
```shell script
pip install --editable .
```

##Datasets
- United Nations Parallel Corpus: [https://conferences.unite.un.org/uncorpus](https://conferences.unite.un.org/uncorpus)
- Conference on Machine Translation (WMT17) Data sets (We only use the test sets) [https://www.statmt.org/wmt17/translation-task.html#download](https://www.statmt.org/wmt17/translation-task.html#download)
- UM-Corpus [http://nlp2ct.cis.umac.mo/um-corpus/](http://nlp2ct.cis.umac.mo/um-corpus/)
- Web Inventory of Transcribed and Translated Talks (WIT3) [https://wit3.fbk.eu/](https://wit3.fbk.eu/) 

##Pre-processing
###Chinese -> English
1. Preprocess raw corpra based on `preprocess_2.sh` in [BERT paper preprocess repo](https://github.com/teslacool/preprocess_iwslt)
2. Furture preprocess the tokenized data for fairseq based on their [documentation](https://fairseq.readthedocs.io/en/latest/getting_started.html#training-a-new-model)
    ```
    TEXT=examples/translation/un.en-zh
    fairseq-preprocess --source-lang en --target-lang zh \
        --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
        --destdir data-bin/un.en-zh
    ```
   
###Arabic/French/Spanish -> English
For language pairs of AR->EN, FR->EN and ES->EN, we pre-process the corpora directly using the following script:
```shell
SRC=ar
TEXT=examples/translation/news_ext/$SRC-en
fairseq-preprocess --source-lang $SRC --target-lang en \
--trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
--destdir data-bin/news_ext/$SRC-en \
--workers 20
```

##Train a NMT model
For the corpora with less than 1 million parallel sentences, we use the following script: 
```shell
MODEL_NAME=transformer-base-x-en
cd checkpoints
mkdir $MODEL_NAME
cd $MODEL_NAME
touch training.log
cd ../../
CUDA_VISIBLE_DEVICES=3 fairseq-train --share-decoder-input-output-embed \
data-bin/news_ext/ar-en \
--source-lang ar --target-lang en \
--arch transformer \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
--lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000  --warmup-init-lr 1e-07 \
--dropout 0.1  --weight-decay 0.0001 \
--max-tokens 4096 \
--max-epoch 50 \
--eval-bleu \
--eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
--eval-bleu-remove-bpe \
--eval-bleu-print-samples \
--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
--save-dir checkpoints/$MODEL_NAME | tee -a checkpoints/$MODEL_NAME/training.log
```

For the corpora with more than 1 million parallel sentences, we the this script:
```shell
MODEL_NAME=transformer-big-x-en
cd checkpoints
mkdir $MODEL_NAME
cd $MODEL_NAME
touch training.log
cd ../../
CUDA_VISIBLE_DEVICES=5 fairseq-train  --share-decoder-input-output-embed \
data-bin/umeln.bpe20k.en-zh --source-lang zh --target-lang en \
--arch transformer_vaswani_wmt_en_de_big --share-decoder-input-output-embed \
--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
--lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
--dropout 0.1 --weight-decay 0.0001 \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--max-tokens 8192 --update-freq 4 \
--max-epoch 50 \
--eval-bleu \
--eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
--eval-bleu-remove-bpe \
--eval-bleu-print-samples \
--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
--save-dir checkpoints/$MODEL_NAME | tee -a checkpoints/$MODEL_NAME/training.log
```

##Evaluation
1. Generate the translation
    ```shell script
    SPLIT_NAME=xxx
    DATA_BIN_NAME=yyy
    fairseq-generate \
        --path checkpoints/$SPLIT_NAME/checkpoint_best.pt data-bin/$DATA_BIN_NAME/$LANG-en \
        --source-lang $LANG \
        --target-lang en \
        --batch-size 32 \
        --gen-subset test \
        --tokenizer moses \
        --beam 5 --remove-bpe > ext_gen_out_$SPLIT_NAME
    ```
2. Evaluate the BLEU score
    ```shell script
    SPLIT_NAME=xxx
    TEST_SET_NAME=yyy
    cat ../translation/ext_all_tests/$TEST_SET_NAME.test.raw.$LANG-en.en | sh tok.sh en > ref_$SPLIT_NAME
    echo "@@@ Evaluating split: $SPLIT_NAME"
    cat ./ext_results_original/ext_gen_out_$SPLIT_NAME | ggrep -P "^H" | sort -V | cut -f 3- | sh tok.sh en > ./ext_results_original/hyp_$SPLIT_NAME
    
    # BLEU
    sacrebleu -tok 'none' ref_$SPLIT_NAME < ./ext_results_original/hyp_$SPLIT_NAME
    ```
   
   
<br><br>


#Question Answering
The QA models are trained based on the code base from huggingface, for more information please see [transformers examples](https://github.com/huggingface/transformers/tree/master/examples/pytorch/language-modeling). 

##Requirements and Installation
Huggingface transformers version 4.13.0 dev and above is required.

Installing huggingface transformers:
```shell
pip install git+https://github.com/huggingface/transformers
``` 

##Datasets
The publicly accessible QA datasets are listed below:
- DREAM [https://dataset.org/dream/](https://dataset.org/dream/)
- RACE [https://github.com/qizhex/RACE_AR_baselines](https://github.com/qizhex/RACE_AR_baselines)
- C3 [https://github.com/nlpdata/c3](https://github.com/nlpdata/c3)
- EXAMS (We only used AR/FR/EN) [https://github.com/mhardalov/exams-qa](https://github.com/mhardalov/exams-qa)

The C3 and EXAMS datasets are translated to English by all the NMT models trained in this paper. The EXAMS dataset is also translated to English using Google Translate. We release all the translated data which which can be downloaded [here](https://drive.google.com/file/d/1X2VLGqFMk0GyX_lN2d84CHtwZ5UkgHOx/view).


##Find-tune a QA Model
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

##Evaluation
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