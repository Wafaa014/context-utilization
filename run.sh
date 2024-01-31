#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --time=5-0
#SBATCH --begin=now
#SBATCH --ntasks=1


#SBATCH -o out_run.o
#SBATCH -e err_run.e

set -euo pipefail

data_dir= ## data folder
save_dir= ## folder to save the model
src_lang= ## source language
tgt_lang= ## target language
wandb_project= ## weights & biases project
comet_dir= ## folder to save comet models
contrapro_dir= ##ContraPro data folder

#train an spm encoder in case of training from scratch
VOCAB_SIZE=20000
for lang in ${src_lang} ${tgt_lang}; do
    python contextual-mt/scripts/spm_train.py \
        ${data_dir}/train.${lang} \
        --vocab-size $VOCAB_SIZE \
        --model-prefix ${data_dir}/prep/spm.${lang} \
        --vocab-file ${data_dir}/prep/dict.${lang}.txt
done

# sentence piece encoding
for split in train valid test; do
    for lang in ${src_lang} ${tgt_lang}; do
        python contextual-mt/scripts/spm_encode.py \
            --model ${data_dir}/prep/spm.$lang.model \
                < ${data_dir}/${split}.${lang} \
                > ${data_dir}/prep/${split}.sp.${lang}
    done
done

# binarizing data
fairseq-preprocess \
    --source-lang ${src_lang} --target-lang ${tgt_lang} \
    --trainpref ${data_dir}/prep/train.sp \
    --validpref ${data_dir}/prep/valid.sp \
    --testpref ${data_dir}/prep/test.sp \
    --srcdict ${data_dir}/prep/dict.${src_lang}.txt \
    --tgtdict ${data_dir}/prep/dict.${tgt_lang}.txt \
    --destdir ${data_dir}/bin

# training a model from scratch
fairseq-train \
    ${data_dir}/bin --user-dir contextual-mt/contextual_mt \
    --task document_translation \
    --log-interval 10 \
    --arch contextual_transformer --share-decoder-input-output-embed  \
    --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.1 \
    --lr 5e-4 --lr-scheduler inverse_sqrt  --warmup-updates 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --dropout 0.3 --weight-decay 0.0001 \
    --max-tokens 4096 --update-freq 8 --patience 10 --seed 42 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-remove-bpe sentencepiece \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --no-epoch-checkpoints \
    --save-dir ${save_dir} \
    --source-context-size 5 --target-context-size 5 \
    --sample-context-size \
    --source-lang ${src_lang} --target-lang ${tgt_lang} \
    --wandb-project ${wandb_project} \
    #--multi-encoder

 # translate test data
python contextual-mt/contextual_mt/docmt_translate.py \
    --path ${save_dir} \
    --spm-path ${data_dir}/prep \
    --source-lang ${src_lang} --target-lang ${tgt_lang} \
    --source-file ${data_dir}/test.${src_lang} \
    --predictions-file ${data_dir}/test.pred.${tgt_lang} \
    --docids-file ${data_dir}/test.docids \
    --batch-size 128 \
    --beam 1 \
    --checkpoint-file checkpoint_best.pt \
    --source-context-size 5 --target-context-size 5 \
    #--random-context ## to test with random context
    #--gold-target-context --reference-file ${data_dir}/test.${tgt_lang} ## to evaluate using golden target context

# generate BLEU and COMET scores for translations
python contextual-mt-bilingual/scripts/score.py ${data_dir}/test.pred.${tgt_lang} ${data_dir}/test.${tgt_lang} \
    --src ${data_dir}/test.${src_lang} \
    --comet-model Unbabel/wmt22-comet-da \
    --comet-path ${comet_dir}

#contrastive evaluation on ContraPro data
python contextual-mt/contextual_mt/docmt_contrastive_eval.py \
    --source-lang ${src_lang} --target-lang ${tgt_lang} \
    --source-file ${contrapro_dir}/contrapro.text.en \
    --src-context-file ${contrapro_dir}/contrapro.context.en \
    --target-file ${contrapro_dir}/contrapro.text.de \
    --tgt-context-file ${contrapro_dir}/contrapro.context.de \
    --source-context-size 5 --target-context-size 5 \
    --path ${save_dir} \
    --checkpoint-file checkpoint_best.pt \
    --batch-size 32 \
    #--save-json log.json 


