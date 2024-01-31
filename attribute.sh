#!/bin/bash
#SBATCH --partition=gpu //use gpu partition
#SBATCH --gres=gpu:1 //the amount of gpu for your job
#SBATCH --ntasks=1 //number of parallel tasks // [OPTIONAL] list of specific nodes to use
#SBATCH --exclude=ilps-cn111 // [OPTIONAL] list of specific nodes to exclude
#SBATCH --cpus-per-task=1 //specify the number of cpus
#SBATCH --mem=32G //the amount of CPU mem
#SBATCH --time=2-10 //formatted running time. Avaliable specifiers: "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds". Default limit is 1h!
##SBATCH --begin=now //when start your job, OPTIONAL

#SBATCH -o out.o //where to write stdout
#SBATCH -e err.e //where to write stderr


set -euo pipefail

model_dir= ## model directory
src_lang= ## source language
tgt_lang= ## target language

## Note: for this script to work, the spm models need to be in the model dir in the format spm.{src_lang}.model and spm.{tgt_lang}.model
# make sure to check the configuration in the demo configuration file. for use with a multi-ecoder model, the flag is_multi_encoder must be True

python stopes/demo/alti/minimal_example/compute_nllb_alti.py \
    input_filename=$(pwd)/stopes/demo/alti/minimal_example/contrapro_de.tsv \
    metrics_filename=$(pwd)/test_output.tsv \
    alignment_filename=$(pwd)/test_output_alignments.jsonl \
    src_lang=${src_lang} \
    tgt_lang=${tgt_lang} \
    +preset=demo +demo_dir=${model_dir}


