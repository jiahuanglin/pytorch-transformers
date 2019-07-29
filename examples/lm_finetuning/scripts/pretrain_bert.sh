# nvprof
NVPROF=/pkgs/cuda-10.0/bin/nvprof
NVPROF_OPTIONS="--profile-from-start off --export-profile huggingface_t4_fused_gelu_profile.nvprof --print-summary"

DATA=/scratch/hdd001/home/jacoblin/NLP-corpus/wiki_corpus/huggingface/pregen_data/

# $NVPROF $NVPROF_OPTIONS \
python finetune_on_pregenerated.py \
    --pregenerated_data $DATA \
    --bert_model bert-large-uncased \
    --do_lower_case \
    --output_dir finetuned_lm/ \
    --epochs 3 \
    --fp16 \
    --train_batch_size 14