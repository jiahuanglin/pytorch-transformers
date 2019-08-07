# nvprof
NVPROF=/pkgs/cuda-10.0/bin/nvprof
NVPROF_OPTIONS="--profile-from-start off --export-profile huggingface_t4_fused_gelu_profile.nvprof --print-summary"

DATA=/scratch/hdd001/home/jacoblin/NLP-corpus/wiki_corpus/huggingface/pregen_data_128/
# DATA=/scratch/ssd001/home/jacoblin/NLP-corpus/wiki_corpus/huggingface/pregen_data_128/


# $NVPROF $NVPROF_OPTIONS \
python finetune_on_pregenerated.py \
    --pregenerated_data $DATA \
    --bert_model bert-large-uncased \
    --do_lower_case \
    --output_dir finetuned_lm/ \
    --epochs 1 \
    --fp16 \
    --batch_size 20\
    --benchmark 

# fp32 training max batch size => 10 of seq len 128
# fp16 training max batch size => 20 of seq 128