# nvprof
NVPROF=/pkgs/cuda-10.0/bin/nvprof
NVPROF_OPTIONS="--profile-from-start off --export-profile huggingface_t4_fused_gelu_profile.nvprof --print-summary"

DATA=/scratch/hdd001/home/jacoblin/NLP-corpus/wiki_corpus/huggingface/pregen_data_128/

# $NVPROF $NVPROF_OPTIONS \
python finetune_on_pregenerated.py \
    --pregenerated_data $DATA \
    --bert_model bert-large-uncased \
    --do_lower_case \
    --restore_dir 4_nodes_p100_grad_acc_5_fp32_bert_large/ \
    --output_dir 4_nodes_p100_grad_acc_5_fp32_bert_large/ \
    --epochs 3 \
    --batch_size 10 \
    --gradient_accumulation_steps 1

# --fp16 \
# --benchmark 

# fp32 training max batch size => 10 of seq len 128
# fp16 training max batch size => 20 of seq 128