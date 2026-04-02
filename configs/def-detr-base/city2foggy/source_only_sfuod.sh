N_GPUS=4
BATCH_SIZE=8
# N_GPUS=1
# BATCH_SIZE=4
DATA_ROOT=/data/pgh2874/SFOpen_suites
OUTPUT_DIR=./outputs/def-detr-base/SFUOD/city2foggy_unk3/source_only
# OUTPUT_DIR=./outputs/def-detr-base/SFUOD_exp/city2foggy/source_only

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 torchrun \
--rdzv_endpoint localhost:26500 \
--nproc_per_node=${N_GPUS} \
main.py \
--backbone resnet50 \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--num_classes 10 \
--dropout 0.1 \
--data_root ${DATA_ROOT} \
--source_dataset cityscapes \
--target_dataset foggy_cityscapes \
--batch_size ${BATCH_SIZE} \
--eval_batch_size ${BATCH_SIZE} \
--lr 2e-4 \
--lr_backbone 2e-5 \
--lr_linear_proj 2e-5 \
--epoch 80 \
--epoch_lr_drop 60 \
--mode single_domain \
--sfuod \
--unk_version 3 \
--print_freq 350 \
--output_dir ${OUTPUT_DIR}

