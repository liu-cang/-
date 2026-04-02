N_GPUS=4
BATCH_SIZE=8
DATA_ROOT=/data/pgh2874/SFOpen_suites
# OUTPUT_DIR=./outputs/def-detr-base/city2foggy/teaching_mask_1_gpu2_bs8
# OUTPUT_DIR=./outputs/def-detr-base/SFUOD/city2foggy_unk2/teaching_mask_unk_gpu4
OUTPUT_DIR=./outputs/def-detr-base/SFUOD/city2foggy_unk3/teaching_mask_unk_gpu4

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 torchrun \
--rdzv_endpoint localhost:26503 \
--nproc_per_node=${N_GPUS} \
main.py \
--backbone resnet50 \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--num_classes 10 \
--dropout 0.0 \
--data_root ${DATA_ROOT} \
--source_dataset cityscapes \
--target_dataset foggy_cityscapes \
--batch_size ${BATCH_SIZE} \
--eval_batch_size ${BATCH_SIZE} \
--lr 2e-4 \
--lr_backbone 2e-5 \
--lr_linear_proj 2e-5 \
--alpha_ema 0.999 \
--epoch 30 \
--epoch_lr_drop 80 \
--mode teaching_mask \
--threshold 0.3 \
--dynamic_update \
--max_update_iter 5 \
--only_class_loss \
--use_pseudo_label_weights \
--sfuod \
--unk_version 3 \
--output_dir ${OUTPUT_DIR} \
--resume ${OUTPUT_DIR}/../source_only/model_last.pth
# OUTPUT_DIR=./outputs/def-detr-base/SFUOD/city2foggy_unk2/source_only