#! /bin/sh

python ../scripts/FBRN_click.py \
    --train_opt True \
    --test_opt True \
    --plot_opt True \
    --dataset_folder ../Datasets/ \
    --output_folder /workspace/repos/FBResNet/test_dir/ \
    --a 1 \
    --p 1 \
    --n 2000 \
    --m 50 \
    --model_id test_gpu_debug_ \
    --nb_blocks 20 \
    --im_set Set1 \
    --noise 0.05 \
    --constraint cube \
    --train_size 400 \
    --val_size 200 \
    --batch_size 32 \
    --lr 1e-2 \
    --nb_epochs 30 \
    --freq_val 1 \
    --loss_elt False \
    --save_signals False \
    --save_outputs False \
    --save_model True \
    --save_hist True \
    &
