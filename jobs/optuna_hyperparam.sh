#! /bin/sh

python ../scripts/FBRN_click.py \
    --optuna True \
    --train_opt False \
    --test_opt False \
    --plot_opt False \
    --dataset_folder ../Datasets/ \
    --output_folder /workspace/repos/FBResNet/test_dir/ \
    --a 1 \
    --p 1 \
    --n 2000 \
    --m 50 \
    --model_id optuna_FBRN_ \
    --nb_blocks 20 \
    --im_set Set1 \
    --noise 0.05 \
    --constraint cube \
    --train_size 400 \
    --val_size 200 \
    --batch_size 64 \
    --lr 1e-1 \
    --nb_epochs 60 \
    --freq_val 1 \
    --loss_elt False \
    --save_signals False \
    --save_outputs False \
    --save_model False \
    --save_hist False \
