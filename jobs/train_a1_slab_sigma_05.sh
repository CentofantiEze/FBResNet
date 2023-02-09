#! /bin/sh

python ../scripts/FBRN_click.py \
    --train_opt True \
    --test_opt True \
    --plot_opt True \
    --dataset_folder ../Datasets/ \
    --rand_seed 42 \
    --output_folder /workspace/repos/FBResNet/test_dir/ \
    --a 1 \
    --p 1 \
    --n 2000 \
    --m 50 \
    --model_id model_1_slab_sigma_05_ \
    --nb_blocks 20 \
    --im_set Set1 \
    --noise 0.5 \
    --constraint slab \
    --train_size 400 \
    --val_size 200 \
    --batch_size 64 \
    --lr 5e-1 \
    --nb_epochs 130 \
    --freq_val 1 \
    --loss_elt True \
    --save_signals False \
    --save_outputs True \
    --save_model True \
    --save_hist True \
    &
