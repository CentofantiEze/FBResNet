#! /bin/sh

NUMBER_OF_MODELS=5

opt[0]="--model_id model_006_test_lr_1e-3_ --lr 1e-3"
opt[1]="--model_id model_007_test_lr_1e-2_ --lr 1e-2"
opt[2]="--model_id model_008_test_lr_5e-2_ --lr 5e-2"
opt[3]="--model_id model_009_test_lr_1e-1_ --lr 1e-1"
opt[4]="--model_id model_010_test_lr_5e-1_ --lr 5e-1"

while [ $NUMBER_OF_MODELS -ge 1 ]
do
    let "NUMBER_OF_MODELS-=1"
    python ../scripts/FBRN_click.py \
        ${opt[$NUMBER_OF_MODELS]} \
        --train_opt True \
        --test_opt True \
        --plot_opt True \
        --dataset_folder ../Datasets/ \
        --output_folder /workspace/repos/FBResNet/test_dir/ \
        --a 1 \
        --p 1 \
        --n 2000 \
        --m 50 \
        --nb_blocks 20 \
        --im_set Set1 \
        --noise 0.05 \
        --constraint cube \
        --train_size 400 \
        --val_size 200 \
        --batch_size 32 \
        --nb_epochs 20 \
        --freq_val 1 \
        --loss_elt True \
        --save_signals False \
        --save_outputs False \
        --save_model False \
        --save_hist True \
    &
done
disown
