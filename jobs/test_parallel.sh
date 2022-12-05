#! /bin/sh

NUMBER_OF_MODELS=2

opt[0]="--model_id test_model_001_"
opt[1]="--model_id test_model_002_"

while [ $NUMBER_OF_MODELS -ge 1 ]
do
    let "NUMBER_OF_MODELS-=1"
    python ../scripts/FBRN_click.py \
        ${opt[$NUMBER_OF_MODELS]} \
        --train_opt True \
        --test_opt True \
        --plot_opt True \
        --dataset_folder ../Datasets/ \
        --output_folder C:/Users/ezece/OneDrive/Documents/Inria-OPIS/repos/FBResNet/test_dir/ \
        --a 1 \
        --p 1 \
        --n 2000 \
        --m 50 \
        --nb_blocks 20 \
        --im_set Set1 \
        --noise 0.05 \
        --constraint cube \
        --train_size 70 \
        --val_size 30 \
        --batch_size 1 \
        --lr 1e-3 \
        --nb_epochs 15 \
        --freq_val 1 \
        --loss_elt False \
        --save_signals False \
        --save_outputs False \
        --save_model False \
        --save_hist True \
    &
done
disown
