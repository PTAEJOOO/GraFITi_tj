### grafiti ###

nlayers="2 4"
attn_head="2 4"
latent_dim="32 64 128"

for nlayer in $nlayers; do
    for at in $attn_head; do
        for ld in $latent_dim; do
            python eval_grafiti.py \
            --epochs 100 --learn-rate 0.001 --batch-size 32 \
            --attn-head $at --latent-dim $ld --nlayers $nlayer \
            --dataset physionet2012 --fold 0 -ct 36 -ft 12
        done
    done   
done