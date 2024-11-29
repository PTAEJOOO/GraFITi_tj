### ex1.py ###

nlayers="1"
attn_head="1"
latent_dim="16"

for nlayer in $nlayers; do
    for at in $attn_head; do
        for ld in $latent_dim; do
            python ex1.py \
            --epochs 200 --learn-rate 0.001 \
            --attn-head $at --latent-dim $ld --nlayers $nlayer \
            --dataset physionet2012 --fold 0 -ct 36 -ft 12
        done
    done   
done
