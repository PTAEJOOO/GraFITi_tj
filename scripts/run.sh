### grafiti ###

nlayers="1"
attn_head="1"
latent_dim="16"
# 128 256

for nlayer in $nlayers; do
    for at in $attn_head; do
        for ld in $latent_dim; do
            python train_grafiti.py \
            --epochs 200 --learn-rate 0.001 --batch-size 128 \
            --attn-head $at --latent-dim $ld --nlayers $nlayer \
            --dataset physionet2012 --fold 0 -ct 36 -ft 12
        done
    done   
done

            # python train_grafiti.py \
            # --epochs 100 --learn-rate 0.001 --batch-size 128 \
            # --attn-head $at --latent-dim $ld --nlayers $nlayer \
            # --dataset physionet2012 --fold 0 -ct 36 -ft 12 -wocat -ax
