### grafiti ###

<<<<<<< HEAD
nlayers="2 4"
attn_head="2 4"
latent_dim="64 128 256"
=======
nlayers="4"
attn_head="2 4"
latent_dim="64"
>>>>>>> ade2303b330cf7caa0f384f18716e101719fd502

for nlayer in $nlayers; do
    for at in $attn_head; do
        for ld in $latent_dim; do
            python eval_grafiti.py \
            --epochs 200 --learn-rate 0.001 --batch-size 128 \
            --attn-head $at --latent-dim $ld --nlayers $nlayer \
            --dataset physionet2012 --fold 0 -ct 36 -ft 12 -wocat
        done
    done   
done

# nlayers="4"
# attn_head="2 4"
# latent_dim="64 128 256"

# for nlayer in $nlayers; do
#     for at in $attn_head; do
#         for ld in $latent_dim; do
#             python eval_grafiti.py \
#             --epochs 200 --learn-rate 0.001 --batch-size 128 \
#             --attn-head $at --latent-dim $ld --nlayers $nlayer \
#             --dataset physionet2012 --fold 0 -ct 36 -ft 12 -wocat -ax
#         done
#     done   
# done