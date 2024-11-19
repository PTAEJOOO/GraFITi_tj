### grafiti ###

nlayers="1 2 3 4"
attn_head="1 2 4"
latent_dim="16 32 64"
# 128 256

# for nlayer in $nlayers; do
#     for at in $attn_head; do
#         for ld in $latent_dim; do
#             python train_grafiti.py \
#             --epochs 200 --learn-rate 0.001 --batch-size 128 \
#             --attn-head $at --latent-dim $ld --nlayers $nlayer \
#             --dataset physionet2012 --fold 0 -ct 36 -ft 12 -wocat
#         done
#     done   
# done

python train_grafiti.py --epochs 100 --learn-rate 0.001 --batch-size 128 --attn-head 1 --latent-dim 64 --nlayers 1 --dataset physionet2012 --fold 0 -ct 36 -ft 12

# do
#     python train_grafiti.py \
#     --epochs 100 --learn-rate 0.001 --batch-size 128 \
#     --attn-head 1 --latent-dim 128 --nlayers 4 \
#     --dataset physionet2012 --fold 0 -ct 36 -ft 12 -ax

#     python train_grafiti.py \
#     --epochs 100 --learn-rate 0.001 --batch-size 128 \
#     --attn-head 1 --latent-dim 128 --nlayers 4 \
#     --dataset physionet2012 --fold 0 -ct 36 -ft 12
# done