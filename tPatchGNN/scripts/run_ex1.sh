### tPatchGNN ###
patience=10
epoch=10
gpu=0

python train_tpatchgnn.py \
    --dataset physionet --state 'def' --history 24 \
    --epoch $epoch --patience $patience --batch_size 32 --lr 1e-3 \
    --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 1 \
    --te_dim 10 --node_dim 10 --hid_dim 64 \
    --outlayer Linear --seed 1 --gpu $gpu