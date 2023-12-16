####### baseline #########
for fold in {0..4}
do
python train.py --seed $((6023 + fold)) --fold $fold --epoch 150 --batch_size 64 --gpus 1 --lr 4e-3 --logdir conv_gnn_l12 --num_workers 2 --disable_compile
python train_finetune.py --seed $((7023 + fold)) --fold $fold --epoch 50 --batch_size 64 --gpus 1 --lr 2e-4 \
    --logdir conv_gnn_l12 --resumedir exp302/conv_gnn_l12 --num_workers 2
python eval.py --fold $fold --batch_size 128 --logdir exp302_finetune/conv_gnn_l12 
python eval_with_err.py --fold $fold --batch_size 128 --logdir exp302_finetune/conv_gnn_l12 
python apply_filter_pl.py --fold $fold --logdir exp302_finetune/conv_gnn_l12 --filter 0.75
python val.py --fold $fold --batch_size 128 --logdir exp302_finetune/conv_gnn_l12
done
python postprocess.py
