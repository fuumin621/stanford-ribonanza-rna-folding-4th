for fold in {0..4}
do
python train.py --seed 2023 --fold $fold --epoch 150 --batch_size 64 --gpus 1 --lr 4e-3 --logdir conv_gnn_l12 --num_workers 8 --disable_compile 
python train_finetune.py --seed 2023 --fold $fold --epoch 50 --batch_size 64 --gpus 1 --lr 5e-4 \
    --logdir conv_gnn_l12 --resumedir exp112/conv_gnn_l12 --num_workers 8
python val.py --fold $fold --batch_size 512 --logdir exp112_finetune/conv_gnn_l12  \
     --resumedir exp112_finetune/conv_gnn_l12
python eval.py --fold $fold --batch_size 512 --logdir exp112_finetune/conv_gnn_l12  \
     --resumedir exp112_finetune/conv_gnn_l12
done
python postprocess.py