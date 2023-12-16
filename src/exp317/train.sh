for fold in {0..4}
do
python train.py --seed $((4023 + fold)) --fold $fold --epoch 150 --batch_size 64 --gpus 1 --lr 4e-3 --logdir conv_gnn_l12 --num_workers 2 --disable_compile  \
   --pseudo_label_df ../../logs/exp302_finetune/conv_gnn_l12/fold$fold/test_pl_filterling_0.75_half.csv 
python train_finetune.py --seed $((6023 + fold)) --fold $fold --epoch 50 --batch_size 64 --gpus 1 --lr 2e-4 \
   --logdir conv_gnn_l12 --resumedir exp317/conv_gnn_l12 \
   --num_workers 2 \
   --pseudo_label_df ../../logs/exp302_finetune/conv_gnn_l12/fold$fold/test_pl_filterling_0.75_half.csv 
python eval.py --fold $fold --batch_size 128 --logdir exp317_finetune/conv_gnn_l12 
python val.py --fold $fold --batch_size 128 --logdir exp317_finetune/conv_gnn_l12 
done
python postprocess.py