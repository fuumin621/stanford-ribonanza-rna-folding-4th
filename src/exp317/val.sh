for fold in {0..4}
do
python val.py --fold $fold --batch_size 128 --logdir exp312_finetune/conv_gnn_l12 
done