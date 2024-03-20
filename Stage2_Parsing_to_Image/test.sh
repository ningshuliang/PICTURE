CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --master_port=6008 test.py
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=6009 train_stage2.py