python /home/ubuntu/jonghoon/AI-TA/test_models.py
sleep 3
CUDA_VISIBLE_DEVICES=0 deepspeed /home/ubuntu/jonghoon/AI-TA/train/train_dpo.py

sleep 3600

sleep 3
CUDA_VISIBLE_DEVICES=0, 1 deepspeed /home/ubuntu/jonghoon/AI-TA/train/train_dpo.py
