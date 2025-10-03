

your_ckpt=InternRobotics/InternVLA-M1-Pretrain-RT-1-Bridge/checkpoints/steps_50000_pytorch_model.pt
port=10093

python deployment/model_server/server_policy_M1.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16