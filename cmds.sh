cd ../recsys_rl_analysis; sh prep_mujoco.sh; cd ../pytorch-a2c-ppo-acktr-gail-modified
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin

CUDA_VISIBLE_DEVICES=0 python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 10000000 --use-linear-lr-decay --use-proper-time-limits --gail
