from stable_baselines3 import A2C
import os

# import wandb

# wandb.init(
#     project='test2',
#     entity='sajadborhani',
#     sync_tensorboard=True,
#     config=None,
#     name='test2',
#     monitor_gym=True,
#     save_code=True,
# )
os.environ['CUDA_VISIBLE_DEVICES'] ='1'
model = A2C("MlpPolicy", "CartPole-v1", verbose=1, tensorboard_log="./a2c_cartpole_tensorboard/")
model.learn(total_timesteps=10_000, tb_log_name="first_run")
# Pass reset_num_timesteps=False to continue the training curve in tensorboard
# By default, it will create a new curve
# Keep tb_log_name constant to have continuous curve (see note below)
model.learn(total_timesteps=10_000, tb_log_name="second_run", reset_num_timesteps=False)
model.learn(total_timesteps=10_000, tb_log_name="third_run", reset_num_timesteps=False)