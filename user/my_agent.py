# # SUBMISSION: Agent
# This will be the Agent class we run in the 1v1. We've started you off with a functioning RL agent (`SB3Agent(Agent)`) and if-statement agent (`BasedAgent(Agent)`). Feel free to copy either to `SubmittedAgent(Agent)` then begin modifying.
# 
# Requirements:
# - Your submission **MUST** be of type `SubmittedAgent(Agent)`
# - Any instantiated classes **MUST** be defined within and below this code block.
# 
# Remember, your agent can be either machine learning, OR if-statement based. I've seen many successful agents arising purely from if-statements - give them a shot as well, if ML is too complicated at first!!
# 
# Also PLEASE ask us questions in the Discord server if any of the API is confusing. We'd be more than happy to clarify and get the team on the right track.
# Requirements:
# - **DO NOT** import any modules beyond the following code block. They will not be parsed and may cause your submission to fail validation.
# - Only write imports that have not been used above this code block
# - Only write imports that are from libraries listed here
# We're using PPO by default, but feel free to experiment with other Stable-Baselines 3 algorithms!

import os
import gdown
from typing import Optional
from environment.agent import Agent
from stable_baselines3 import PPO, A2C # Sample RL Algo imports
from sb3_contrib import RecurrentPPO # Importing an LSTM
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MLPExtractor(BaseFeaturesExtractor):
    '''
    Class that defines an MLP Base Features Extractor
    '''
    def __init__(self, observation_space, features_dim: int = 384, hidden_dim: int = 384):
        super(MLPExtractor, self).__init__(observation_space, features_dim)
        self.model = nn.Linear(observation_space.shape[0], hidden_dim - observation_space.shape[0], dtype=torch.float32)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.cat((torch.nn.functional.leaky_relu(self.model(obs)), obs), axis=1)
    
    @classmethod
    def get_policy_kwargs(cls, features_dim: int = 384, hidden_dim: int = 384) -> dict:
        return dict(
            net_arch=dict(
                pi=[384, 512, 256, 128, 64],  # Deep policy network
                vf=[384, 512, 256, 128, 64],  # Deep value network
            ),
            activation_fn=torch.nn.SiLU,
            features_extractor_class=cls,
            features_extractor_kwargs=dict(features_dim=features_dim, hidden_dim=hidden_dim)
        )
    
class SubmittedAgent(Agent):
    '''
    Input the **file_path** to your agent here for submission!
    '''
    def __init__(
        self,
        file_path: Optional[str] = None,
    ):
        self.move_to_p1 = 0.0
        self.move_to_p2 = 0.0
        self.attack_max = 0.0
        self.state_ctr = 0
        self.prev_player_state = None
        super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = PPO("MlpPolicy", self.env, verbose=0)
            del self.env
        else:
            self.model = PPO.load(self.file_path, custom_objects={"policy_kwargs": MLPExtractor.get_policy_kwargs()})
            

    def _gdown(self) -> str:
        data_path = "rl-model-2ab96390c7db.zip"
        if not os.path.isfile(data_path):
            print(f"Downloading {data_path}...")
            # Place a link to your PUBLIC model data here. This is where we will download it from on the tournament server.
            url = "https://drive.google.com/file/d/1g_aBhCYDgnBFreBbh9ClniUr_QM-6Gvc/view?usp=sharing"
            gdown.download(url, output=data_path, fuzzy=True)
        return data_path
    
    def patch_obs(self, obs):
        dt = 1 / 30
        new_obs = []
        for player in range(2):
            curr = obs[player*32:(player+1)*32]
            # future pos first order
            new_obs.append(curr[0:2] + curr[2:4]*dt)
            # past pos first order
            new_obs.append(curr[0:2] - curr[2:4]*dt)
            # xy pos, xy vel, right, grounded, air, jumps left
            new_obs.append(curr[0:8])
            # state one hot
            new_obs.append(np.eye(13)[int(curr[8])] if 0.0 <= curr[8] < 13 else np.zeros(13))
            # recoveries left, dodge timer, stun frames, damage, stocks
            new_obs.append(curr[9:14])
            # move type
            new_obs.append(np.eye(12)[int(curr[14])] if 0.0 <= curr[14] < 12 else np.zeros(12))
            # curr weapon
            new_obs.append(np.eye(3)[int(curr[15])] if 0.0 <= curr[15] < 3 else np.zeros(3))
            spawners = []
            num_inactive = 0
            for i in range(4):
                active = not (curr[16 + 3*i:16 + 3*i+2] == 0).all()
                if not active:
                    num_inactive += 1
                    continue
                curr_spawner = []
                # spawn abs
                curr_spawner.append(curr[16 + 3*i:16 + 3*i + 2])
                # spawn dx dy
                curr_spawner.append(curr[16 + 3*i:16 + 3*i + 2] - curr[0:2])
                # spawn weapon
                curr_spawner.append(np.eye(3)[int(curr[16 + 3*i+2])] if 0.0 <= curr[16 + 3*i+2] < 3 else np.zeros(3))
                # spawner active
                curr_spawner.append(np.array([active]))
                spawners.append(curr_spawner)
            spawners.sort(key=lambda curr_spawner:-np.linalg.norm(curr_spawner[1]))
            new_obs.extend([j for sub in spawners for j in sub])
            for i in range(num_inactive):
                new_obs.append(np.zeros(8))
            spawner_active = num_inactive < 4
            # absolute position of platform
            new_obs.append(curr[28:29]+1.25)
            new_obs.append(curr[28:29]-1.25)
            new_obs.append(curr[29:30])
            new_obs.append(curr[30:32])
            # relative pos to platform
            rel_plat = curr[28:30]-curr[0:2]
            new_obs.append(rel_plat)
            new_obs.append(curr[30:32]-curr[2:4])
            below_platform = 1.0 if curr[29] > curr[1] else 0.0
            plat_angle = np.atan2(rel_plat[0], rel_plat[1])
            new_obs.append(np.array([below_platform, plat_angle]))
        # dx dy player
        dx_dy = obs[32:34]-obs[0:2]
        new_obs.append(dx_dy)
        # dvx dvy player
        new_obs.append(obs[34:36]-obs[2:4])
        # angle and policy
        new_obs.append(np.array([np.atan2(dx_dy[0], dx_dy[1]), dx_dy[0] > 0, self.move_to_p1, self.move_to_p2, self.attack_max, self.state_ctr, spawner_active]))
        obs = np.concat(new_obs)
        return obs
    
    def update_policy(self, raw_obs, reset=False):
        player_state = self.obs_helper.get_section(raw_obs, "player_state")
        if reset or player_state != self.prev_player_state:
            self.state_ctr = 0
            self.prev_player_state = player_state
        else:
            self.state_ctr += 1
        self.move_to_p1 = 0.0
        self.move_to_p2 = 0.0
        self.attack_max = 0.0
        x_pos, y_pos = raw_obs[0:2]
        op_x_pos, op_y_pos = raw_obs[32:34]
        if op_x_pos < -2.0:
            if not (-5.0 < x_pos < -1.5 and y_pos <= 2.85):
                self.move_to_p1 = 1.0
        elif 2.0 < op_x_pos:
            if not (2.5 < x_pos < 5.0 and y_pos <= 0.85):
                self.move_to_p2 = 1.0

        if not self.move_to_p1 and not self.move_to_p2 and ((-5.0 < op_x_pos < -1.5 and op_y_pos <= 2.85) or (2.5 < op_x_pos < 5.0 and op_y_pos <= 0.85)):
            self.attack_max = 1.0
        

    def predict(self, obs):
        self.update_policy(obs)
        obs = self.patch_obs(obs)
        action, _ = self.model.predict(obs)
        
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    # If modifying the number of models (or training in general), modify this
    def learn(self, env, total_timesteps, log_interval: int = 4):
        self.model.set_env(env)
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)