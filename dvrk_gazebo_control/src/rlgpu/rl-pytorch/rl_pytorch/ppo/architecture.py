import numpy as np

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import pickle
import sys
sys.path.append("/home/baothach/dvrk_shape_servo/src/dvrk_env/dvrk_gazebo_control/src/rlgpu/rl-pytorch/rl_pytorch/ppo")
from deformernet import DeformerNet


class DeformerNetActorCritic(nn.Module):

    def __init__(self, obs_shape, states_shape, actions_shape, initial_std, model_cfg, asymmetric=False):
        super(DeformerNetActorCritic, self).__init__()

        self.asymmetric = asymmetric

        # Policy
        self.num_points = 1024
        self.actor = DeformerNet(out_dim = 3, in_num_points = self.num_points, normal_channel=False)

        # Value function
        self.critic = DeformerNet(out_dim = 1, in_num_points = self.num_points, normal_channel=False)

        # print(self.actor)
        # print(self.critic)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        # actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        # actor_weights.append(0.01)
        # critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        # critic_weights.append(1.0)
        # self.init_weights(self.actor, actor_weights)
        # self.init_weights(self.critic, critic_weights)
        self.actor.apply(weights_init)
        self.critic.apply(weights_init)


        # self.observations = torch.randn(1, 3, 512).to(torch.device("cuda"))
        with open('/home/baothach/shape_servo_data/generalization/surgical_setup/goal_data/sample 1.pickle', 'rb') as handle:
            data = pickle.load(handle)
        self.goal_pc = torch.from_numpy(data["partial pc"]).permute(1,0).unsqueeze(0).to(torch.device("cuda"))

        # print("******self.goal_pc.shape", self.goal_pc.shape)


    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def forward(self):
        raise NotImplementedError

    def act(self, observations, states):
        
        goal_pc = self.goal_pc.repeat(observations.shape[0], 1, 1)
        actions_mean = self.actor(observations, goal_pc)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        if self.asymmetric:
            value = self.critic(states)
        else:
            value = self.critic(observations, goal_pc)

        return actions.detach(), actions_log_prob.detach(), value.detach(), actions_mean.detach(), self.log_std.repeat(actions_mean.shape[0], 1).detach()

    def act_inference(self, observations):
        actions_mean = self.actor(observations, self.goal_pc)
        return actions_mean

    def evaluate(self, observations, states, actions):
        goal_pc = self.goal_pc.repeat(observations.shape[0], 1, 1)
        actions_mean = self.actor(observations, goal_pc)

        actions_mean = self.actor(observations, goal_pc)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        if self.asymmetric:
            value = self.critic(states)
        else:
            value = self.critic(observations, goal_pc)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)