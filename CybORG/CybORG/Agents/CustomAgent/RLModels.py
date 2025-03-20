import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def act(self, state, memory, deterministic=False, full=False):
        action_probs = self.actor(torch.tensor(state).float())
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        # return list of probs
        if full:
            return action_probs

        # for training
        if not deterministic:
            memory.states.append(torch.tensor(state).float())
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)
            return action.item()

        # no sense following deterministic policy during training, so no memory needed
        else:
            max_actions = torch.argmax(action_probs, dim=1)
            return max_actions


    def evaluate(self, state, action):
        state_value = self.critic(state)

        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(state_value), dist_entropy
    

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_terminals = []
        self.logprobs = []

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.logprobs[:]


class PPOModel():
    def __init__(self, input_dim, action_space,lr=3e-4, betas=[0.9,0.99], gamma=0.99,k_epochs=4, eps_clip=0.2, restore= False, ckpt=None, training= True):
        
       
        
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.k_epochs = k_epochs
        self.eps_clip = eps_clip
        self.restore = restore
        self.ckpt = ckpt
        # self.deterministic = deterministic
        self.training = training

        self.memory = Memory()
        self.policy = ActorCritic(input_dim,action_space).to(device)

        self.set_initial_values(input_dim, action_space)



    def set_initial_values(self,input_dim, action_space):
        if self.restore:
            pretrained_model = torch.load(self.ckpt, map_location=lambda storage, loc:storage)
            self.policy.load_state_dict(pretrained_model)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)    

        self.old_policy = ActorCritic(input_dim, action_space).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.MSE_loss = nn.MSELoss()

    
    def end_episode(self):
        #reset stuff
        pass

    def train(self):
        rewards=[]
        discounted_reward = 0

        for reward, is_terminal in zip(reversed(self.memory.rewards),reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward= 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)


        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_states = torch.squeeze(torch.stack(self.memory.states).to(device)).detach()
        old_actions = torch.squeeze(torch.stack(self.memory.actions).to(device)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.memory.logprobs).to(device)).detach()


        for _ in range(self.k_epochs):
            logprobs, state_values, dist_entropy =  self.policy.evaluate(old_states, old_actions)

            ratios = torch.exp(logprobs - old_logprobs)

            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)* advantages

            actor_loss = torch.min(surr1, surr2)

            critic_loss = 0.5 * self.MSE_loss(rewards, state_values) - 0.01 * dist_entropy

            loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        self.old_policy.load_state_dict(self.policy.state_dict())

    def store(self, reward, done):
        self.memory.rewards.append(reward)
        self.memory.is_terminals.append(done)

    def clear_memory(self):
        self.memory.clear_memory()


    def get_action(self, observation, action_space=None):
        return self.old_policy.act(state=observation,memory=self.memory)
    
    def save_checkpoint(self, ckpt):
        torch.save(self.policy.state_dict(), ckpt)