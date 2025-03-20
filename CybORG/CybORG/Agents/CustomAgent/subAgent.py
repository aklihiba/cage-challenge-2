from CybORG.Agents.SimpleAgents import BaseAgent, SleepAgent
from CybORG.Agents.Wrappers import ChallengeWrapper
from RLModels import ActorCritic,PPOModel,Memory

import os

class SubAgent():
    def __init__(self, env:ChallengeWrapper, target_attacker_type= 0, training=True, restore=False,
                 lr=3e-4, betas=[0.9,0.99], gamma=0.99,k_epochs=4, eps_clip=0.2):
        # super().__init__()
        self.env = env
        self.action_space = env.get_action_space('Blue')
        self.input_dim = env.observation_space.shape[0]
        self.type = target_attacker_type # 0:Bline 1:meander 2:sleep
        # to work use cyborg = CybORG(path,'sim',agents=attacker_type)
        # env = ChallengeWrapper(env=cyborg, agent_name='Blue')
        # input_dim = env.observation_space.shape[0] = 52
        # action_space = env.get_action_space('Blue') = 145
        ckpt = None
        if restore and self.type <2:
            ckpt = self.get_ckpt_bline() if self.type==0 else self.get_ckpt_meander()
        
        
        self.RLModel = PPOModel(input_dim=self.input_dim,
                                action_space=self.action_space,
                                training=training,
                                restore=restore,
                                ckpt=ckpt,
                                lr=lr,
                                gamma=gamma,
                                betas=betas,
                                k_epochs=k_epochs,
                                eps_clip=eps_clip)
        self.set_initial_values()
   
    def get_ckpt_bline(self):
        return os.path.join(os.getcwd(),"Models","Bline","10000.pth")
        

    def get_ckpt_meander(self):
        return os.path.join(os.getcwd(),"Models","meander_train","100000.pth")
        
    
    def get_action(self, observation, action_space):
        # return super().get_action(observation, action_space)
        return self.RLModel.get_action(observation,action_space)
    
    def train(self,max_episodes, max_timesteps, update_timestep,ckpt_folder,print_interval=1, save_interval=10):
        running_reward , time_step = 0,0
        for episode in range(1,max_episodes+1): 
            state = self.env.reset()
            for i in range(max_timesteps):
               time_step += 1
               action = self.RLModel.get_action(state,self.action_space)
               state, reward, done, _ = self.env.step(action)

               if max_timesteps is not None and time_step >= max_timesteps:
                   done = True
               
               self.RLModel.store(reward, done)

               if time_step % update_timestep == 0:
                self.RLModel.train()
                self.RLModel.clear_memory()
                time_step = 0

                running_reward += reward
                
            self.end_episode()

            if episode % save_interval == 0:
                ckpt = os.path.join(ckpt_folder, '{}.pth'.format(episode))
                self.RLModel.save_checkpoint(ckpt)
                # torch.save(agent.policy.state_dict(), ckpt)
                print('Checkpoint saved')

            if episode % print_interval == 0:
                running_reward = int((running_reward / print_interval))
                print('Episode {} \t Avg reward: {}'.format(episode, running_reward))
                running_reward = 0

           


    def end_episode(self):
        self.RLModel.end_episode()
        # return super().end_episode()
    
    def set_initial_values(self):
        obs = self.env.reset()
        # return super().set_initial_values(self.action_space, obs)
