import gymnasium as gym
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
lmbda = 0.95
eps_clip = 0.1
K_epoch = 3
T_horizon = 20

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()

        self.reset_batch()
        self.backbone = nn.Sequential(nn.Linear(4, 256), nn.ReLU())
        self.pi_head  = nn.Sequential(nn.Linear(256, 2), nn.Softmax(dim=1))
        self.v_head   = nn.Sequential(nn.Linear(256, 1))
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x): return self.pi_head(self.backbone(x))
    def v(self, x): return self.v_head(self.backbone(x))

    def reset_batch(self):
        self.s = T.empty(0, 4, dtype=T.float)
        self.a = T.empty(0, 1, dtype=T.int)
        self.r = T.empty(0, 1, dtype=T.float)
        self.s_prime = T.empty(0, 4, dtype=T.float)
        self.prob_a = T.empty(0, 1, dtype=T.float)
        self.done = T.empty(0, 1, dtype=T.float)

    def push(self, transition):
        s, a, r, s_prime, prob_a, done = transition
        self.s = T.cat((self.s, T.tensor(s).unsqueeze(0)), 0)
        self.a = T.cat((self.a, T.tensor([a]).unsqueeze(0)), 0)
        self.r = T.cat((self.r, T.tensor([r]).unsqueeze(0)), 0)
        self.s_prime = T.cat((self.s_prime, T.tensor(s_prime).unsqueeze(0)), 0)
        self.prob_a = T.cat((self.prob_a, T.tensor([prob_a]).unsqueeze(0)), 0)
        self.done = T.cat((self.done, T.tensor([0 if done else 1]).unsqueeze(0)), 0)

    def train_net(self):

        for i in range(K_epoch):
            td_target = self.r + gamma * self.v(self.s_prime) * self.done
            delta = td_target - self.v(self.s)

            advantage_tensor = T.empty(0, 1, dtype=T.float)
            advantage = 0.0
            for delta_t in delta.squeeze(0).flip(0):
                with T.no_grad(): advantage = gamma * lmbda * advantage + delta_t
                advantage_tensor = T.cat((T.tensor([[advantage]]), advantage_tensor), 0)

            pi_a = self.pi(self.s).gather(1, self.a)
            ratio = T.exp(T.log(pi_a) - T.log(self.prob_a))

            surr1 = ratio * advantage
            surr2 = T.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -T.min(surr1, surr2) + F.smooth_l1_loss(self.v(self.s) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        self.reset_batch()

def main():
    env = gym.make('CartPole-v1')
    model = PPO()
    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        s, _ = env.reset()
        done = False
        while not done:
            for t in range(T_horizon):
                prob = model.pi(T.from_numpy(s.reshape(1, len(s))).float())[0]
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, truncated, info = env.step(a)
                model.push((s, a, r/100.0, s_prime, prob[a].item(), done))
                s = s_prime

                score += r
                if done: break

            model.train_net()

        if n_epi%print_interval==0 and n_epi!=0:
            print(f"# of episode :{n_epi}, avg score : {score/print_interval:.1f}")
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()
