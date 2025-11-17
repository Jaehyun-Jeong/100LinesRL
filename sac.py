import gymnasium as gym
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

#Hyperparameters
lr_pi = 0.0005
lr_q = 0.001
init_alpha = 0.01
gamma = 0.98
batch_size = 32
buffer_limit = 50000
tau = 0.01 # for target network soft update
target_entropy = -1.0 # for automated alpha update
lr_alpha = 0.001  # for automated alpha update

class ReplayBuffer():
    def __init__(self):
        self.s_lst = T.empty(0, 3)
        self.a_lst = T.empty(0, 1)
        self.r_lst = T.empty(0, 1)
        self.s_prime_lst = T.empty(0, 3)
        self.done_lst = T.empty(0, 1)

    def push(self, transition):  # Tensor Queue
        s, a, r, s_prime, done = transition
        self.s_lst = T.cat((T.tensor(s).unsqueeze(0), self.s_lst), 0)[:buffer_limit]
        self.a_lst = T.cat((T.tensor([a]).unsqueeze(0), self.a_lst), 0)[:buffer_limit]
        self.r_lst = T.cat((T.tensor([r]).unsqueeze(0), self.r_lst), 0)[:buffer_limit]
        self.s_prime_lst = T.cat((T.tensor(s_prime).unsqueeze(0), self.s_prime_lst), 0)[:buffer_limit]
        self.done_lst = T.cat((T.tensor([0. if done else 1.]).unsqueeze(0), self.done_lst), 0)[:buffer_limit]

    def sample(self, n):
        idx = T.randperm(self.__len__())[:batch_size]

        return self.s_lst[idx], self.a_lst[idx], self.r_lst[idx], \
            self.s_prime_lst[idx], self.done_lst[idx]

    def __len__(self):
        return int(self.s_lst.shape[0])

class PolicyNet(nn.Module):
    def __init__(self, learning_rate):
        super(PolicyNet, self).__init__()
        self.backbone = nn.Sequential(nn.Linear(3, 128), nn.ReLU())
        self.mu_head = nn.Sequential(nn.Linear(128, 1))
        self.std_head = nn.Sequential(nn.Linear(128, 1), nn.Softplus())
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.log_alpha = T.tensor(init_alpha).detach().log()
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

    def forward(self, x):
        mu = self.mu_head(self.backbone(x))
        std = self.std_head(self.backbone(x))
        dist = Normal(mu, std)
        action = dist.rsample()
        real_action = T.tanh(action)
        real_log_prob = dist.log_prob(action) - T.log(1 - T.tanh(action).pow(2) + 1e-7)
        return real_action, real_log_prob

    def train_net(self, q1, q2, mini_batch):
        s, _, _, _, _ = mini_batch
        a, log_prob = self.forward(s)
        entropy = -self.log_alpha.exp() * log_prob

        q1_val, q2_val = q1(s, a), q2(s, a)
        q1_q2 = T.cat([q1_val, q2_val], dim=1)
        min_q = T.min(q1_q2, 1, keepdim=True)[0]

        loss = -min_q - entropy # for gradient ascent
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_prob + target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

class QNet(nn.Module):
    def __init__(self, learning_rate):
        super(QNet, self).__init__()
        self.s_backbone = nn.Sequential(nn.Linear(3, 64), nn.ReLU())
        self.a_backbone = nn.Sequential(nn.Linear(1, 64), nn.ReLU())
        self.q_head = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, a):
        cat = T.cat([self.s_backbone(x), self.a_backbone(a)], dim=1)
        q = self.q_head(cat)
        return q

    def train_net(self, target, mini_batch):
        s, a, r, s_prime, done = mini_batch
        loss = F.smooth_l1_loss(self.forward(s, a), target)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def soft_update(self, net_target):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

    @staticmethod
    @T.no_grad()
    def calc_target(pi, q1, q2, mini_batch):
        s, a, r, s_prime, done = mini_batch
        a_prime, log_prob = pi(s_prime)
        entropy = -pi.log_alpha.exp() * log_prob
        q1_val, q2_val = q1(s_prime, a_prime), q2(s_prime, a_prime)
        q1_q2 = T.cat([q1_val, q2_val], dim=1)
        min_q = T.min(q1_q2, 1, keepdim=True)[0]
        target = r + gamma * done * (min_q + entropy)

        return target

def main():
    env = gym.make('Pendulum-v1')
    memory = ReplayBuffer()
    q1, q2, q1_target, q2_target = QNet(lr_q), QNet(lr_q), QNet(lr_q), QNet(lr_q)
    pi = PolicyNet(lr_pi)

    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        s, _ = env.reset()
        done = False
        count = 0

        while count < 200 and not done:
            a, log_prob = pi(T.from_numpy(s).float())
            s_prime, r, done, truncated, info = env.step([2.0*a.item()])
            memory.push((s, a.item(), r/10.0, s_prime, done))
            score += r
            s = s_prime
            count += 1

        if len(memory) > 1000:
            for i in range(20):
                mini_batch = memory.sample(batch_size)
                td_target = QNet.calc_target(pi, q1_target, q2_target, mini_batch)
                q1.train_net(td_target, mini_batch)
                q2.train_net(td_target, mini_batch)
                pi.train_net(q1, q2, mini_batch)
                q1.soft_update(q1_target)
                q2.soft_update(q2_target)

        if n_epi % print_interval == 0 and n_epi != 0:
            print(f"# of episode :{n_epi}, avg score : {score/print_interval:.1f} alpha:{pi.log_alpha.exp():.4f}")
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()
