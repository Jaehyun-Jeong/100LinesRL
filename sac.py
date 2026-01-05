import gymnasium as gym, torch as T, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Normal
from types import SimpleNamespace

lr_pi, lr_q, lr_alpha, gamma, tau = 0.0005, 0.001, 0.001, 0.98, 0.01
batch_size, buffer_limit, init_alpha, target_entropy = 32, 50000, 0.01, -1.0

class ReplayBuffer:
    def __init__(self):
        self.s, self.a, self.r, self.sp, self.d = [T.empty(0, i) for i in [3, 1, 1, 3, 1]]  # state, action, reward, s prime, done

    def push(self, transition):
        s, a, r, sp, d = transition
        self.s, self.a, self.r, self.sp, self.d = [
            T.cat((T.tensor(new).unsqueeze(0), buffer), 0)[:buffer_limit]
            for new, buffer in zip([s, [a], [r], sp, [d]], [self.s, self.a, self.r, self.sp, self.d])
        ]

    def sample(self, n):
        idx = T.randperm(len(self))[:n]
        return SimpleNamespace(s=self.s[idx], a=self.a[idx], r=self.r[idx], sp=self.sp[idx], done=self.d[idx])

    def __len__(self): return self.s.shape[0]

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(3, 128), nn.ReLU())
        self.mu_head, self.std_head = nn.Linear(128, 1), nn.Sequential(nn.Linear(128, 1), nn.Softplus())
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr_pi)
        self.log_alpha = T.tensor(init_alpha).log().requires_grad_()
        self.log_alpha_optimizer = T.optim.Adam([self.log_alpha], lr=lr_alpha)

    def forward(self, x):
        bb = self.backbone(x)
        dist = Normal(self.mu_head(bb), self.std_head(bb))
        action = dist.rsample()
        log_prob = dist.log_prob(action) - T.log(1 - T.tanh(action).pow(2) + 1e-7)
        return T.tanh(action), log_prob

    def train_net(self, q1, q2, mb):  # mb: mini-batch
        a, log_prob = self.forward(mb.s)
        loss = -(T.min(q1(mb.s, a), q2(mb.s, a)) - self.log_alpha.exp() * log_prob)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_prob + target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr_q)

    def forward(self, s, a): return self.net(T.cat([s, a], dim=1))

    def train_net(self, target, mb):  # mb: mini-batch
        loss = F.smooth_l1_loss(self.forward(mb.s, mb.a), target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self, tgt):
        for pt, p in zip(tgt.parameters(), self.parameters()): pt.data.copy_(pt.data * (1 - tau) + p.data * tau)

def calc_target(pi, q1_t, q2_t, mb):  # mb: mini-batch
    with T.no_grad():
        a_prime, log_prob = pi(mb.sp)
        return mb.r + gamma * mb.done * (T.min(q1_t(mb.sp, a_prime), q2_t(mb.sp, a_prime)) - pi.log_alpha.exp() * log_prob)

if __name__ == '__main__':
    env, memory = gym.make('Pendulum-v1'), ReplayBuffer()
    q1, q2, q1_t, q2_t, pi = QNet(), QNet(), QNet(), QNet(), PolicyNet()  # q1_t, q2_t: target Q
    [qt.load_state_dict(q.state_dict()) for qt, q in [(q1_t, q1), (q2_t, q2)]]
    score = 0.0
    for n_epi in range(10000):
        s, _ = env.reset()
        for _ in range(200):
            a, _ = pi(T.from_numpy(s).float())
            sp, r, done, truncated, _ = env.step([2.0 * a.item()])
            memory.push((s, a.item(), r/10.0, sp, 0. if done else 1.))
            score += r
            s = sp
            if done: break
        if len(memory) > 1000:
            for _ in range(20):
                mb = memory.sample(batch_size)
                td_target = calc_target(pi, q1_t, q2_t, mb)
                [q.train_net(td_target, mb) for q in [q1, q2]]
                pi.train_net(q1, q2, mb)
                [q.soft_update(q_t) for q, q_t in [(q1, q1_t), (q2, q2_t)]]
        if n_epi % 20 == 0 and n_epi != 0:  # 20 is print interval
            print(f"Episode: {n_epi}, Avg Score: {score/20:.1f}, Alpha: {pi.log_alpha.exp():.4f}")
            score = 0.0
    env.close()
