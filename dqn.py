import gymnasium as gym
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32

class ReplayBuffer():
    def __init__(self):
        self.size = 0
        self.s_lst = T.empty(0, 4)
        self.a_lst = T.empty(0, 1)
        self.r_lst = T.empty(0, 1)
        self.s_prime_lst = T.empty(0, 4)
        self.done_mask_lst = T.empty(0, 1)

    def push(self, transition):  # Tensor Queue
        s, a, r, s_prime, done_mask = transition
        self.s_lst = T.cat((T.tensor(s).unsqueeze(0), self.s_lst), 0)[:buffer_limit]
        self.a_lst = T.cat((T.tensor([a]).unsqueeze(0), self.a_lst), 0)[:buffer_limit]
        self.r_lst = T.cat((T.tensor([r]).unsqueeze(0), self.r_lst), 0)[:buffer_limit]
        self.s_prime_lst = T.cat((T.tensor(s_prime).unsqueeze(0), self.s_prime_lst), 0)[:buffer_limit]
        self.done_mask_lst = T.cat((T.tensor([done_mask]).unsqueeze(0), self.done_mask_lst), 0)[:buffer_limit]
        if self.size < buffer_limit:
            self.size += 1

    def sample(self, n):
        idx = T.randperm(self.size)[:batch_size]

        return self.s_lst[idx], self.a_lst[idx], self.r_lst[idx], \
            self.s_prime_lst[idx], self.done_mask_lst[idx]

    def __len__(self):
        return int(self.s_lst.shape[0])

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def sample_action(self, obs, eps):
        out = self.forward(obs)
        coin = T.rand(())
        return (T.randint(0, 2, ()).item() if coin < eps else out.argmax().item())

def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1, True)[0]
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    env = gym.wrappers.NumpyToTorch(env)
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    print_interval = 20

    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(10000):
        eps = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1%
        s, _ = env.reset()
        done = False

        while not done:
            a = q.sample_action(s, eps)
            s_prime, r, done, _, _ = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.push((s, a, r/100.0, s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                break

        if len(memory) > 2000:
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:  # print interval is 20
            q_target.load_state_dict(q.state_dict())
            print(f"n_episode :{n_epi}, score : {score/print_interval:.1f}, " \
                  f"n_buffer : {len(memory)}, epsilon : {eps*100:.1f}%")
            score = 0.0

    env.close()
