import gymnasium as gym
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensordict import TensorDict
from torchrl.data import ReplayBuffer, LazyMemmapStorage, RandomSampler

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32

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
        batch = memory.sample(batch_size).apply(
            lambda x: x.unsqueeze(1) if x.ndim == 1 else x
        )

        q_out = q(batch["s"])
        q_a = q_out.gather(1, batch["a"])
        max_q_prime = q_target(batch["s_prime"]).max(1, True)[0]
        target = batch["r"] + gamma * max_q_prime * batch["done"]
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
    memory = ReplayBuffer(
        storage=LazyMemmapStorage(buffer_limit),
        sampler=RandomSampler()
    )
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
            memory.add(TensorDict({
                "s": s, "s_prime": s_prime, "r": r/100.0, "a": a, "done": done_mask
            }))
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
