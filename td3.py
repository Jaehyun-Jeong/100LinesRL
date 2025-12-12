import gymnasium as gym, torch as T, torch.nn as nn, torch.nn.functional as F

seed, learning_starts, total_timesteps, buffer_limit, batch_size, learning_rate, gamma, policy_noise, policy_frequency, noise_clip, action_scale, exploration_noise, tau = 0, 25e3, int(1e6), int(1e6), 256, 3e-4, 0.005, 0.1, 2, 0.5, T.tensor(2, dtype=T.float32), 0.1, 0.005

class ReplayBuffer:
    def __init__(self):
        self.s, self.a, self.r, self.sp, self.d = [T.empty(0, i) for i in [3, 1, 1, 3, 1]]  # state, action, reward, s prime, done

    def push(self, transition):
        s, a, r, sp, d = transition
        self.s, self.a, self.r, self.sp, self.d = [
            T.cat((T.Tensor(new).unsqueeze(0), buffer), 0)[:buffer_limit]
            for new, buffer in zip([s, a, [r], sp, [d]], [self.s, self.a, self.r, self.sp, self.d])
        ]

    def sample(self, n):
        idx = T.randperm(len(self))[:n]
        return {'s': self.s[idx], 'a': self.a[idx], 'r': self.r[idx], 'sp': self.sp[idx], 'done': self.d[idx]}

    def __len__(self): return self.s.shape[0]

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(4, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x, a): return self.fc(T.cat([x, a], 1))

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_mu = nn.Sequential(nn.Linear(3, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x): return T.tanh(self.fc_mu(x))

if __name__ == "__main__":

    env = gym.make('Pendulum-v1')
    actor = PolicyNet()
    qf1, qf1_target, qf2, qf2_target = [QNet() for _ in range(4)]
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    actor_target = PolicyNet()
    actor_target.load_state_dict(actor.state_dict())
    q_optimizer = T.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=learning_rate)
    actor_optimizer = T.optim.Adam(list(actor.parameters()), lr=learning_rate)
    rb = ReplayBuffer()

    obs, _ = env.reset(seed=seed)
    score = 0.
    for global_step in range(total_timesteps):
        if global_step < learning_starts:
            action = env.action_space.sample()
        else:
            with T.no_grad():
                action = actor(T.Tensor(obs))
                action += T.normal(0, action_scale * exploration_noise)  # Smoothing q funciton
                action = action.cpu().numpy().clip(env.action_space.low, env.action_space.high)

        next_obs, reward, done, _, _ = env.step(action)
        real_next_obs = next_obs.copy()
        score += reward
        rb.push((real_next_obs, action, reward, obs, done))
        obs = next_obs

        if global_step > learning_starts:
            data = rb.sample(batch_size)
            with T.no_grad():
                clipped_noise = (T.randn_like(data['a']) * policy_noise).clamp(
                    -noise_clip, noise_clip
                ) * action_scale
                next_state_action = (actor_target(data['s']) + clipped_noise).clamp(
                    env.action_space.low[0], env.action_space.high[0]
                )
                qf1_next_target, qf2_next_target = [q_target(data['s'], next_state_action) for q_target in [qf1_target, qf2_target]]
                min_qf_next_target = T.min(qf1_next_target, qf2_next_target)
                next_q_value = (data['r'] + gamma * (1 - data['done']) * min_qf_next_target).view(-1)
            qf1_a_value, qf2_a_value = [qf(data['sp'], data['a']).view(-1) for qf in [qf1, qf2]]
            qf_loss = F.mse_loss(qf1_a_value, next_q_value) + F.mse_loss(qf2_a_value, next_q_value)

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % policy_frequency == 0:  # Delayed Policy Update
                actor_loss = -qf1(data['sp'], actor(data['sp'])).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # Polyak averaging
                for nn, nn_target in zip([actor, qf1, qf2], [actor_target, qf1_target, qf2_target]):
                    for param, param_target in zip(nn.parameters(), nn_target.parameters()):
                        param_target.data.copy_(tau * param.data + (1 - tau) * param_target.data)

            if global_step % 20 == 0:
                print(f"# of step:{global_step}, avg score : {score/20:1f}")
                score = 0.0
    envs.close()
