import gymnasium as gym, gymnasium.wrappers as wrappers, torch as T, torch.nn as nn, torch.nn.functional as F, torch.optim as optim, numpy as np, ppo_config as conf, shimmy
from torch.distributions import Categorical, Normal
from tqdm import tqdm
from plot import save_gif

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

def make_envs(env_id, n_envs):
    envs = gym.vector.SyncVectorEnv([lambda: gym.make(env_id) for _ in range(n_envs)])
    envs = wrappers.vector.FlattenObservation(envs)
    envs = wrappers.vector.RecordEpisodeStatistics(envs)
    envs = wrappers.vector.NormalizeObservation(envs)
    envs = wrappers.vector.TransformObservation(envs, lambda obs: np.clip(obs, -10, 10))
    envs = wrappers.vector.NormalizeReward(envs, gamma=conf.gamma)
    envs = wrappers.vector.TransformReward(envs, lambda r: np.clip(r, -10, 10))
    if (cts := isinstance(envs.single_action_space, gym.spaces.Box)):
        envs = wrappers.vector.RescaleAction(envs, -1.0, 1.0)
        envs = wrappers.vector.ClipAction(envs)
    return envs, cts, envs.single_observation_space.shape[0], envs.single_action_space.shape[0] if cts else envs.single_action_space.n

class PPO(nn.Module):
    def __init__(self, s_dim, a_dim, cts):
        super().__init__()
        self.cts, self.s_dim, self.a_dim = cts, s_dim, a_dim
        self.buf, self.p = [T.zeros(conf.T_horizon, conf.n_envs, i, device=conf.device) for i in [s_dim, a_dim if cts else 1, 1, s_dim, 1, 1]], 0

        self.pi_net, self.v_net = [nn.Sequential(
            layer_init(nn.Linear(s_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh()
        ) for _ in range(2)]
        self.mu_head = layer_init(nn.Linear(256, a_dim), std=0.01) if cts else None
        self.log_std = nn.Parameter(T.zeros(a_dim)) if cts else None
        self.pi_head = nn.Sequential(layer_init(nn.Linear(256, a_dim), std=0.01)) if not cts else None
        self.v_head = layer_init(nn.Linear(256, 1), std=1.0)
        self.to(conf.device)
        self.optimizer = optim.Adam(self.parameters(), lr=conf.lr)

    def pi(self, x, a=None):
        if self.cts:
            dist = Normal(self.mu_head(self.pi_net(x)), T.exp(self.log_std))
            if a is None: a = dist.sample()
            return a, a.detach().cpu().numpy(), dist.log_prob(a).sum(dim=-1, keepdim=True), dist

        dist = Categorical(logits=self.pi_head(self.pi_net(x)))
        if a is None: a = dist.sample()
        return a, a.detach().cpu().numpy(), dist.log_prob(a.squeeze(-1).long() if a.dim() > 1 else a.long()).unsqueeze(-1), dist

    def v(self, x): return self.v_head(self.v_net(x))

    def push(self, transition):
        for val, buf in zip(transition, self.buf): buf[self.p].copy_(T.as_tensor(val, device=conf.device).view_as(buf[self.p]))
        self.p = (self.p + 1) % conf.T_horizon

    def train_net(self):
        s, a, r, sp, log_p, not_d = self.buf
        with T.no_grad():  # Calculate GAE
            vals, next_vals = self.v(s), self.v(sp)
            deltas = r + conf.gamma * next_vals * not_d - vals
            advantages, gae = T.zeros_like(r), 0
            for t in reversed(range(len(r))):
                advantages[t] = gae = deltas[t] + conf.gamma * conf.lmbda * not_d[t] * gae
            returns = advantages + vals

        s, a, log_p, advantages, returns = [x.flatten(0, 1) for x in (s, a, log_p, advantages, returns)]
        for _ in range(conf.K_epoch):  # Update Parameters
            for inds in T.randperm(len(s), device=conf.device).split(conf.mb_size):
                mb_adv = advantages[inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                _, _, log_prob_a, dist = self.pi(s[inds], a[inds])
                ratio = T.exp(log_prob_a - log_p[inds])

                loss = -T.min(ratio * mb_adv, T.clamp(ratio, 1-conf.eps_clip, 1+conf.eps_clip) * mb_adv).mean() \
                    + 0.5 * conf.vf_coef * F.mse_loss(self.v(s[inds]), returns[inds]) \
                    - conf.ent_coef * (dist.entropy().sum(-1) if self.cts else dist.entropy()).mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), conf.max_grad_norm)
                self.optimizer.step()

if __name__ == '__main__':
    envs, cts, s_dim, a_dim = make_envs(conf.env_name, conf.n_envs)
    model, total_step = PPO(s_dim, a_dim, cts), conf.max_timesteps // conf.n_envs
    s, score, n_epi, print_interval = envs.reset()[0], 0.0, 0, 20
    best_score = -np.inf
    for n_step in tqdm(range(total_step), unit_scale=conf.n_envs, unit="step"):
        model.optimizer.param_groups[0]['lr'] = lr = conf.lr * (1 - n_step / total_step)
        a, a_in, log_prob, _ = model.pi(T.from_numpy(s).float().to(conf.device))
        sp, r, done, trunc, info = envs.step(a_in)
        model.push((s, a, r.copy(), sp, log_prob.detach(), 1. - (done | trunc)))
        s = sp
        if "episode" in info and "_episode" in info:
            for i in np.where(info["_episode"])[0]:
                score += float(np.array(info["episode"]["r"][i]).item())
                if (n_epi := n_epi + 1) % print_interval == 0:
                    if best_score < (score / print_interval):
                        best_score = (score / print_interval)
                        save_gif(model, envs, conf.env_name, filename=f"ppo_{conf.env_name.split('/')[-1]}_step_{n_step}.gif")
                    tqdm.write(f"step {(n_step+1)*conf.n_envs} episode {n_epi} avg score {score/print_interval:.1f} lr {lr:.6f}")
                    score = 0.0
        if (n_step+1) % conf.T_horizon == 0: model.train_net()
    envs.close()
