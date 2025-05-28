import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# ==== ENVIRONMENT ====
class CaroEnv:
    def __init__(self, board_size=10, win_len=4):
        self.size = board_size
        self.win_len = win_len
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.done = False
        self.winner = 0
        return self._get_image()

    def _get_image(self):
        img = np.zeros((self.size, self.size), dtype=np.uint8)
        img[self.board == 0] = 127
        img[self.board == 1] = 255
        img[self.board == -1] = 0
        return np.expand_dims(img, axis=0).astype(np.float32) / 255.0

    def step(self, action):
        if self.done:
            raise ValueError("Game is over")
        x, y = divmod(action, self.size)
        if self.board[x, y] != 0:
            return self._get_image(), -10.0, True, {"illegal": True}
        self.board[x, y] = 1
        chain_len = self._longest_chain(1, x, y)
        reward = 0.1 * (chain_len - 1)
        if self._check_win(1, x, y):
            self.done = True
            self.winner = 1
            return self._get_image(), 10.0, True, {}
        opp = self._random_move()
        if opp is not None:
            ox, oy = opp
            self.board[ox, oy] = -1
            if self._check_win(-1, ox, oy):
                self.done = True
                self.winner = -1
                return self._get_image(), -10.0, True, {}
        if not np.any(self.board == 0):
            self.done = True
            self.winner = 0
            return self._get_image(), 0.0, True, {}
        reward -= 0.01
        return self._get_image(), reward, False, {}

    def _longest_chain(self, player, x, y):
        directions = [(1,0), (0,1), (1,1), (1,-1)]
        max_chain = 1
        for dx, dy in directions:
            count = 1
            for dir_ in [1, -1]:
                nx, ny = x + dx*dir_, y + dy*dir_
                while 0 <= nx < self.size and 0 <= ny < self.size and self.board[nx, ny] == player:
                    count += 1
                    nx += dx*dir_
                    ny += dy*dir_
            max_chain = max(max_chain, count)
        return max_chain

    def _check_win(self, player, x, y):
        directions = [(1,0), (0,1), (1,1), (1,-1)]
        for dx, dy in directions:
            count = 1
            for dir_ in [1, -1]:
                nx, ny = x + dx*dir_, y + dy*dir_
                while 0 <= nx < self.size and 0 <= ny < self.size and self.board[nx, ny] == player:
                    count += 1
                    nx += dx*dir_
                    ny += dy*dir_
            if count >= self.win_len:
                return True
        return False

    def _random_move(self):
        empties = np.argwhere(self.board == 0)
        return tuple(random.choice(empties)) if len(empties) > 0 else None

# ==== PPO NETWORK ====
class PPOPolicy(nn.Module):
    def __init__(self, board_size):
        super(PPOPolicy, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        conv_out_size = 64 * board_size * board_size
        self.fc_actor = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, board_size * board_size)
        )
        self.fc_critic = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        logits = self.fc_actor(x)
        value = self.fc_critic(x)
        return logits, value.squeeze(1)

# ==== GAE ====
def compute_gae(rewards, masks, values, gamma=0.99, lam=0.95):
    values = values + [0]
    gae, returns = 0, []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step+1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        returns.insert(0, gae + values[step])
    advantages = np.array(returns) - np.array(values[:-1])
    return np.array(returns), advantages

# ==== PPO UPDATE ====
def ppo_update(policy, optimizer, states, actions, old_log_probs, returns, advantages,
               clip_eps=0.2, value_loss_coef=0.5, entropy_coef=0.01, batch_size=64, epochs=4):
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions)
    old_log_probs = torch.tensor(old_log_probs)
    returns = torch.tensor(returns, dtype=torch.float32)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_loss, total_entropy = 0, 0
    dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, returns, advantages)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        for s, a, old_lp, ret, adv in loader:
            logits, value = policy(s)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(a)
            entropy = dist.entropy().mean()

            ratio = torch.exp(log_probs - old_lp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(value, ret)
            loss = actor_loss + value_loss_coef * critic_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_entropy += entropy.item()

    return total_loss / len(loader), total_entropy / len(loader)

# ==== TRAINING LOOP ====
def train_ppo(env, episodes=5000, batch_size=64, gamma=0.99, lr=2.5e-4,
              update_timestep=2000, clip_eps=0.2, value_loss_coef=0.5,
              entropy_coef=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PPOPolicy(env.size).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    memory = {k: [] for k in ["states", "actions", "log_probs", "rewards", "masks", "values"]}
    rewards_history, loss_history, entropy_history = [], [], []

    timestep = 0

    for ep in range(1, episodes + 1):
        state = env.reset()
        ep_reward = 0
        done = False

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            logits, value = policy(state_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, done, info = env.step(action.item())
            if info.get("illegal"):
                reward = -10.0
                done = True

            memory["states"].append(state)
            memory["actions"].append(action.item())
            memory["log_probs"].append(log_prob.item())
            memory["rewards"].append(reward)
            memory["masks"].append(0 if done else 1)
            memory["values"].append(value.item())

            state = next_state
            ep_reward += reward
            timestep += 1

            if timestep % update_timestep == 0:
                returns, advantages = compute_gae(
                    memory["rewards"], memory["masks"], memory["values"], gamma
                )
                loss, entropy = ppo_update(policy, optimizer,
                                           memory["states"],
                                           memory["actions"],
                                           memory["log_probs"],
                                           returns,
                                           advantages,
                                           clip_eps,
                                           value_loss_coef,
                                           entropy_coef,
                                           batch_size)

                loss_history.append(loss)
                entropy_history.append(entropy)
                memory = {k: [] for k in memory}

        rewards_history.append(ep_reward)
        if len(loss_history) < len(rewards_history):
            loss_history.append(loss_history[-1] if loss_history else 0)
            entropy_history.append(entropy_history[-1] if entropy_history else 0)

        print(f"Episode {ep}/{episodes} - Reward: {ep_reward:.2f}, Loss: {loss_history[-1]:.4f}, Entropy: {entropy_history[-1]:.4f}")

    # Plot
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.plot(rewards_history)
    plt.title("Total Reward")
    plt.subplot(1,3,2)
    plt.plot(loss_history)
    plt.title("Average Loss")
    plt.subplot(1,3,3)
    plt.plot(entropy_history)
    plt.title("Policy Entropy")
    plt.tight_layout()
    plt.show()

    os.makedirs("models", exist_ok=True)
    torch.save(policy.state_dict(), "models/caro_ppo.pt")
    print("Training complete. Model saved to models/caro_ppo.pt")

# ==== MAIN ====
if __name__ == "__main__":
    env = CaroEnv(board_size=10, win_len=4)
    train_ppo(env, episodes=5000, batch_size=64)
