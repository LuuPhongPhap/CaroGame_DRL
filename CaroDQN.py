import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import pygame
import time
import matplotlib.pyplot as plt

# --- Gomoku Environment with Visualization Toggle ---
class GomokuEnv:
    def __init__(self, size=10, win_len=5, cell_size=40, delay=0.3, visualize=False):
        self.size = size
        self.win_len = win_len
        self.cell_size = cell_size
        self.delay = delay
        self.visualize = visualize
        if self.visualize:
            pygame.init()
            self.window_size = self.size * self.cell_size
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Caro DQN Agent Training")
            self.font = pygame.font.SysFont(None, self.cell_size - 10)
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.current_player = 1
        self.done = False
        if self.visualize:
            self._draw_board()
        return self._get_state()

    def step(self, action):
        x, y = divmod(action, self.size)

        if self.board[x, y] != 0 or self.done:
            return self._get_state(), -10, True, {}

        self.board[x, y] = self.current_player
        reward, self.done = self._check_game(x, y)

        if self.done:
            if reward == 1.0:
                reward = 10 if self.current_player == 1 else -10
            elif reward == 0.5:
                reward = 0
            else:
                reward = -10
        else:
            reward = -0.1  # Khuyến khích kết thúc nhanh hơn
            if self._is_blocking_win(x, y, -self.current_player):
                reward += 10
            elif self._is_blocking_four(x, y, -self.current_player):
                reward += 0.5

        reward = np.clip(reward, -10, 10)

        self.current_player *= -1
        if self.visualize:
            self._draw_board()
            time.sleep(self.delay)

        return self._get_state(), reward, self.done, {}

    def legal_actions(self):
        return [i for i in range(self.size * self.size) if self.board.flat[i] == 0]

    def _get_state(self):
        return self.board.flatten().astype(np.float32)

    def _check_game(self, x, y):
        player = self.board[x, y]
        dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in dirs:
            count = 1
            for sign in [1, -1]:
                nx, ny = x, y
                while True:
                    nx += dx * sign
                    ny += dy * sign
                    if 0 <= nx < self.size and 0 <= ny < self.size and self.board[nx, ny] == player:
                        count += 1
                    else:
                        break
            if count >= self.win_len:
                return 1.0, True
        if np.all(self.board != 0):
            return 0.5, True
        return 0.0, False

    def _count_sequence(self, board, x, y, dx, dy, player):
        count = 0
        for sign in [1, -1]:
            nx, ny = x, y
            while True:
                nx += dx * sign
                ny += dy * sign
                if 0 <= nx < self.size and 0 <= ny < self.size and board[nx, ny] == player:
                    count += 1
                else:
                    break
        return count + 1

    def _is_blocking_win(self, x, y, opponent):
        temp_board = self.board.copy()
        temp_board[x, y] = 0
        dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in dirs:
            count = self._count_sequence(temp_board, x, y, dx, dy, opponent)
            if count >= self.win_len:
                return True
        return False

    def _is_blocking_four(self, x, y, opponent):
        temp_board = self.board.copy()
        temp_board[x, y] = 0
        dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in dirs:
            count = self._count_sequence(temp_board, x, y, dx, dy, opponent)
            if count == 4:
                return True
        return False

    def _draw_board(self):
        self.screen.fill((230, 185, 139))
        for i in range(self.size + 1):
            pygame.draw.line(self.screen, (0, 0, 0), (i * self.cell_size, 0), (i * self.cell_size, self.window_size))
            pygame.draw.line(self.screen, (0, 0, 0), (0, i * self.cell_size), (self.window_size, i * self.cell_size))
        for x in range(self.size):
            for y in range(self.size):
                val = self.board[x, y]
                if val != 0:
                    text = 'X' if val == 1 else 'O'
                    color = (255, 0, 0) if val == 1 else (0, 0, 255)
                    img = self.font.render(text, True, color)
                    rect = img.get_rect(center=(y * self.cell_size + self.cell_size // 2,
                                                x * self.cell_size + self.cell_size // 2))
                    self.screen.blit(img, rect)
        pygame.display.flip()

    def close(self):
        if self.visualize:
            pygame.display.quit()
            pygame.quit()

# --- DQN Network ---
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        return self.model(x)

# --- DQN Agent ---
class DQNAgent:
    def __init__(self, state_size, action_size, lr=1e-5, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=10000)
        self.batch_size = 64

        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_target()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.last_loss = 0.0

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state, legal_actions):
        if np.random.rand() < self.epsilon:
            return random.choice(legal_actions)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)[0].detach().numpy()
        legal_q_values = np.full_like(q_values, -np.inf)
        legal_q_values[legal_actions] = q_values[legal_actions]
        return np.argmax(legal_q_values)

    def remember(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            return

        batch = random.sample(self.memory, self.batch_size)
        s_batch, a_batch, r_batch, s__batch, d_batch = zip(*batch)

        s_batch = torch.FloatTensor(s_batch)
        s__batch = torch.FloatTensor(s__batch)
        r_batch = torch.FloatTensor(r_batch)
        a_batch = torch.LongTensor(a_batch)
        d_batch = torch.FloatTensor(d_batch)

        q_values = self.model(s_batch).gather(1, a_batch.unsqueeze(1)).squeeze()
        max_q_next = self.target_model(s__batch).max(1)[0]
        target = r_batch + self.gamma * max_q_next * (1 - d_batch)

        loss = self.loss_fn(q_values, target.detach())
        self.last_loss = min(loss.item(), 100.0)  # Giới hạn loss tối đa để tránh mất ổn định

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# --- Training Function ---
def train(env, agent, episodes=5000, update_target_freq=10):
    rewards = []
    epsilons = []
    losses = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            legal = env.legal_actions()
            action = agent.act(state, legal)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay()

        rewards.append(total_reward)
        epsilons.append(agent.epsilon)
        losses.append(agent.last_loss)

        if ep % update_target_freq == 0:
            agent.update_target()
        print(f"Episode {ep}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}, Loss: {agent.last_loss:.4f}")

    return rewards, epsilons, losses

# --- Plot Function ---
def plot_metrics(rewards, epsilons, losses):
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(rewards)
    axs[0].set_title("Total Rewards")
    axs[1].plot(epsilons)
    axs[1].set_title("Epsilon")
    axs[2].plot(losses)
    axs[2].set_title("Loss")
    for ax in axs:
        ax.set_xlabel("Episode")
    plt.tight_layout()
    plt.show()

# --- Main Execution ---
env = GomokuEnv(visualize=False)
state_size = env.size * env.size
action_size = state_size
agent = DQNAgent(state_size, action_size)

rewards, epsilons, losses = train(env, agent, episodes=5000)
plot_metrics(rewards, epsilons, losses)

env.close()
