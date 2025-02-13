import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random
import math
from tqdm import trange

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
print(device)   

# Look at past obs and actions when generating hidden state
# Prioritized Experience Replay
# Chance temperature over time
# We also scale the gradient at the start of the dynamics function by 1/2. This ensures that the total gradient applied to the dynamics function stays constant.

# %%
class CartPole: 
    def __init__(self, render=False):
        self.env = gym.make('CartPole-v1', render_mode='human' if render else 'rgb_array')
        self.action_size = self.env.action_space.n

    def __repr__(self):
        return 'CartPole'

    def get_initial_state(self):
        observation, info = self.env.reset()
        valid_locations = self.action_size
        reward = 0
        is_terminal = False
        return observation, valid_locations, reward, is_terminal

    def step(self, action):
        observation, reward, is_terminal, _, _ = self.env.step(action)
        valid_locations = self.action_size
        if is_terminal:
            reward = 0
        return observation, valid_locations, reward, is_terminal

    def get_canonical_state(self, hidden_state, player):
        return hidden_state

    def get_encoded_observation(self, observation):
        return observation.copy()

    def get_opponent_player(self, player):
        return player

    def get_opponent_value(self, value):
        return value

# %%
class ReplayBuffer:
    def __init__(self, args, game):
        self.memory = []
        self.trajectories = []
        self.args = args
        self.game = game

    def __len__(self):
        return len(self.trajectories)

    def empty(self):
        self.memory = []
        self.trajectories = []

    def build_trajectories(self):
        for i in range(len(self.memory)):
            observation, action, policy, reward, _, game_idx = self.memory[i]
            policy_list, action_list, value_list, reward_list = [policy], [action], [], [reward]

            # value bootstrap for N-step return
            # value starts at root value n steps ahead
            if i + self.args['N'] + 1 < len(self.memory) and self.memory[i + self.args['N'] + 1][5] == game_idx:
                value = self.memory[i + self.args['N'] + 1][4] * self.args['gamma'] ** self.args['N']
            else:
                value = 0
            # add discounted rewards until end of game or N steps
            for n in range(2, self.args['N'] + 2):
                if i + n < len(self.memory) and self.memory[i + n][5] == game_idx:
                    _, _, _, reward, _, _ = self.memory[i + n]
                    value += reward * self.args['gamma'] ** (n - 2)
                else:
                    break
            value_list.append(value)

            for k in range(1, self.args['K'] + 1):
                if i + k < len(self.memory) and self.memory[i + k][5] == game_idx:
                    _, action, policy, reward, _, _ = self.memory[i + k]
                    action_list.append(action)
                    policy_list.append(policy)
                    reward_list.append(reward)

                    if i + k + self.args['N'] + 1 < len(self.memory) and self.memory[i + k + self.args['N'] + 1][5] == game_idx:
                        value = self.memory[i + k + self.args['N'] + 1][4] * self.args['gamma'] ** self.args['N']
                    else:
                        value = 0
                    for n in range(2, self.args['N'] + 2):
                        if i + k + n < len(self.memory) and self.memory[i + k + n][5] == game_idx:
                            _, _, _, reward, _, _ = self.memory[i + k + n]
                            value += reward * self.args['gamma'] ** (n - 2)
                        else:
                            break
                    value_list.append(value)

                else:
                    action_list.append(np.random.choice(self.game.action_size))
                    policy_list.append(np.ones(self.game.action_size) / self.game.action_size)
                    value_list.append(0)
                    reward_list.append(0)

            policy_list = np.stack(policy_list)
            self.trajectories.append((observation, action_list, policy_list, value_list, reward_list))


# %%
class MinMaxStats:
    def __init__(self, known_bounds):
        self.maximum = known_bounds['max'] if known_bounds else -float('inf')
        self.minimum = known_bounds['min'] if known_bounds else float('inf')

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

class Node:
    def __init__(self, state, reward, prior, muZero, args, game, parent=None, action_taken=None, visit_count=0):
        self.state = state
        self.reward = reward
        self.children = []
        self.parent = parent
        self.total_value = 0
        self.visit_count = visit_count # Should start at 1 for root node
        self.prior = prior
        self.muZero = muZero
        self.action_taken = action_taken
        self.args = args
        self.game = game

    @torch.no_grad()
    def expand(self, action_probs):
        actions = [a for a in range(self.game.action_size) if action_probs[a] > 0]
        expand_state = self.state.copy()
        expand_state = np.expand_dims(expand_state, axis=0).repeat(len(actions), axis=0)

        expand_state, reward = self.muZero.dynamics(
            torch.tensor(expand_state, dtype=torch.float32, device=self.muZero.device), actions)
        expand_state = expand_state.cpu().numpy()
        expand_state = self.game.get_canonical_state(expand_state, -1).copy()
        reward = self.muZero.inverse_reward_transform(reward).cpu().numpy().flatten()
        
        for i, a in enumerate(actions):
            child = Node(
                expand_state[i],
                reward[i],
                action_probs[a],
                self.muZero,
                self.args,
                self.game,
                parent=self,
                action_taken=a,
            )
            self.children.append(child)

    def backpropagate(self, value, minMaxStats):
        self.total_value += value
        self.visit_count += 1
        minMaxStats.update(self.value())
        if self.parent is not None:
            value = self.reward + self.args['gamma'] * self.game.get_opponent_value(value)
            self.parent.backpropagate(value, minMaxStats)

    def is_expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.total_value / self.visit_count

    def select_child(self, minMaxStats):
        best_score = -np.inf
        best_child = None

        for child in self.children:
            ucb_score = self.get_ucb_score(child, minMaxStats)
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child

        return best_child

    def get_ucb_score(self, child, minMaxStats):
        pb_c = math.log((self.visit_count + self.args["pb_c_base"] + 1) /
                  self.args["pb_c_base"]) + self.args["pb_c_init"]
        pb_c *= math.sqrt(self.visit_count) / (child.visit_count + 1)
        prior_score = pb_c * child.prior
        if child.visit_count > 0:
            value_score = minMaxStats.normalize(child.reward + self.args['gamma'] * child.value())
        else:
            # value_score = 0
            value_score = minMaxStats.normalize(child.reward)
        return prior_score + value_score

class MCTS:
    def __init__(self, muZero, game, args):
        self.muZero = muZero
        self.game = game
        self.args = args

    @torch.no_grad()
    def search(self, state, reward, available_actions):
        minMaxStats = MinMaxStats(self.args['known_bounds'])
        hidden_state = self.muZero.represent(
            torch.tensor(state, dtype=torch.float32, device=self.muZero.device).unsqueeze(0)
        )
        action_probs, _ = self.muZero.predict(hidden_state)
        hidden_state = hidden_state.cpu().numpy().squeeze(0)
        
        root = Node(hidden_state, reward, 0, self.muZero, self.args, self.game, visit_count=1)

        action_probs = torch.softmax(action_probs, dim=1).cpu().numpy().squeeze(0)
        action_probs = action_probs * (1 - self.args['dirichlet_epsilon']) + self.args['dirichlet_epsilon'] * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)
        action_probs *= available_actions
        action_probs /= np.sum(action_probs)

        root.expand(action_probs)

        for simulation in range(self.args['num_mcts_runs']):
            node = root

            while node.is_expanded():
                node = node.select_child(minMaxStats)

            action_probs, value = self.muZero.predict(
                torch.tensor(node.state, dtype=torch.float32, device=self.muZero.device).unsqueeze(0)
            )
            action_probs = torch.softmax(action_probs, dim=1).cpu().numpy().squeeze(0)
            value = self.muZero.inverse_value_transform(value).item()

            node.expand(action_probs)
            node.backpropagate(value, minMaxStats)

        return root


# %%
class MuZero(nn.Module):
    def __init__(self, game):
        super().__init__()
        self.game = game
        self.device = device

        self.value_support = DiscreteSupport(-20, 20)
        self.reward_support = DiscreteSupport(-5, 5)
        
        self.predictionFunction = PredictionFunction(self.game, self.value_support)
        self.dynamicsFunction = DynamicsFunction(self.reward_support)
        self.representationFunction = RepresentationFunction()

    def predict(self, hidden_state):
        return self.predictionFunction(hidden_state)

    def represent(self, observation):
        return self.representationFunction(observation)

    def dynamics(self, hidden_state, action):
        actionArr = torch.zeros((hidden_state.shape[0], 2), device=self.device, dtype=torch.float32)
        for i, a in enumerate(action):
            actionArr[i, a] = 1
        x = torch.hstack((hidden_state, actionArr))
        return self.dynamicsFunction(x)

    def inverse_value_transform(self, value):
        return self.inverse_scalar_transform(value, self.value_support)

    def inverse_reward_transform(self, reward):
        return self.inverse_scalar_transform(reward, self.reward_support)

    def inverse_scalar_transform(self, output, support):
        output_propbs = torch.softmax(output, dim=1)
        output_support = torch.ones(output_propbs.shape, dtype=torch.float32, device=self.device)
        output_support[:, :] = torch.tensor([x for x in support.range], device=self.device)
        scalar_output = (output_propbs * output_support).sum(dim=1, keepdim=True)

        epsilon = 0.001
        sign = torch.sign(scalar_output)
        inverse_scalar_output = sign * (((torch.sqrt(1 + 4 * epsilon * (torch.abs(scalar_output) + 1 + epsilon)) - 1) / (2 * epsilon)) ** 2 - 1)
        return inverse_scalar_output

    def scalar_transform(self, x):
        epsilon = 0.001
        sign = torch.sign(x)
        output = sign * (torch.sqrt(torch.abs(x) + 1) - 1 + epsilon * x)
        return output

    def value_phi(self, x):
        return self._phi(x, self.value_support.min, self.value_support.max, self.value_support.size)

    def reward_phi(self, x):
        return self._phi(x, self.reward_support.min, self.reward_support.max, self.reward_support.size)

    def _phi(self, x, min, max, set_size):
        x.clamp_(min, max)
        x_low = x.floor()
        x_high = x.ceil()
        p_high = (x - x_low)
        p_low = 1 - p_high

        target = torch.zeros(x.shape[0], x.shape[1], set_size).to(x.device)
        x_high_idx, x_low_idx = x_high - min, x_low - min
        target.scatter_(2, x_high_idx.long().unsqueeze(-1), p_high.unsqueeze(-1))
        target.scatter_(2, x_low_idx.long().unsqueeze(-1), p_low.unsqueeze(-1))
        return target

# Creates hidden state + reward based on old hidden state and action 
class DynamicsFunction(nn.Module):
    def __init__(self, reward_support):
        super().__init__()
        
        self.startBlock = nn.Sequential(
            nn.Linear(34, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
        )

        self.rewardBlock = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, reward_support.size),
        )

    def forward(self, x):
        x = self.startBlock(x)
        reward = self.rewardBlock(x)
        return x, reward
    
# Creates policy and value based on hidden state
class PredictionFunction(nn.Module):
    def __init__(self, game, value_support):
        super().__init__()
        self.game = game
        
        self.startBlock = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(64, self.game.action_size)
        )
        self.value_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, value_support.size), 
        )

    def forward(self, x):
        x = self.startBlock(x)
        p = self.policy_head(x)
        v = self.value_head(x)
        return p, v

# Creates initial hidden state based on observation | several observations
class RepresentationFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.startBlock = nn.Sequential(
            nn.Linear(4, 32),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.startBlock(x)
        return x

class DiscreteSupport:
    def __init__(self, min, max):
        assert min < max
        self.min = min
        self.max = max
        self.range = range(min, max + 1)
        self.size = len(self.range)

class Trainer:
    def __init__(self, muZero, optimizer, game, args):
        self.muZero = muZero
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(self.muZero, self.game, self.args)
        self.replayBuffer = ReplayBuffer(self.args, self.game)
        self.device = device

    def self_play(self, game_idx):
        game_memory = []
        player = 1
        observation, valid_locations, reward, is_terminal = self.game.get_initial_state()

        while True:
            encoded_observation = self.game.get_encoded_observation(observation)
            canonical_observation = self.game.get_canonical_state(encoded_observation, player).copy()
            root = self.mcts.search(canonical_observation, reward, valid_locations)

            action_probs = [0] * self.game.action_size
            for child in root.children:
                action_probs[child.action_taken] = child.visit_count
            action_probs /= np.sum(action_probs)

            # sample action from the mcts policy | based on temperature
            if self.args['temperature'] == 0:
                action = np.argmax(action_probs)
            elif self.args['temperature'] == float('inf'):
                action = np.random.choice([r for r in range(self.game.action_size) if action_probs[r] > 0])
            else:
                temperature_action_probs = action_probs ** (1 / self.args['temperature'])
                temperature_action_probs /= np.sum(temperature_action_probs)
                action = np.random.choice(len(temperature_action_probs), p=temperature_action_probs)

            game_memory.append((canonical_observation, action, player, action_probs, root.total_value / root.visit_count, reward))
            observation, valid_locations, reward, is_terminal = self.game.step(action)

            if is_terminal:
                return_memory = []
                for hist_state, hist_action, hist_player, hist_action_probs, hist_root_value, hist_reward in game_memory:
                    return_memory.append((
                        hist_state,
                        hist_action, 
                        hist_action_probs,
                        hist_reward,
                        hist_root_value,
                        game_idx,
                    ))
                return return_memory

            player = self.game.get_opponent_player(player)

    def train(self):
        random.shuffle(self.replayBuffer.trajectories)
        for batchIdx in range(0, len(self.replayBuffer), self.args['batch_size']): 
            state, action, policy, value, reward = list(zip(*self.replayBuffer.trajectories[batchIdx:batchIdx+self.args['batch_size']]))

            state = torch.tensor(np.stack(state), dtype=torch.float32, device=self.device)
            action = np.array(action)
            policy = torch.tensor(np.stack(policy), dtype=torch.float32, device=self.device)
            value = torch.tensor(np.array(value), dtype=torch.float32, device=self.device)
            reward = torch.tensor(np.array(reward), dtype=torch.float32, device=self.device)

            transformed_reward = self.muZero.scalar_transform(reward)
            phi_reward = self.muZero.reward_phi(transformed_reward)
            transformed_value = self.muZero.scalar_transform(value)
            phi_value = self.muZero.value_phi(transformed_value)

            state = self.muZero.represent(state)
            out_policy, out_value = self.muZero.predict(state)

            policy_loss = F.cross_entropy(out_policy, policy[:, 0]) 
            value_loss = self.scalar_value_loss(out_value, phi_value[:, 0])
            reward_loss = torch.zeros(value_loss.shape, device=self.device)

            for k in range(1, self.args['K'] + 1):
                state, out_reward = self.muZero.dynamics(state, action[:, k - 1])
                reward_loss += self.scalar_reward_loss(out_reward, phi_reward[:, k])
                state.register_hook(lambda grad: grad * 0.5)

                out_policy, out_value = self.muZero.predict(state)

                policy_loss += F.cross_entropy(out_policy, policy[:, k])
                value_loss += self.scalar_value_loss(out_value, phi_value[:, k])

            loss = (value_loss * self.args['value_loss_weight'] + policy_loss + reward_loss).mean()
            loss.register_hook(lambda grad: grad * 1 / self.args['K'])

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.muZero.parameters(), self.args['max_grad_norm'])
            self.optimizer.step()

    def run(self):
        for iteration in range(self.args['num_iterations']):
            print(f"iteration: {iteration}")
            self.replayBuffer.empty()

            self.muZero.eval()
            for train_game_idx in (self_play_bar := trange(self.args['num_train_games'], desc="train_game")):
                self.replayBuffer.memory += self.self_play(train_game_idx + iteration * self.args['num_train_games'])
                self_play_bar.set_description(f"Avg. steps per Game: {len(self.replayBuffer.memory) / (train_game_idx + 1):.2f}")
            self.replayBuffer.build_trajectories()

            self.muZero.train()
            for epoch in trange(self.args['num_epochs'], desc="epochs"):
                self.train()

            torch.save(self.muZero.state_dict(), f"./Environments/{self.game}/Models/model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"./Environments/{self.game}/Models/optimizer_{iteration}.pt")

    def scalar_reward_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def scalar_value_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)


# %%
args = {
    'num_iterations': 1,
    'num_train_games': 100,
    'num_mcts_runs': 50,
    'num_epochs': 4,
    'batch_size': 64,
    'temperature': 1,
    'K': 5,
    'pb_c_base': 19625,
    'pb_c_init': 2,
    'N': 10,
    'dirichlet_alpha': 0.3,
    'dirichlet_epsilon': 0.25,
    'gamma': 0.997,
    'value_loss_weight': 0.25,
    'max_grad_norm': 5,
    'known_bounds': {} #{'min': 0, 'max': 1},
}

LOAD = True

game = CartPole()
muZero = MuZero(game).to(device)
optimizer = torch.optim.Adam(muZero.parameters(), lr=0.001)

if LOAD:
    muZero.load_state_dict(torch.load(f"./Environments/{game}/Models/model.pt", map_location=device))
    optimizer.load_state_dict(torch.load(f"./Environments/{game}/Models/optimizer.pt", map_location=device))

trainer = Trainer(muZero, optimizer, game, args)
trainer.run()

# args = {
#     'num_iterations': 20,
#     'num_train_games': 100,
#     'num_mcts_runs': 50,
#     'num_epochs': 4,
#     'batch_size': 64,
#     'temperature': 1,
#     'K': 5,
#     'pb_c_base': 19625,
#     'pb_c_init': 2,
#     'N': 10,
#     'dirichlet_alpha': 0.3,
#     'dirichlet_epsilon': 0.05,
#     'gamma': 0.997,
#     'value_loss_weight': 0.25,
#     'max_grad_norm': 5,
#     'known_bounds': {} #{'min': 0, 'max': 1},
# }


# env = gym.make('CartPole-v1', render_mode='human')
# testGame = CartPole()

# muZero = MuZero(testGame).to(device)
# muZero.load_state_dict(torch.load("../../Environments/{testGame}/Models/model.pt", map_location=device))
# muZero.eval()

# mcts = MCTS(muZero, testGame, args)

# TEMPERATURE = 0

# results = []

# for i in range(1):
#       observation, info = env.reset()
#       counter = 0

#       with torch.no_grad():
#             while True:
#                   encoded_observation = testGame.get_encoded_observation(observation)
#                   root = mcts.search(encoded_observation, 0, 2)

#                   action_probs = [0] * testGame.action_size
#                   for child in root.children:
#                         action_probs[child.action_taken] = child.visit_count
#                   action_probs /= np.sum(action_probs)

#                   if TEMPERATURE == 0:
#                         action = np.argmax(action_probs)
#                   elif TEMPERATURE == float('inf'):
#                         action = np.random.choice([r for r in range(testGame.action_size) if action_probs[r] > 0])
#                   else:
#                         temperature_action_probs = action_probs ** (1 / TEMPERATURE)
#                         temperature_action_probs /= np.sum(temperature_action_probs)
#                         action = np.random.choice(len(temperature_action_probs), p=temperature_action_probs)

#                   observation, reward, terminated, truncated, info = env.step(action)

#                   if terminated or truncated:
#                         results.append(counter)
#                         counter = 0
#                         break

#                   counter += 1

# env.close()

# print(sum(results) / len(results))


