import numpy as np
import sys, os
sys.path.append(os.path.abspath('..'))
from game.tictactoe import TicTacToe
# from mcts import Node, MCTS
import math

class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        
        self.children = []
        self.expandable_moves = game.get_valid_moves(state)
        
        self.visit_count = 0
        self.value_sum = 0
        
    def is_fully_expanded(self):
        return np.sum(self.expandable_moves) == 0 and len(self.children) > 0
    
    def select(self):
        # return best_child
        # for each child cal ucb, update best child, ucb
        # find best child

        best_child = None
        best_ucb = float('-inf')
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child
        return best_child
        
    def get_ucb(self, child):
        q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) / child.visit_count)
    
    def expand(self):
        # rand action from expandable_moves
        # append child to self children and return child
        action = np.random.choice(np.where(self.expandable_moves==1)[0])
        self.expandable_moves[action] = 0

        child_state = self.state.copy()
        child_state = self.game.get_next_state(child_state, action, player=1)
        child_state = self.game.change_perspective(child_state, player=-1)

        child = Node(self.game, self.args, child_state, self, action)
        self.children.append(child)
        return child
    
    def simulate(self):
        # v, istermin
        # return val if terminal

        # rollout_state, rollout_player
        # while
            # valid mov = get_valdmv
            # action = radnom valid ==1
            # roll_state = get next state
            # v, ister
            # if ter
                # if rollout_player = -1
                # val = get opponenet
            # return val
            # rollout_player = getopponent
        value, is_terminal = self.game.get_value_and_terminated(self.state, self.action_taken)
        if is_terminal:
            return value
        
        rollout_state = self.state.copy()
        rollout_player = 1
        while True:
            action = np.random.choice(np.where(self.game.get_valid_moves(rollout_state)==1)[0])
            rollout_state = self.game.get_next_state(rollout_state, action, rollout_player)
            value, is_terminal = self.game.get_value_and_terminated(rollout_state, action)
            if is_terminal:
                if rollout_player == -1:
                    value = self.game.get_opponent_value(value)
                return value
            rollout_player = self.game.get_opponent(rollout_player)
            

    def backpropagate(self, value):
        # update val sum and vis count
        # value opponent
        # parent back pro
        self.value_sum += value
        self.visit_count += 1
        value = self.game.get_opponent_value(value)
        if self.parent:
            self.parent.backpropagate(value) 


class MCTS:
    def __init__(self, game, args):
        self.game = game
        self.args = args

    def search(self, state):
        ### return action_probs
        
        # for num_searches
            # select leaf node til not fully explaneded
            # v, istermin = getvalueand terminaled
            # v get oppenet value
            # if not istermin:
                # node = node.expand
                # v = node.simulate 
            # backprop 
        # action_probs = zero action size
        # for child in chilrens
            # action_probs[child.action_taken] = child vis_count
        # action_prob / np sum actionpob
        #return act prob
        root = Node(self.game, self.args, state)
        for i in range(self.args["num_searches"]):
            node = root
            while node.is_fully_expanded():
                node = node.select()
            value, is_terminal = self.game.get_value_and_terminated(state, node.action_taken)
            value = self.game.get_opponent_value(value)
            if not is_terminal:
                node = node.expand()
                value = node.simulate()
            node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs

tictactoe = TicTacToe()
player = 1
args = {
    'C': 1.41,
    'num_searches': 1000
}
mcts = MCTS(tictactoe, args)
state = tictactoe.get_initial_state()

while True:
    print(state)
    
    if player == 1:
        valid_moves = tictactoe.get_valid_moves(state)
        print("valid_moves", [i for i in range(tictactoe.action_size) if valid_moves[i] == 1])
        action = int(input(f"{player}:"))

        if valid_moves[action] == 0:
            print("action not valid")
            continue
            
    else:
        neutral_state = tictactoe.change_perspective(state, player)
        mcts_probs = mcts.search(neutral_state)
        action = np.argmax(mcts_probs)
        
    state = tictactoe.get_next_state(state, action, player)
    
    value, is_terminal = tictactoe.get_value_and_terminated(state, action)
    
    if is_terminal:
        print(state)
        if value == 1:
            print(player, "won")
        else:
            print("draw")
        break
        
    player = tictactoe.get_opponent(player)