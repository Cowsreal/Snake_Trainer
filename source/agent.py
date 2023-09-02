import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import math

MAX_MEMORY = 100_000
BATCH_SIZE = 1_000
LR = 0.001
BLOCK_SIZE = 20

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 #Randomness
        self.gamma = 0.9 #Discount factor
        self.memory = deque(maxlen=MAX_MEMORY) #popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma)
        # TODO: Model, Trainer

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT 
        dir_r = game.direction == Direction.RIGHT 
        dir_u = game.direction == Direction.UP 
        dir_d = game.direction == Direction.DOWN 

        nearX = game.food[0].x
        nearY = game.food[0].y
        currDistance = math.sqrt((nearX - head.x)**2 + (nearY - head.y)**2)

        for food in game.food:
            if math.sqrt((food.x - head.x)**2 + (food.y - head.y)**2) < currDistance:
                currDistance = math.sqrt((food.x - head.x)**2 + (food.y - head.y)**2)
                nearX = food.x
                nearY = food.y

        state = [
            #Danger straight
            (dir_r and game.is_collision(point_r) or dir_l and game.is_collision(point_l) or dir_u and game.is_collision(point_u) or dir_d and game.is_collision(point_d)),

            #Danger right
            (dir_u and game.is_collision(point_r) or dir_r and game.is_collision(point_d) or dir_d and game.is_collision(point_l) or dir_l and game.is_collision(point_u)),

            #Danger left
            (dir_u and game.is_collision(point_l) or dir_r and game.is_collision(point_u) or dir_d and game.is_collision(point_r) or dir_l and game.is_collision(point_d)),

            #Move Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            #Food location
            nearX < game.head.x,
            nearX > game.head.x,
            nearY < game.head.y,
            nearY > game.head.y
        ]
        return np.array(state, dtype = int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) #list of tuples
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        #random moves: trade off between exploration and exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else: 
            state0 = torch.tensor(state, dtype = torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        #Get old state
        state_old = agent.get_state(game)

        #Get action
        final_move = agent.get_action(state_old)

        #perform action and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        #Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        #Remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            #Train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f'Game: {agent.n_games}, Score: {score}, Record: {record}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()