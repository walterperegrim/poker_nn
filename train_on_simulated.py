import json
import sys
import csv
from treys import Card, Deck
import numpy as np
import pandas as pd
import csv
import holdem_calc
from holdem_calc import holdem_functions
import time
import random
from poker_NN_prototype import PokerNN
from sklearn.model_selection import train_test_split
import tensorflow as tf

data = []
players = []
pair_strengths = {}
with open('hole_rankings.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    for row in reader:
        pair_strengths[row[0]] = row[1:]
pair_strengths = {k:[round(float(vx) / 100, 5) for vx in v] for (k,v) in pair_strengths.items()}

#For tracking player order
class Queue:
    def __init__(self):
        self.list = []
    def __init__(self,players):
        self.list = []
        for k in players:
            self.push(k)
    def push(self,item):
        self.list.insert(0,item)
    def pop(self):
        return self.list.pop()
    def getSize(self):
        return len(self.list)
    def isEmpty(self):
        return len(self.list) == 0

def estimate_win_rate(num_players,hole_cards,board):        #calculates exact win probabilites
    exact_calculation = True
    num_sims = 1
    verbose = False
    read_from_file = None
    odds = holdem_calc.calculate(board, exact_calculation,num_sims, read_from_file ,hole_cards, verbose)
    return odds[1:]

class PokerAgent:                                           #generic poker agent class
    def __init__(self, name, money, position ):
        self.name = name
        self.money = money
        self.win_prob = 0.0
        self.action = ''
        self.position = position
        self.string_cards = []

    def set_cards(self,cards):
        self.cards = cards
        self.string_cards = [[Card.int_to_str(card)[0], Card.int_to_str(card)[1]] for card in self.cards]

    def calc_hand_potential(self,num_players):                                 #calculates hand potentials for hole stage
        hole = sorted(self.string_cards,key=lambda c:c[0], reverse=True)
        s = str(hole[0][0]) + str(hole[1][0])
        if hole[0][1] == hole[1][1]:
            s += "s"
        if s not in pair_strengths and "s" not in s:
            s = s[::-1]
        elif s not in pair_strengths and "s" in s:
            s = ''.join([ s[x:x+2][::-1] for x in range(0, len(s), 2) ])
        self.win_prob = pair_strengths[s][num_players-1]

    def take_action(self, actions):
        if self.win_prob < 0.1:
            action = actions[0]
        elif self.win_prob > 0.1 and self.win_prob < 0.25 :
            action = actions[1]
        else:
            action = actions[2]
        self.action = action

    def __repr__(self):
        return str(self.name)

#All of these agents are the same as of now
class AggressiveAgent(PokerAgent):
    def __init__(self,name, money, position):
        super().__init__(name,money,position)

class StandardAgent(PokerAgent):
    def __init__(self,name, money, position):
        super().__init__(name,money,position)

class PassiveAgent(PokerAgent):
    def __init__(self,name, money, position):
        super().__init__(name,money,position)

class Poker:
    def __init__(self,players):
        self.deck = Deck()
        self.queue = Queue(players)
        self.board = []
        self.stage = ''
        self.pocket_cards = []
        self.last_action = ''
    def action_helper(self,player,previous_actions):                            #get possible actions
        if not previous_actions or previous_actions[-1] == 'checks':
            self.last_action = 'checks'
            return ['checks','bets','bets']
        elif previous_actions[-1] == 'bets':
            self.last_action = 'bets'
            return ['folds','calls','raises']
        elif previous_actions[-1] == 'raises':
            self.last_action = 'raises'
            return ['folds','calls','re-raises']
        elif previous_actions[-1] == 're-raises':
            self.last_action = 're-raises'
            return ['folds','folds','calls']
        elif previous_actions[-1] in ['folds','calls']:
            i = 1
            while i < len(previous_actions)+1:
                if previous_actions[-i] == 'bets':
                    self.last_action = 'bets'
                    return ['folds','calls','raises']
                elif previous_actions[-i] == 'raises':
                    self.last_action = 'raises'
                    return ['folds','calls','re-raises']
                elif previous_actions[-i] == 're-raises':
                    self.last_action = 're-raises'
                    return ['folds','folds','calls']
                i+= 1

    def iterate_table(self):                                                #iterate around poker table
        t = [k.win_prob for k in self.queue.list]
        temp_queue = Queue([])
        previous_actions = []
        while self.queue.getSize() > 0:
            my_dict = {}
            my_dict['stage'] = self.stage
            my_dict['num_players'] = self.queue.getSize() + temp_queue.getSize()
            player = self.queue.pop()
            actions = self.action_helper(player,previous_actions)
            player.take_action(actions)
            my_dict['hand_odds'] = player.win_prob
            my_dict['std_odds'] = np.std(t) + np.random.uniform(-0.05,0.06)
            my_dict['last_action'] = self.last_action
            my_dict['action'] = player.action
            data.append(my_dict)
            my_dict = {}
            #print(player.name,player.win_prob,player.string_cards,player.action)
            if player.action in ['bets', 'raises', 're-raises']:
                while(temp_queue.getSize() > 0):
                    self.queue.push(temp_queue.pop())
                temp_queue.push(player)
            elif player.action in ['calls','checks']:
                temp_queue.push(player)

            previous_actions.append(player.action)
        if self.queue.getSize() == 0 and temp_queue.getSize() == 1:
            #print("WINNER:",temp_queue.pop().name)
            [agent.__init__(agent.name,agent.money,agent.position) for agent in players]
            self.__init__(players)
            return False
        self.queue.list = sorted(temp_queue.list, key=lambda x: x.position, reverse=True)
        return True

    def run(self,stage):
        self.stage = stage
        if self.stage == 'preflop':
            [(agent.set_cards(self.deck.draw(2)),agent.calc_hand_potential(self.queue.getSize())) for agent in self.queue.list]

        elif self.stage == 'flop':
            self.board = self.deck.draw(3)
            string_comm = [str(Card.int_to_str(card)[0]+ Card.int_to_str(card)[1]) for card in self.board]
            #print(string_comm)
            self.pocket_cards = [str(Card.int_to_str(card)[0]+ Card.int_to_str(card)[1]) for agent in self.queue.list for card in agent.cards]
            odds = estimate_win_rate(self.queue.getSize(),self.pocket_cards,string_comm)
            for k in range(self.queue.getSize()):
                self.queue.list[k].win_prob = odds[k]

        elif self.stage == 'turn':
            self.board.append(self.deck.draw(1))
            string_comm = [str(Card.int_to_str(card)[0]+ Card.int_to_str(card)[1]) for card in self.board]
            #print(string_comm)
            self.pocket_cards = [str(Card.int_to_str(card)[0]+ Card.int_to_str(card)[1]) for agent in self.queue.list for card in agent.cards]
            odds = estimate_win_rate(self.queue.getSize(),self.pocket_cards,string_comm)
            for k in range(self.queue.getSize()):
                self.queue.list[k].win_prob = odds[k]

        elif self.stage == 'river':
            self.board.append(self.deck.draw(1))
            string_comm = [str(Card.int_to_str(card)[0]+ Card.int_to_str(card)[1]) for card in self.board]
            #print(string_comm)
            self.pocket_cards = [str(Card.int_to_str(card)[0]+ Card.int_to_str(card)[1]) for agent in self.queue.list for card in agent.cards]
            odds = estimate_win_rate(self.queue.getSize(),self.pocket_cards,string_comm)
            for k in range(self.queue.getSize()):
                self.queue.list[k].win_prob = odds[k]

        succ = self.iterate_table()
        return succ

a = StandardAgent("a",10,1)
a_1 = StandardAgent("a_1",10,2)
a_2 = AggressiveAgent("a_2",10,3)
s = StandardAgent("s",10,4)
s_1 = StandardAgent("s_1",10,5)
s_2 = StandardAgent("s_2",10,6)
p = PassiveAgent("p",10,7)
p_1 = PassiveAgent("p_1",10,8)
p_2 = PassiveAgent("p_2",10,9)
poker = Poker([a,a_1,a_2,s,s_1,s_2,p,p_1,p_2])
players = [a,a_1,a_2,s,s_1,s_2,p,p_1,p_2]
for k in range(750):
    for stage in ['preflop','flop','turn','river']:
        succ = poker.run(stage)
        if not succ:
            break

df = pd.DataFrame(data, columns = ['stage', 'num_players','hand_odds','std_odds','last_action','action'])
#df.to_csv('more_data.csv',index=False)
#df = pd.read_csv('more_data.csv')
features = df.drop(['action'], axis=1)
features = pd.get_dummies(data=features, columns=['stage', 'last_action'])
labels = pd.get_dummies(df['action'])
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33)
nn = PokerNN((11, 16, 6), features, labels, ('relu', 'softmax'))
t_score = nn.eval(25,64,X_train, X_test, y_train, y_test, False)
