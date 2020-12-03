import json
import sys
import csv
from treys import Card, Deck, Evaluator
import numpy as np
import pandas as pd
import csv
import time

pair_strengths = {}
with open('hole_rankings.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    for row in reader:
        pair_strengths[row[0]] = row[1:]

pair_strengths = {k:[round(float(vx) / 100, 5) for vx in v] for (k,v) in pair_strengths.items()}
evaluator = Evaluator()
poker_data = []

def calc_hand_potential(my_cards, num_players):                                 #calculates hand potentials for hole stage
    hole = sorted(my_cards,key=lambda c:c[0], reverse=True)
    s = str(hole[0][0]) + str(hole[1][0])
    if hole[0][1] == hole[1][1]:
        s += "s"
    if s not in pair_strengths and "s" not in s:
        s = s[::-1]
    elif s not in pair_strengths and "s" in s:
        s = ''.join([ s[x:x+2][::-1] for x in range(0, len(s), 2) ])
    return pair_strengths[s][num_players]
def trans_cards(cards):                                                         #just for printing out cards
    trans = []
    for card in cards:
        trans.append([Card.int_to_str(card)[0], Card.int_to_str(card)[1]])
    return trans
def re_init_players(players):
    for player in players:
        player.hand_cards = []
        player.string_cards = []
        player.hand_strength = 9
        player.hand_potential = 0
        player.action = ''
        player.rest_cards = Deck().cards
        player.win_prob = 0.0
        player.action_history = []
    return players
# Estimate the ratio of winning games given the current state of the game
def estimate_win_rate(num_players,hole_cards,community_cards,deck):
    items = deck
    opponents_hole = [[items[i],items[j]] for i in range(len(items)) for j in range(i+1, len(items))]
    opponent_scores = []
    for h in opponents_hole:
        hand_potential = 1 - evaluator.get_five_card_rank_percentage(evaluator.evaluate(community_cards, h))
        opponent_scores.append(hand_potential)
    hand_potential = 1 - evaluator.get_five_card_rank_percentage(evaluator.evaluate(community_cards, hole_cards))
    my_score = hand_potential
    win_prob = [1  for s in opponent_scores if my_score >= s]
    return len(win_prob) / len(opponent_scores)
#NOTE: all player classes are very similar, only difference are the quality of hands played
class AgressivePlayer:
    def __init__(self,name,money):
        self.name = name
        self.money = money
        self.hand_cards = []
        self.string_cards = []
        self.hand_strength = 9
        self.hand_potential = 0
        self.action = ''
        self.rest_cards = Deck().cards
        self.win_prob = 0.0
        self.action_history = []

    def set_cards(self,cards):
        self.hand_cards = cards
        self.string_cards = trans_cards(self.hand_cards)
        self.agg = 0.0
    def take_action(self,stage,actions, current_amount):
        action = ''
        n = 0
        if stage == 'hole':
            n = 0.5
            bet = self.win_prob * 10
        else:
            bet = self.win_prob * 2.5

        if self.win_prob < 0.6 - n:
            action = actions[0]
            self.action = action
            return action
        if bet < current_amount - 2:
            action = actions[0]
        elif bet >= current_amount - 2 and bet < current_amount + 1:
            action = actions[1]
        else:
            action = actions[2]
        self.action = action
        return action

    def __str__(self):
        return self.name + ": " + str(self.string_cards[0]) + ", " + str(self.string_cards[1]) + ", " + str(self.win_prob)

class AveragePlayer:
    def __init__(self,name,money):
        self.name = name
        self.money = money
        self.hand_cards = []
        self.string_cards = []
        self.hand_strength = 9
        self.hand_potential = 0
        self.action = ''
        self.rest_cards = Deck().cards
        self.win_prob = 0.0

    def set_cards(self,cards):
        self.hand_cards = cards
        self.string_cards = trans_cards(self.hand_cards)
        self.agg = 0.0
    def take_action(self,stage,actions, current_amount):
        action = ''
        n = 0
        if stage == 'hole':
            n = 0.5
            bet = self.win_prob * 7
        else:
            bet = self.win_prob * 2.5

        if self.win_prob < 0.6 - n:
            action = actions[0]
            self.action = action
            return action
        if bet < current_amount - 1:
            action = actions[0]
        elif bet >= current_amount - 1 and bet < current_amount + 1:
            action = actions[1]
        else:
            action = actions[2]
        self.action = action
        return action

    def __str__(self):
        return self.name + ": " + str(self.string_cards[0]) + ", " + str(self.string_cards[1]) + ", " + str(self.win_prob)

class SafePlayer:
    def __init__(self,name,money):
        self.name = name
        self.money = money
        self.hand_cards = []
        self.string_cards = []
        self.hand_strength = 9
        self.hand_potential = 0
        self.action = ''
        self.win_prob = 0.0
        self.rest_cards = Deck().cards

    def set_cards(self,cards):
        self.hand_cards = cards
        self.string_cards = trans_cards(self.hand_cards)
        self.agg = 0.0
    def take_action(self,stage,actions, current_amount):
        action = ''
        n = 0
        if stage == 'hole':
            n = 0.5
            bet = self.win_prob * 7
        else:
            bet = self.win_prob * 2.5
        if self.win_prob < 0.6 - n:
            action = actions[0]
            self.action = action
            return action
        if bet < current_amount:
            action = actions[0]
        elif bet >= current_amount and bet < current_amount + 2:
            action = actions[1]
        else:
            action = actions[2]
        self.action = action
        return action

    def __str__(self):
        return self.name + ": " + str(self.string_cards[0]) + ", " + str(self.string_cards[1]) + ", " + str(self.win_prob)

class MyPlayer:
    def __init__(self,name,money):
        self.name = name
        self.money = money
        self.hand_cards = []
        self.string_cards = []
        self.hand_strength = 9
        self.hand_potential = 0
        self.action = ''
        self.win_prob = 0.0
        self.rest_cards = Deck().cards
        self.agg = 0.0
    def set_cards(self,cards):
        self.hand_cards = cards
        self.string_cards = trans_cards(self.hand_cards)

    def take_action(self,stage,actions, current_amount):
        action = ''
        n = 0
        if stage == 'hole':
            n = 0.5
            bet = self.win_prob * 7
        else:
            bet = self.win_prob * 2.5
        if self.win_prob < 0.6 - n:
            action = actions[0]
            self.action = action
            return action
        if bet < current_amount - 1:
            action = actions[0]
        elif bet >= current_amount - 1 and bet < current_amount + 1:
            action = actions[1]
        else:
            action = actions[2]
        self.action = action
        return action

    def __str__(self):
        return self.name + ": " + str(self.string_cards[0]) + ", " + str(self.string_cards[1]) + ", " + str(self.win_prob)
#Class for tracking game data
class Poker:
    def __init__(self):
        self.deck = Deck()
        self.num_players = 0
        self.players = []
        self.board = []
        self.state = 'hole'
        self.big_blind = 'myPlayer'
        self.current_amount = 1.0
    def add_players(self,players):
        for player in players:
            self.num_players += 1
            self.players.append(player)
    def re_init(self):
        self.deck = Deck()
        self.board = []
        self.num_players = 0
        self.state = 'hole'
        self.players = []
        self.big_blind = 'myPlayer'
        self.current_amount = 1.0
    def deal(self):                                                             #Deal cards
        #print()
        #print(self.state.upper())
        if self.state == 'hole':
            for player in self.players:
                player.set_cards(self.deck.draw(2))
                player.hand_potential = calc_hand_potential(player.string_cards, self.num_players)
                player.win_prob = player.hand_potential
                #print(player)
                player.rest_cards.remove(player.hand_cards[0])
                player.rest_cards.remove(player.hand_cards[1])
                if player.name == self.big_blind:
                    player.action = 'calls'
            #print()
            self.act()
            if len(self.players) > 1:                                                          #get player action
                self.state = 'flop'

        elif self.state == 'flop':
            self.current_amount = 0.0
            self.board = self.deck.draw(3)
            #print(trans_cards(self.board))
            for player in self.players:
                for k in self.board:
                    player.rest_cards.remove(k)
                player.win_prob = estimate_win_rate(self.num_players,player.hand_cards,self.board,player.rest_cards)

            self.act()
            if len(self.players) > 1:
                self.state = 'turn'

        elif self.state == 'turn':
            self.current_amount = 0.0
            self.board.append(self.deck.draw(1))
            #print(trans_cards(self.board))
            for player in self.players:
                player.rest_cards.remove(self.board[3])
                player.win_prob = estimate_win_rate(self.num_players,player.hand_cards,self.board,player.rest_cards)
            self.act()
            if len(self.players) > 1:
                self.state = 'river'

        elif self.state == 'river':
            self.current_amount = 0.0
            self.board.append(self.deck.draw(1))
            #print(trans_cards(self.board))
            for player in self.players:
                player.rest_cards.remove(self.board[4])
                player.win_prob = estimate_win_rate(self.num_players,player.hand_cards,self.board,player.rest_cards)

            self.act()
            self.state = 'showdown'

        if self.state == 'showdown':
            best_hand = 0.0
            winner = self.players[0]
            for k in self.players:
                if (k.win_prob) > best_hand:
                    best_hand = k.win_prob
                    winner = k
            self.players = [winner]
            self.get_result()

    def get_result(self):
        return
        #print()
        #print("WINNER:", self.players[0].name)
        #print("--------------------------------------------------------------------------------------------------------")
    def handle_bet(self,current_players,ordered_players,actions,amount):
        row = {}
        prev_action = 'bets'
        curr_action = ''
        t = 0
        for player in ordered_players:
            curr_action = player.take_action(self.state,actions,self.current_amount)
            if curr_action == 'calls':
                print(player.name,curr_action,'$'+str(self.current_amount))
            elif curr_action == 'folds':
                print(player.name,curr_action)



            if player.name == 'agressivePlayer' or player.name == 'averagePlayer' or player.name == 'safePlayer':
                row['stage'] = self.state
                row['hand_quality'] = round(player.win_prob,5)
                row['play_style'] = player.name[:-6]
                if curr_action == 'calls':
                    player.agg += 1
                    row['aggression'] = player.agg + self.current_amount / 10
                    row['stake'] = self.current_amount
                    row['action'] = curr_action
                    poker_data.append(row)
                    row = {}
                elif curr_action == 'folds':
                    player.agg -= 1
                    row['aggression'] = player.agg + self.current_amount / 10
                    row['stake'] = self.current_amount
                    row['action'] = curr_action
                    poker_data.append(row)
                    row = {}

            if curr_action == 'folds':   #remove folded
                current_players.remove(player)
            elif curr_action == actions[2]:     #if bet,raise,reraise
                idx = current_players.index(player)
                new_iter = current_players[idx+1:] + current_players[:idx]
                if self.state == 'hole':
                    self.current_amount = round(player.win_prob * 7,5)
                else:
                    self.current_amount = round(player.win_prob * 2.5,5)

                if player.name == 'agressivePlayer' or player.name == 'averagePlayer' or player.name == 'safePlayer':
                    player.agg += 1
                    row['aggression'] = player.agg - self.current_amount / 10
                    row['stake'] = self.current_amount
                    row['action'] = curr_action
                    row['play_style'] = player.name[:-6]
                    poker_data.append(row)
                    row = {}
                if actions[2] == 'raises':
                    #print(player.name,curr_action,'$'+str(self.current_amount))
                    self.handle_bet(current_players,new_iter,['folds','calls','re-raises'],self.current_amount )
                    break
                elif actions[2] == 're-raises':
                    #print(player.name,curr_action,'$'+str(self.current_amount))
                    self.handle_bet(current_players,new_iter,['folds','calls','calls'],self.current_amount)
                    break
        self.players = current_players

    def act(self):
        prev_action = ''
        curr_action = ''
        t = 0
        row = {}
        current_players = self.players.copy()
        for player in self.players:
            if player.name == self.big_blind and self.state == 'hole':
                player.action = 'calls'
                curr_action = 'calls'

            elif ((player.name == self.big_blind and self.state != 'hole') or (prev_action == 'checks') or (self.state != 'hole' and t==0)):
                curr_action = player.take_action(self.state,['checks', 'checks','bets'], 0.0)

            elif prev_action == 'calls' or prev_action == 'folds':
                curr_action = player.take_action(self.state,['folds', 'calls','bets'], self.current_amount)

            if curr_action == 'calls':
                print(player.name,curr_action, '$'+str(self.current_amount))
            elif curr_action == 'folds' or curr_action == 'checks':
                print(player.name,curr_action)

            if player.name == 'agressivePlayer' or player.name == 'averagePlayer' or player.name == 'safePlayer':
                row['stage'] = self.state
                row['hand_quality'] = round(player.win_prob,5)
                row['play_style'] = player.name[:-6]
                if curr_action == 'calls':
                    player.agg += 1
                    row['aggression'] = player.agg + self.current_amount / 10
                    row['stake'] = self.current_amount
                    row['action'] = curr_action
                    poker_data.append(row)
                    row = {}
                elif curr_action == 'checks':
                    row['aggression'] = player.agg
                    row['stake'] = 0.0
                    row['action'] = curr_action
                    poker_data.append(row)
                    row = {}
                elif curr_action == 'folds':
                    player.agg -= 1
                    row['aggression'] = player.agg - self.current_amount / 10
                    row['stake'] = self.current_amount
                    row['action'] = curr_action
                    poker_data.append(row)
                    row = {}

            if curr_action == 'folds':
                current_players.remove(player)
            elif curr_action == 'bets':
                if self.state == 'hole':
                    self.current_amount = round(player.win_prob * 7,5)
                else:
                    self.current_amount = round(player.win_prob * 2.5,5)
                if player.name == 'agressivePlayer' or player.name == 'averagePlayer' or player.name == 'safePlayer':
                    player.agg += 1
                    row['aggression'] = player.agg + self.current_amount / 10
                    row['stake'] = self.current_amount
                    row['action'] = curr_action
                    row['play_style'] = player.name[:-6]
                    poker_data.append(row)
                    row = {}
                #print(player.name,curr_action, '$'+str(self.current_amount))
                new_iter = current_players[t+1:] + current_players[:t]
                self.handle_bet(current_players,new_iter,['folds','calls','raises'], self.current_amount)
                break

            t += 1
            prev_action = curr_action
        self.players = current_players
        if len(self.players) == 1:
            self.state = 'showdown'

poker = Poker()
w = MyPlayer("myPlayer",100)
e = AgressivePlayer("agressivePlayer",50)
a = AveragePlayer("averagePlayer",25)
p = SafePlayer("safePlayer",50)
poker.add_players([w,e,a,p])
s_time = time.time()
for i in range(500):
    for state in ['hole','flop','flop+turn','flop+turn+river']:    #iterate through stages of hand
        poker.deal()
        if poker.state == 'showdown':
            poker.re_init()
            w,e,a,p = re_init_players([w,e,a,p])
            poker.add_players([w,e,a,p])
            break

print(time.time() - s_time)
df = pd.DataFrame(poker_data, columns = ['stage', 'aggression','action','stake','play_style','hand_quality'])
print(df.shape)
#df.to_csv('simulated_game_data.csv',index=False)

#Half the implementation of the larger network
"""poker = Poker()
w = AveragePlayer("myPlayer",100)
e = AgressivePlayer("agressivePlayer",50)
a = AveragePlayer("averagePlayer",25)
p = SafePlayer("safePlayer",50)
poker.add_players([w,e,a,p])
s_time = time.time()
poker_data = []
for q in range(5):
    for state in ['hole','flop','flop+turn','flop+turn+river']:    #iterate through stages of hand
        stage = 'stage_' + poker.state
        poker.deal()
        players = poker.players
        row = {}
        row['stage'] = stage
        row['num_players'] = len(players)
        row['agressive_estimate'] = 0
        row['average_estimate'] = 0
        row['safe_estimate'] = 0
        h = [player.name for player in players]
        for player in players:
            if player.name == 'agressivePlayer' or player.name == 'averagePlayer' or player.name == 'safePlayer':
                action = 'action_' + player.action
                stake = player.stake
                play_style = 'play_style_' + player.name[:-6]
                aggression = player.agg
                opponent = np.zeros((1, 13))
                opponent[0][0] = aggression
                opponent[0][1] = stake
                for cat_feature in [action,stage, play_style]:
                    idx = np.where(np.array(X_train.columns.tolist()) == cat_feature)[0][0]
                    opponent[0][idx] = 1
                yhat = model.predict(opponent)[0][0]
                row[player.name[:-6] + '_estimate'] = yhat
                #print(yhat, player.name,player.win_prob)
            else:
                if not player.win_prob:
                    player.win_prob = 0
                row['hand_strength'] = player.win_prob
                if not player.bet:
                    player.bet = 0
                row['action'] = player.bet
        if poker.state == 'showdown' or len(players) == 1:
            poker.re_init()
            w,e,a,p = re_init_players([w,e,a,p])
            poker.add_players([w,e,a,p])
            break
        poker_data.append(row)
        #print(row)
n_df = pd.DataFrame(poker_data, columns = ['stage', 'num_players','hand_strength','agressive_estimate','average_estimate','safe_estimate','action'])
print(n_df.head(15))
if poker.state == 'showdown':
    sys.exit()
    poker.re_init()
    w,e,a,p = re_init_players([w,e,a,p])
    poker.add_players([w,e,a,p])
    break"""
