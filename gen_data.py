import json
import sys
import csv
from treys import Card, Deck, Evaluator
import numpy as np

pair_strengths = json.load(open('new_hole_card_rankings.json'))
evaluator = Evaluator()

def calc_hand_potential(my_cards):                                              #calculates hand potentials for hole stage
    hole = sorted(my_cards,key=lambda c:c[0], reverse=True)
    s = str(hole[0][0]) + str(hole[1][0])
    if hole[0][1] == hole[1][1]:
        s += "s"
    if s not in pair_strengths and "s" not in s:
        s = s[::-1]
    elif s not in pair_strengths and "s" in s:
        s = ''.join([ s[x:x+2][::-1] for x in range(0, len(s), 2) ])
    return pair_strengths[s]

def trans_cards(cards):                                                         #just for printing out cards
    trans = []
    for card in cards:
        trans.append([Card.int_to_str(card)[0], Card.int_to_str(card)[1]])
    return trans

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

    def set_cards(self,cards):
        self.hand_cards = cards
        self.string_cards = trans_cards(self.hand_cards)

    def take_action(self,stage,actions):
        action = ''
        if self.win_prob < 0.1:
            action = actions[0]
        elif self.win_prob >= 0.1 and self.win_prob < 0.4:
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

    def take_action(self,stage,actions):
        action = ''
        if self.win_prob < 0.2:
            action = actions[0]
        elif self.win_prob >= 0.2 and self.win_prob < 0.5:
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

    def take_action(self,stage,actions):
        action = ''
        if self.win_prob < 0.3:
            action = actions[0]
        elif self.win_prob >= 0.3 and self.win_prob < 0.75:
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

    def set_cards(self,cards):
        self.hand_cards = cards
        self.string_cards = trans_cards(self.hand_cards)

    def take_action(self,stage,actions):
        action = ''
        if self.win_prob < 0.2:
            action = actions[0]
        elif self.win_prob >= 0.2 and self.win_prob < 0.5:
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

    def add_players(self,players):
        for player in players:
            self.num_players += 1
            self.players.append(player)

    def deal(self):                                                             #Deal cards
        print()
        print(self.state.upper())
        if self.state == 'hole':
            for player in self.players:
                player.set_cards(self.deck.draw(2))
                player.hand_potential = calc_hand_potential(player.string_cards)
                player.win_prob = 10 * ((1.0-0.0)/(2.32--0.15)*(float(player.hand_potential)-2.32)+1.0)
                print(player)
                player.rest_cards.remove(player.hand_cards[0])
                player.rest_cards.remove(player.hand_cards[1])
                if player.name == self.big_blind:
                    #player.metric = get_metric(self.state,float(player.hand_potential),int(player.hand_strength))
                    player.action = 'call'
            self.act()
            if len(self.players) > 1:                                                          #get player action
                self.state = 'flop'

        elif self.state == 'flop':
            self.board = self.deck.draw(3)
            print(trans_cards(self.board))
            for player in self.players:
                for k in self.board:
                    player.rest_cards.remove(k)
                player.win_prob = estimate_win_rate(self.num_players,player.hand_cards,self.board,player.rest_cards)

            self.act()
            if len(self.players) > 1:
                self.state = 'turn'

        elif self.state == 'turn':
            self.board.append(self.deck.draw(1))
            print(trans_cards(self.board))
            for player in self.players:
                player.rest_cards.remove(self.board[3])
                player.win_prob = estimate_win_rate(self.num_players,player.hand_cards,self.board,player.rest_cards)
            self.act()
            if len(self.players) > 1:
                self.state = 'river'

        elif self.state == 'river':
            self.board.append(self.deck.draw(1))
            print(trans_cards(self.board))
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
        print()
        print("WINNER:", self.players[0])

    def handle_bet(self,current_players,ordered_players,actions):
        prev_action = 'bet'
        curr_action = ''
        t = 0
        for player in ordered_players:
            curr_action = player.take_action(self.state,actions)
            print(player.name,player.win_prob,curr_action)
            if curr_action == 'fold':   #remove folded
                current_players.remove(player)

            elif curr_action == actions[2]:     #if bet,raise,reraise
                idx = current_players.index(player)
                new_iter = current_players[idx+1:] + current_players[:idx]
                if actions[2] == 'raise':
                    self.handle_bet(current_players,new_iter,['fold','call','re-raise'])
                    break
                elif actions[2] == 're-raise':
                    self.handle_bet(current_players,new_iter,['fold','call','call'])
                    break
        self.players = current_players

    def act(self):
        prev_action = ''
        curr_action = ''
        t = 0
        current_players = self.players.copy()
        for player in self.players:
            if player.name == self.big_blind and self.state == 'hole':
                player.action = 'call'
                curr_action = 'call'

            elif ((player.name == self.big_blind and self.state != 'hole') or (prev_action == 'check') or (self.state != 'hole' and t==0)):
                curr_action = player.take_action(self.state,['check', 'check','bet'])

            elif prev_action == 'call' or prev_action == 'fold':
                curr_action = player.take_action(self.state,['fold', 'call','bet'])

            print(player.name,player.win_prob,curr_action)
            if curr_action == 'fold':
                current_players.remove(player)

            elif curr_action == 'bet':
                new_iter = current_players[t+1:] + current_players[:t]
                self.handle_bet(current_players,new_iter,['fold','call','raise'])
                break

            t += 1
            prev_action = curr_action
        self.players = current_players
        if len(self.players) == 1:
            self.state = 'showdown'

for i in range(10):
    poker = Poker()
    w = MyPlayer("myPlayer",100)
    e = AgressivePlayer("agressivePlayer",50)
    a = AveragePlayer("averagePlayer",25)
    p = SafePlayer("safePlayer",50)
    poker.add_players([w,e,a,p])
    for state in ['hole','flop','flop+turn','flop+turn+river']:    #iterate through stages of hand
        poker.deal()
        if poker.state == 'showdown':
            break
