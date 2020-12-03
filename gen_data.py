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

def get_metric(stage,potential,strength):                                       #account for hand strength and potential
    states = ['hole','flop','turn','river']
    int_state = states.index(stage) + 1
    prob = float(int_state * (9 - strength) + (potential / int_state))
    return prob

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
        self.metric = -1.0
    def set_cards(self,cards):
        self.hand_cards = cards
        self.string_cards = trans_cards(self.hand_cards)

    def update_strength_potential(self,stage,board):

        if stage == 'hole':
            self.hand_potential = calc_hand_potential(self.string_cards)
            if self.string_cards[0][0] == self.string_cards[1][0]:
                self.hand_strength = 8
            else:
                self.hand_strength = 9
        else:
            self.hand_strength = evaluator.get_rank_class(evaluator.evaluate(poker.board, self.hand_cards))
            self.hand_potential = 1 - evaluator.get_five_card_rank_percentage(evaluator.evaluate(board, self.hand_cards))

    def take_action(self,stage,actions):
        action = ''
        prob = get_metric(stage,float(self.hand_potential),int(self.hand_strength))
        self.metric = prob
        if stage == 'hole':
            if prob > -0.14 and prob < 0.0:     #check
                action = actions[1]
            elif prob >= 0.0:     #bet
                action = actions[2]
            else:                               #fold
                action = actions[0]
        elif stage == 'flop':
            if (prob > 0.03 and prob < 2.0):
                action = actions[1]
            elif prob >= 2.0:
                action = actions[2]
            else:
                action = actions[0]
        elif stage == 'turn':
            if (prob > 0.05 and prob < 3.0):
                action = actions[1]
            elif prob >= 3.0:
                action = actions[2]
            else:
                action = actions[0]
        else:
            if (prob > 0.05 and prob < 4.0):
                action = actions[1]
            elif prob >= 4.0:
                action = actions[2]
            else:
                action = actions[0]

        self.action = action
        return action

    def __str__(self):
        return self.name + ": " + str(self.string_cards[0]) + ", " + str(self.string_cards[1]) + ", " + str(self.hand_strength) + ", " + str(self.hand_potential)

class AveragePlayer:
    def __init__(self,name,money):
        self.name = name
        self.money = money
        self.hand_cards = []
        self.string_cards = []
        self.hand_strength = 9
        self.hand_potential = 0
        self.action = ''
        self.metric = -1.0
    def set_cards(self,cards):
        self.hand_cards = cards
        self.string_cards = trans_cards(self.hand_cards)

    def update_strength_potential(self,stage,board):
        if stage == 'hole':
            self.hand_potential = calc_hand_potential(self.string_cards)
            if self.string_cards[0][0] == self.string_cards[1][0]:
                self.hand_strength = 8
            else:
                self.hand_strength = 9
        else:
            self.hand_strength = evaluator.get_rank_class(evaluator.evaluate(poker.board, self.hand_cards))
            self.hand_potential = 1 - evaluator.get_five_card_rank_percentage(evaluator.evaluate(board, self.hand_cards))

    def take_action(self,stage,actions):
        action = ''
        prob = get_metric(stage,float(self.hand_potential),int(self.hand_strength))
        self.metric = prob
        if stage == 'hole':
            if prob > -0.07 and prob < 0.04:     #check
                action = actions[1]
            elif prob >= 0.04:     #bet
                action = actions[2]
            else:                               #fold
                action = actions[0]
        elif stage == 'flop':
            if (prob > 0.05 and prob < 2.0):
                action = actions[1]
            elif prob >= 2.0:
                action = actions[2]
            else:
                action = actions[0]
        elif stage == 'turn':
            if (prob > 3.0 and prob < 6.0):
                action = actions[1]
            elif prob >= 6.0:
                action = actions[2]
            else:
                action = actions[0]
        else:
            if (prob > 4.0 and prob < 8.0):
                action = actions[1]
            elif prob >= 8.0:
                action = actions[2]
            else:
                action = actions[0]
        self.action = action
        return action

    def __str__(self):
        return self.name + ": " + str(self.string_cards[0]) + ", " + str(self.string_cards[1]) + ", " + str(self.hand_strength) + ", " + str(self.hand_potential)

class SafePlayer:
    def __init__(self,name,money):
        self.name = name
        self.money = money
        self.hand_cards = []
        self.string_cards = []
        self.hand_strength = 9
        self.hand_potential = 0
        self.action = ''
        self.metric = -1.0
    def set_cards(self,cards):
        self.hand_cards = cards
        self.string_cards = trans_cards(self.hand_cards)

    def update_strength_potential(self,stage,board):
        if stage == 'hole':
            self.hand_potential = calc_hand_potential(self.string_cards)
            if self.string_cards[0][0] == self.string_cards[1][0]:
                self.hand_strength = 8
            else:
                self.hand_strength = 9
        else:
            self.hand_strength = evaluator.get_rank_class(evaluator.evaluate(poker.board, self.hand_cards))
            self.hand_potential = 1 - evaluator.get_five_card_rank_percentage(evaluator.evaluate(board, self.hand_cards))


    def take_action(self,stage,actions):
        action = ''
        prob = get_metric(stage,float(self.hand_potential),int(self.hand_strength))
        self.metric = prob
        if stage == 'hole':
            if prob >= 0.0 and prob < 0.4:     #check
                action = actions[1]
            elif prob >= 0.4:     #bet
                action = actions[2]
            else:                               #fold
                action = actions[0]
        elif stage == 'flop':
            if (prob > 2.0 and prob < 4.0):
                action = actions[1]
            elif prob >= 4.0:
                action = actions[2]
            else:
                action = actions[0]
        elif stage == 'turn':
            if (prob > 3.0 and prob < 9.0):
                action = actions[1]
            elif prob >= 9.0:
                action = actions[2]
            else:
                action = actions[0]
        else:
            if (prob > 4.0 and prob < 12.0):
                action = actions[1]
            elif prob >= 12.0:
                action = actions[2]
            else:
                action = actions[0]

        self.action = action
        return action

    def __str__(self):
        return self.name + ": " + str(self.string_cards[0]) + ", " + str(self.string_cards[1]) + ", " + str(self.hand_strength) + ", " + str(self.hand_potential)

class MyPlayer:
    def __init__(self,name,money):
        self.name = name
        self.money = money
        self.hand_cards = []
        self.string_cards = []
        self.hand_strength = 9
        self.hand_potential = 0
        self.action = ''
        self.metric = -1.0
    def set_cards(self,cards):
        self.hand_cards = cards
        self.string_cards = trans_cards(self.hand_cards)

    def update_strength_potential(self,stage,board):
        if stage == 'hole':
            self.hand_potential = calc_hand_potential(self.string_cards)
            if self.string_cards[0][0] == self.string_cards[1][0]:
                self.hand_strength = 8
            else:
                self.hand_strength = 9
        else:
            self.hand_strength = evaluator.get_rank_class(evaluator.evaluate(poker.board, self.hand_cards))
            self.hand_potential = 1 - evaluator.get_five_card_rank_percentage(evaluator.evaluate(board, self.hand_cards))

    def take_action(self,stage,actions):
        action = ''
        prob = get_metric(stage,float(self.hand_potential),int(self.hand_strength))
        self.metric = prob
        if stage == 'hole':
            if prob > -0.07 and prob < 0.04:     #check
                action = actions[1]#'check'
            elif prob >= 0.04:     #bet
                action = actions[2]#'bet'
            else:                               #fold
                action = actions[0]#'fold'
        elif stage == 'flop':
            if (prob > 0.05 and prob < 2.0):
                action = actions[1]
            elif prob >= 2.0:
                action = actions[2]
            else:
                action = actions[0]
        elif stage == 'turn':
            if (prob > 3.0 and prob < 6.0):
                action = actions[1]
            elif prob >= 6.0:
                action = actions[2]
            else:
                action = actions[0]
        else:
            if (prob > 4.0 and prob < 8.0):
                action = actions[1]
            elif prob >= 8.0:
                action = actions[2]
            else:
                action = actions[0]

        self.action = action
        return action

    def __str__(self):
        return self.name + ": " + str(self.string_cards[0]) + ", " + str(self.string_cards[1]) + ", " + str(self.hand_strength) + ", " + str(self.hand_potential)

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
                player.update_strength_potential(self.state,self.board)
                if player.name == self.big_blind:
                    player.metric = get_metric(self.state,float(player.hand_potential),int(player.hand_strength))
                    player.action = 'call'
            self.act()                                                          #get player action
            self.state = 'flop'

        elif self.state == 'flop':
            self.board = self.deck.draw(3)
            print(trans_cards(self.board))
            for player in self.players:
                player.update_strength_potential(self.state,self.board)
            self.act()
            self.state = 'turn'

        elif self.state == 'turn':
            self.board.append(self.deck.draw(1))
            print(trans_cards(self.board))
            for player in self.players:
                player.update_strength_potential(self.state,self.board)
            self.act()
            self.state = 'river'

        elif self.state == 'river':
            self.board.append(self.deck.draw(1))
            print(trans_cards(self.board))
            for player in self.players:
                player.update_strength_potential(self.state,self.board)
            self.act()
            self.state = 'showdown'

        if self.state == 'showdown':
            best_hand = 0.0
            winner = self.players[0]
            for k in self.players:
                if k.hand_potential > best_hand:
                    best_hand = k.hand_potential
                    winner = k
            self.players = [winner]
            self.get_result()

    def get_result(self):
        print()
        print("HAND OVER")
        print("WINNER:", self.players[0])
        sys.exit()

    def handle_bet(self,current_players,ordered_players,actions):
        prev_action = 'bet'
        curr_action = ''
        t = 0
        for player in ordered_players:
            curr_action = player.take_action(self.state,actions)
            print(player,curr_action)
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

            print(player,curr_action)
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
            self.get_result()

poker = Poker()
w = MyPlayer("myPlayer",100)
e = AgressivePlayer("agressivePlayer",50)
a = AveragePlayer("averagePlayer",25)
p = SafePlayer("safePlayer",50)
poker.add_players([w,e,a,p])
poker.deal()
for state in ['flop','flop+turn','flop+turn+river']:    #iterate through stages of hand
    poker.deal()
