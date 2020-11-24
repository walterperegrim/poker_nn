import numpy as np
import pandas as pd
import sys
import json
import os
import cnn_poker
import re
from treys import Card, Deck, Evaluator

pair_strengths = json.load(open('new_hole_card_rankings.json'))
def calc_hand_potential(my_cards):
    hole = sorted(my_cards,key=lambda c:c[0], reverse=True)
    s = str(hole[0][0]) + str(hole[1][0])
    if hole[0][1] == hole[1][1]:
        s += "s"
    if s not in pair_strengths and "s" not in s:
        s = s[::-1]
    elif s not in pair_strengths and "s" in s:
        s = ''.join([ s[x:x+2][::-1] for x in range(0, len(s), 2) ])
    return pair_strengths[s]

ldf = []
def parse_game_log(game_log):   #extracts necessary information from each hand
    game_id = game_log[1].split(" ")[2]
    game_start = game_log[0].split(" ")[3] + " " + game_log[0].split(" ")[4]
    game_end = game_log[len(game_log)-1].split(" ")[3] + " " + game_log[len(game_log)-1].split(" ")[4]
    actions_start = 0
    num_players = -3
    money = []
    hole_cards = []
    summary = []
    for k in range(2,len(game_log)):
        if "Player IlxxxlI sitting out" in game_log[k]: #only keep hands that were played
            return
        if("Seat" in game_log[k] or "blind" in game_log[k]):    #get opponent & blinds information
            num_players += 1
            money.append(game_log[k])
        if "received card" in game_log[k]:      #
            i = game_log[k].split("[")[1][:-1]
            if len(i) > 2:
                i = i.replace("10","2")
        if "IlxxxlI received card" in game_log[k]:      #hole cards for our player
            i = game_log[k].split("[")[1][:-1]
            if len(i) > 2:
                i = i.replace("10","T")
            hole_cards.append(i)
        if "------ Summary ------" in game_log[k]:      #get actions & summary of hand
            actions = game_log[actions_start:k-1]
            summary = game_log[k+1:len(game_log)-1]
        if "card" not in game_log[k] and "card" in game_log[k-1]:
            actions_start = k
    if hole_cards:
        ldf.append([game_id,game_start,num_players,hole_cards,money,actions,summary,game_end])  #put data in list

#iterate data files and combine into one dataset
dataset = []
for filename in os.listdir('data/'):
    if filename.endswith('.txt'):
        with open(os.path.join('data/', filename)) as f:
            dataset += f.readlines()

#strip newline characters and leading/ending whitespaces
lines = [j.strip() for j in dataset]
start_index = 0
end_index = -1
#iterate each poker hand
for i in range(len(lines)-1):
    if("Game started at" in lines[i]):  #starting index of hand
        start_index = i
    elif ("Game ended at" in lines[i]): #ending index of hand
        end_index = i + 1
        game_log = lines[start_index:end_index]
        parse_game_log(game_log)

#convert to dataframe
df = pd.DataFrame(ldf, columns=['id', 'start', 'num_players', 'hole_cards','money', 'actions', 'summary', 'end'])
df.to_csv('game_log_data.csv',index=False)
df = pd.read_csv('game_log_data.csv')
os.remove('game_log_data.csv')
dealers = []
big_blinds = []
small_blinds = []
df = df.drop(['id', 'start','end'], axis=1)
pots = []
seat_order = []
money = []
players = []
m_players = []

#extract opponent & game information
for r in df['money']:
    r = r.strip('][').replace("'","").split(', ')
    players = r[1:len(r)]
    h = [(player[0:player.find(':')], player[player.find(':')+2:player.find(' (')],player[player.find(' (')+2:len(player)-2]) if "Seat" in player else ((player[player.find("Player") + 7:player.find("has")-1], player[player.find("has")+4:player.find("blind")+5],player[player.find("blind")+7:len(player)-1]) ) for player in players]
    blinds = h[-2:]
    h = h[:-2]
    seats = [p[0] for p in h]
    player = [p[1] for p in h]
    mon = [p[2] for p in h]
    small_blind = blinds[0][0]
    big_blind = blinds[1][0]
    m_players.append(player)
    seat_order.append(seats)
    money.append(mon)
    big_blinds.append(big_blind)
    small_blinds.append(small_blind)

#update dataframe
df['big_blind'] = big_blinds
df['small_blind'] = small_blinds
df['players'] = m_players
df['money'] = money
df['seat_order'] = seat_order
#get pot information & update dataframe
pots = []
for r in df['summary']:
    r = r.strip('][').replace("'","").split(', ')
    pot = float(r[0].split(" ")[1][:-1])
    pots.append(pot)
df['final_pot'] = pots
#calculate hand potentials & update dataframe
hand_potentials = []
for r in df['hole_cards']:
    r = r.strip('][').replace("'","").split(', ')
    p = [[f[0],f[1]] for f in r]
    hand_potentials.append(calc_hand_potential(p))
df['hand_potential'] = hand_potentials

#class for our player
class MyPlayer:
    def __init__(self,name,money,hand_potential,seat_number,actions,role):
        self.name = name
        self.stack = money
        self.hand_strength = -1
        self.hand_potential = hand_potential
        self.seat_number = seat_number
        self.actions = actions
        self.role = role    #dealer:0,small=1,big=2
    def __str__(self):      #override __str__ method to easily print player information
        return str(self.name) + ", " + str(self.stack) + ", " + str(self.seat_number) + ", " + str(self.hand_potential)
#class for opponents
class StandardPlayer:
    def __init__(self,name,stack, seat_number,actions,role):
        self.name = name
        self.stack = stack
        self.seat_number = seat_number
        self.actions = actions
        self.role = role    #dealer:0,small=1,big=2
    def __str__(self):      #override __str__ method to easily print player information
        return str(self.name) + ", " + str(self.stack) + ", " + str(self.seat_number) + ", " + str(self.role)

#Split each hand into individual actions per stage
evaluator = Evaluator()
data = {}
data['stage'] = []
data['hole_cards'] = []
data['comm_cards'] = []
data['hand_strength'] = []
data['hand_potential'] = []
data['action'] = []
for index, row in df.iterrows():    #iterate hands
    board = []
    hand = []
    holes = row['hole_cards'].replace("[","").replace("]","").replace("'","").split(", ")
    hand = [Card.new(h) for h in holes ]
    action = ''
    #get our player's preflop action
    for act in row['actions'].split(","):
        if "IlxxxlI folds" in act or "IlxxxlI calls" in act or "IlxxxlI bets" in act or "IlxxxlI checks" in act or "IlxxxlI allin" in act or "IlxxxlI raises" in act:
            if "IlxxxlI folds" in act:
                action = 'Player IlxxxlI folds'
            else:
                action = act.replace("[","").replace("]","")
            break

    #print("PREFLOP: ", holes,row['hand_potential'],action)
    #store preflop information
    data['stage'].append('PREFLOP')
    data['hole_cards'].append(str(holes))
    data['hand_potential'].append(row['hand_potential'])
    data['hand_strength'].append(-1)
    data['comm_cards'].append(-1)
    data['action'].append(action)
    if 'folds' in action:       #if our player folds, reinitialize for the next hand
        flop = ''
        turn = ''
        river = ''
        continue
    #get our player's postflop action
    if "FLOP ***" in row['actions']:
        flop = ''
        for act in row['actions'].split(","):
            if "FLOP ***" in act:
                act = act.replace("['***","'***")
                flop = act.replace("10","T").split("[")[1][:-2].split(" ")
                board = [Card.new(f) for f in flop]     #add flop cards to board
                strength = evaluator.get_rank_class(evaluator.evaluate(board, hand))    #calculate hand strength
                potential = 1 - evaluator.get_five_card_rank_percentage(evaluator.evaluate(board, hand))    #calculate hand potential
            #if we have flop cards & our player is still in hand
            if flop != '' and ("IlxxxlI mucks cards" in act or "IlxxxlI folds" in act or "IlxxxlI calls" in act or "IlxxxlI bets" in act or "IlxxxlI checks" in act or "IlxxxlI allin" in act):
                if "IlxxxlI folds" in act:
                    action = 'Player IlxxxlI folds'
                else:
                    action = act.replace("[","").replace("]","")
                #print("FLOP: ",flop,holes,strength,potential,action)
                data['stage'].append('FLOP')
                data['hole_cards'].append(str(holes))
                data['hand_potential'].append(potential)
                data['hand_strength'].append(strength)
                data['comm_cards'].append(str(flop))
                data['action'].append(action)
                break

    if 'folds' in action:
        flop = ''
        turn = ''
        river = ''
        continue
    #get our player's turn action
    if "TURN ***" in row['actions'] and "FLOP ***" in row['actions']:
        turn = ''
        for act in row['actions'].split(","):
            if "TURN ***" in act:
                comm = act.replace("10","T").split("[")[1][:-2].split(" ")
                turn = act.replace("10","T").split(" ")[-1].replace("[","").replace("]","").replace("'","")
                comm.append(turn)
                Card.print_pretty_cards(board)
                board.append(Card.new(turn))
                Card.print_pretty_cards(board)

                strength = evaluator.get_rank_class(evaluator.evaluate(board, hand))
                potential = 1 - evaluator.get_five_card_rank_percentage(evaluator.evaluate(board, hand))
            if turn != '' and ("IlxxxlI mucks cards" in act or"IlxxxlI folds" in act or "IlxxxlI calls" in act or "IlxxxlI bets" in act or "IlxxxlI checks" in act or "IlxxxlI allin" in act):
                if "IlxxxlI folds" in act:
                    action = 'Player IlxxxlI folds'
                else:
                    action = act.replace("[","").replace("]","")
                #print("TURN: ",flop,turn,holes,strength,potential,action)
                data['stage'].append('TURN')
                data['hole_cards'].append(str(holes))
                data['hand_potential'].append(potential)
                data['hand_strength'].append(strength)
                data['comm_cards'].append(str(comm))
                data['action'].append(action)
                break

    if 'folds' in action:
        flop = ''
        turn = ''
        river = ''
        continue
    #get our player's river action
    if "RIVER ***" in row['actions'] and "TURN ***" in row['actions'] and "FLOP ***" in row['actions']:
        river = ''
        for act in row['actions'].split(","):
            if "RIVER ***" in act:
                comm = act.replace("10","T").split("[")[1][:-2].split(" ")
                river = act.replace("10","T").split(" ")[-1].replace("[","").replace("]","").replace("'","")
                comm.append(river)
                Card.print_pretty_cards(board)
                board.append(Card.new(river))
                Card.print_pretty_cards(board)
                strength = evaluator.get_rank_class(evaluator.evaluate(board, hand))
                potential = 1 - evaluator.get_five_card_rank_percentage(evaluator.evaluate(board, hand))
            if river != '' and ("IlxxxlI mucks cards" in act or "IlxxxlI folds" in act or "IlxxxlI calls" in act or "IlxxxlI bets" in act or "IlxxxlI checks" in act or "IlxxxlI allin" in act):
                if "IlxxxlI folds" in row['actions']:
                    action = 'Player IlxxxlI folds'
                else:
                    action = act.replace("[","").replace("]","")
                #print("RIVER: ",flop,turn,river,holes,strength,potential,action)
                data['stage'].append('RIVER')
                data['hole_cards'].append(str(holes))
                data['hand_potential'].append(potential)
                data['hand_strength'].append(strength)
                data['comm_cards'].append(str(comm))
                data['action'].append(action)
                break

    flop = ''
    turn = ''
    river = ''
#convert to dataframe
df = pd.DataFrame (data, columns = ['stage','hole_cards','comm_cards', 'hand_potential','hand_strength','action'])
df.to_csv('game_data.csv',index=False)

def label_data(data_mat):
    done = False
    i = 0
    while not done:
        try:
            if 'Fold' == data_mat[i, 5]:
                data_mat[i, 5] = 'Fold'
            if 'Call' == data_mat[i, 5]:
                data_mat[i, 5] = 'Bet'
            if 'Bet' in data_mat[i, 5]:
                data_mat[i, 5] = 'Bet'
            if 'Check' in data_mat[i, 5]:
                data_mat[i, 5] = 'Check'
            if 'Raise' in data_mat[i, 5]:
                data_mat[i, 5] = 'Bet'
            if 'Allin' in data_mat[i, 5]:
                data_mat[i, 5] = 'Bet'
            i += 1
            if i == data_mat.shape[0]:
                done = True
        except:
            data_mat = np.delete(data_mat, i, 0)
    return data_mat

data = pd.read_csv("game_data.csv")
dataset = data.values
data_mat = data.to_numpy()
data_mat = label_data(data_mat)
df = pd.DataFrame(data_mat,columns = ['stage','hole_cards','comm_cards', 'hand_potential','hand_strength','action'])
df.to_csv('game_data.csv',index=False)
