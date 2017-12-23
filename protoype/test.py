import random
import math

import numpy as np
import pandas as pd

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_SELECT_ONE_UNIT = actions.FUNCTIONS.select_unit.id
_USE_STIM = 234


_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_RELATIVE_ENEMY = 4

_PLAYER_SELF = 1

_NOT_QUEUED = [0]
_QUEUED = [1]


ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_ARMY = 'selectarmy'
ACTION_ATTACK = 'attack'
ACTION_MOVE_TO = 'moveto'
ACTION_SELECT_UNIT_TYPE = 'selectunittype'
ACTION_USE_STIM = 'usestim'
ACTION_SELECT_ONE_UNIT = 'selectoneunit'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_ARMY,
    ACTION_ATTACK,
    ACTION_MOVE_TO,
    ACTION_SELECT_UNIT_TYPE,
    ACTION_USE_STIM,
    ACTION_SELECT_ONE_UNIT
]

KILL_UNIT_REWARD = 0.2
_UNITS_ALIVE_REWARD = .0
_UNITS_LOST = -0.1

# Stolen from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions)


    def choose_action(self, observation):
        self.check_state_exist(observation)
        
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]
            
            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            
            action = state_action.values.argmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
            
        return action


    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)
        
        q_predict = self.q_table.ix[s, a]
        q_target = r + self.gamma * self.q_table.ix[s_, :].max()
        
        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)


    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))



class SmartAgent(base_agent.BaseAgent):

    def __init__(self):
        super(SmartAgent, self).__init__()
        
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
        
        self.previous_killed_unit_score = 0
        #self.previous_killed_building_score = 0
        
        self.previous_action = None
        self.previous_state = None
        

    #_PLAYER_REALTIVE gibt an in welcher Beziheung die EInheiten zum Spieler stehen.
    # 1 = eigene Einheiten, 2 = gegenerische Einheiten
    def getUnitTypes(self, obs, player_relative_index):
        units =(obs.observation['screen'][features.SCREEN_FEATURES.player_relative.index] == player_relative_index).nonzero()
        unit_types = obs.observation['screen'][features.SCREEN_FEATURES.unit_type.index]
        unit_type = []

        for i in range(len(units[0])):
            unit_type.append(unit_types[units[0][i]][units[1][i]])

        unit_type, indices = np.unique(unit_type, return_index=True)
        return unit_type


    def getUnitHealthPoint(self, obs, player_relative_index):
        units =(obs.observation['screen'][features.SCREEN_FEATURES.player_relative.index] == player_relative_index).nonzero()
        unit_hp = obs.observation['screen'][features.SCREEN_FEATURES.unit_hit_point_ratio.index]
        hp_average = 0

        for i in range(len(units)):
            hp_average += unit_hp[units[0][i]][units[1][i]]

        hp_average = hp_average / len(units)
        return hp_average


    def getDistance(self, y1, x1, y2, x2):
        return np.sqrt(pow(y1 - y2,2) + pow(x1 - x2,2))


    def getAverageRelativeDistance(self, units):

        distance = 0.0

        for i in range(len(units[0])):
            for j in range(len(units[0])-i):
                distance += self.getDistance(units[0][i], units[1][i], units[0][j], units[1][j])

        relative_distance = distance / len(units)
        return relative_distance


    def step(self, obs):
        super(SmartAgent, self).step(obs)

        unit_type = obs.observation['screen'][_UNIT_TYPE]

        army_supply = obs.observation['player'][5]
        enemy_units =(obs.observation['screen'][features.SCREEN_FEATURES.player_relative.index] == _PLAYER_RELATIVE_ENEMY).nonzero()
        player_units = (obs.observation['screen'][features.SCREEN_FEATURES.player_relative.index] == _PLAYER_SELF).nonzero()
        enemy_type = self.getUnitTypes(obs, _PLAYER_RELATIVE_ENEMY)
        player_unit_type = self.getUnitTypes(obs, _PLAYER_SELF)
        player_average_distance = self.getAverageRelativeDistance(player_units)
        player_average_hit_point = self.getUnitHealthPoint(obs, _PLAYER_SELF)
        enemey_average_hit_point = self.getUnitHealthPoint(obs, _PLAYER_RELATIVE_ENEMY)


        #TODO: enemy_units soll die Anzahl der gegnerischen Einheiten werden
        enemy = [
            enemy_units, #[y-koordinate][x-koordinate]  enemy_units[0] = y-koordinate    enemy_units[1] = x-koordinate
            enemy_type,  # enthält die unit_ids der feindllichen Einheiten
            #remaining_hp # entweder Summe über alle Einheiten oder jeder Einzelnen
            enemey_average_hit_point    #Oder kehrtwert?  Dann (1 - enemey_average_hit_point)
        ]

        player ={
            army_supply,
            player_unit_type,
            player_average_distance,
            player_average_hit_point
        }

        # Score addieren
        # score = killed_units, units_alive, hp_lost, damage_done
        killed_unit_score = obs.observation['score_cumulative'][5]

        # Enemy = {enemy units, enemy types, remaining_HP}
        # Player = {army_supply, unit_types, relative_distance, }
        current_state = [
            enemy,
            player
            #army_supply,
            #enemy_units,
            #unit_type
            #units_selected,
            #hp_remaining
        ]

        #Check if there is something to reward
        if self.previous_action is not None:
            reward = 0

            # Check if the player killed some Units
            if killed_unit_score > self.previous_killed_unit_score:
                reward += KILL_UNIT_REWARD

            if army_supply != self.previous_army_supply:
                if army_supply < self.previous_army_supply:
                    reward -= _UNITS_LOST
                else:
                    reward += _UNITS_ALIVE_REWARD


            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))
        
        rl_action = self.qlearn.choose_action(str(current_state))
        smart_action = smart_actions[rl_action]

        self.previous_army_supply = army_supply
        self.previous_killed_unit_score = killed_unit_score
        self.previous_state = current_state
        self.previous_action = rl_action
        
        if smart_action == ACTION_DO_NOTHING:
            return actions.FunctionCall(_NO_OP, [])
        
        elif smart_action == ACTION_SELECT_ARMY:
            if _SELECT_ARMY in obs.observation['available_actions']:
                return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
        
        elif smart_action == ACTION_ATTACK:
            if _ATTACK_MINIMAP in obs.observation["available_actions"]:
                    return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [32, 32]])
            
        elif smart_action == ACTION_SELECT_ONE_UNIT:
            if _SELECT_ARMY in obs.observation['available_actions']:
                return actions.FunctionCall(_USE_STIM, [_NOT_QUEUED])