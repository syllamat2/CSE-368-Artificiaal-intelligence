# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 14:08:19 2017
Author: Nils Napp
Modified by: Jiwon Choi
"""

import numpy as np
import random
from agentinteraction import AgentInteraction
from drawprob import plotQQual

''' 
Build you own Q-Learning agent!
'''

''' 
for type hinting return types of fuctions
actions are strings
'''
action_t = str


class RLAgent:

    def __init__(self,
                 interaction: AgentInteraction,
                 alpha: float = 0.1):
        '''
        Interact with the world
        Normally all you would get are:
            
            1) a way to execute actions 
            2) the resulting next state 
            3) the resulting reward
            
        here you also get some additional informaiton, like the optimal actions
        that allow you to evaluate how well the RL agent is doing
        '''
        self.interaction = interaction
        self.alpha = alpha
        self.gamma = interaction.mdp.gamma

        '''
        set up the Q function based on the states and actions from the interaciton
        '''
        self.Q = np.zeros((interaction.mdp.stateSize, len(interaction.mdp.actions)))

        self.state = interaction.state
        self.actions = interaction.mdp.actions

        '''
        dict to translate the action names and indecies for entering into Q
        '''

        self.actionIdx = dict(zip(self.actions, range(len(self.actions))))
        self.idxAction = dict(zip(range(len(self.actions)), self.actions))

        self.qualityQhistory = []

    '''
    Returns the possible actions from  a given state        
    '''

    def availableActions(self):

        actlist = self.interaction.mdp.maze.actionList(self.state)
        actlist.append('stop')
        return actlist

    '''
    Return an epsilon-greedy action using the current esimate of Q 
    '''

    def epsGreedy(self, eps: float = 0.1) -> action_t:
        p = random.random()
        if p < eps:
            j = np.random.choice(len(self.availableActions()))
            a = self.availableActions()[j]
        else:
            s: int = self.state
            a = self.bestAction(s)
        return a

    '''
    Return the best action based on the current estimate of Q
    uses the action list from 'interaction'        
    '''

    def bestAction(self, state: int) -> action_t:
        s = self.Q[state]
        idx = np.argmax(s)
        a = self.actions[idx]
        return a


    '''
    Take a Q-Learning Step
    It should execute an epislon-greedty action
    Update the Q function
    And then update the RLAgent state to the new state 
    '''

    def qstep(self):
        s: int = self.state
        a = self.epsGreedy(0.1)
        nextState_and_reward = self.interaction.takeAction(a)
        nextState: int = nextState_and_reward[0]
        reward: int = nextState_and_reward[1]
        nextStateAction = self.bestAction(nextState)
        findIndex_currentAction = [i for i in range(len(self.actions)) if self.actions[i] == a]
        idx_currentAction = findIndex_currentAction[0]
        findIndex_nextStateAction = [i for i in range(len(self.actions)) if self.actions[i] == nextStateAction]
        idx_nextStateAction = findIndex_nextStateAction[0]
        sample = reward + self.Q[nextState][idx_nextStateAction]
        self.Q[s][idx_currentAction] = (1-self.alpha)*self.Q[s][idx_currentAction] + self.gamma*sample






    '''
    Run a training episode for steps number of steps
    Append the quality of the Q policy to slef.qualityQhistory  
    '''

    def trainingEpisode(self, steps: int = 20):
        self.interaction.reset()
        for i in range(steps):
            self.qstep()
            self.state = self.interaction.takeAction((self.bestAction(self.state)))[0]
        self.qualityQhistory.append(self.evalQ())

    '''
    reset the history of Q quality
    run a number of training episdoes 
    '''

    def train(self, episodes: int = 3000):
        self.qualityQhistory.clear()
        for i in range(episodes):
            self.trainingEpisode(20)

    '''
    Compare the current policy from Q to the policy computed from the MDP
    '''

    def evalQ(self) -> float:

        places = 0
        match = 0
        for s in range(self.interaction.mdp.stateSize):
            action = self.bestAction(s)
            if not self.interaction.mdp.maze.isWall(s):
                places = places + 1
                if action == self.interaction.optimalPolicy[s]:
                    match = match + 1

        return 1.0 * match / places

    '''
    Print the current policy to teh consolse
    '''

    def showPolicy(self):
        row, col = self.interaction.mdp.maze.worldShape
        cellstr = '{!s: ^7}'

        policylist = []

        for s in range(self.interaction.mdp.stateSize):
            policylist.append(self.bestAction(s))

        policylist.reverse()

        for i in range(row):
            for j in range(col):
                element = policylist.pop()
                if not self.interaction.mdp.maze.world[i, j] == 0:
                    element = '#'
                print(cellstr.format(element), end='')
            print('')


if __name__ == '__main__':
    '''
    Set up a world and give an interaface for the 
    q-learning agent to interact with
    '''
    worldInteraction = AgentInteraction()
    QAgent = RLAgent(worldInteraction)

    '''
    Run a few episodes and show the quality before and after
    
    '''

    # Thise should all be 'up' since that is the one chosen in a tie breakker
    print('\n\n{!s:-^70}'.format('Q-policy after intialization'))
    QAgent.showPolicy()

    QAgent.trainingEpisode()
    QAgent.trainingEpisode()
    QAgent.trainingEpisode()

    print('\n\n{!s:-^70}'.format('After running 3 episodes'))
    QAgent.showPolicy()

    QAgent.train()
    plotQQual(QAgent.qualityQhistory)

    print('\n\n{!s:-^70}'.format('Optimal Policy from MDP'))
    QAgent.interaction.mdp.showPolicy()

    print('\n\n{!s:-^70}'.format('Q-learning policy after training'))
    QAgent.showPolicy()
