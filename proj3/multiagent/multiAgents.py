# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        def Manhattan(x1,y1,x2,y2):
            s = abs(x1-x2) + abs(y1-y2) + 1
            return s if s > 1 else 0.01

        score = 0
        Foods = list(newFood)
        Foods = [(i,j) for i in range(len(Foods)) for j in range(len(Foods[i])) if Foods[i][j] == True]
        x,y = newPos

        # add score of food
        for food in Foods:
            score += 1/Manhattan(food[0],food[1],x,y)
        score -= len(Foods)

        # add score of Ghost             
        newGhostPos = [ghost.getPosition() for ghost in newGhostStates]
        score -= min([1/Manhattan(x,y,i,j) for i,j in newGhostPos])

        # add score of scared time 
        for time in newScaredTimes:
            score += time 

        return score
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        numGhosts = gameState.getNumAgents() - 1
        maxDepth = self.depth
        INF = 1000000

        def maxValue(gameState: GameState, depth = 1):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState),None
            if depth > maxDepth: 
                return self.evaluationFunction(gameState),None
            
            value = -INF
            move = ''

            actions = gameState.getLegalActions()
            for action in actions:
                v = minValue(gameState.generateSuccessor(0,action), depth)
                if v > value:
                    value = v
                    move = action

            return value, move

        def minValue(gameState: GameState, depth, ghostIndex = 1):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            value = INF

            actions = gameState.getLegalActions(agentIndex=ghostIndex)
            if ghostIndex == numGhosts:
                if depth > maxDepth: 
                    return self.evaluationFunction(gameState)
                for action in actions:
                    v,_ = maxValue(gameState.generateSuccessor(ghostIndex,action), depth+1)
                    if v < value:
                        value = v

            else:
                for action in actions:
                    v = minValue(gameState.generateSuccessor(ghostIndex,action), depth, ghostIndex+1)
                    if v < value:
                        value = v

            return value
            
        return maxValue(gameState)[1]
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numGhosts = gameState.getNumAgents() - 1
        maxDepth = self.depth
        INF = 1000000

        def maxValue(gameState: GameState, alpha=-INF, beta=INF, depth=1):

            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState),''
            if depth > maxDepth: 
                return self.evaluationFunction(gameState),''
            
            value = -INF
            move = ''

            actions = gameState.getLegalActions()
            for action in actions:
                v = minValue(gameState.generateSuccessor(0,action), alpha, beta, depth)
                if v > value:
                    value = v
                    move = action
                    if value > beta: return value, move
                alpha = max(alpha,v)

            return value, move

        def minValue(gameState: GameState, alpha, beta, depth=1, ghostIndex=1):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            value = INF

            actions = gameState.getLegalActions(agentIndex=ghostIndex)
            if ghostIndex == numGhosts:
                for action in actions:
                    v,_ = maxValue(gameState.generateSuccessor(ghostIndex,action), alpha, beta, depth+1)
                    if v < value:
                        value = v
                        if value < alpha: return value
                    beta = min(v,beta)      

            else:
                for action in actions:
                    v = minValue(gameState.generateSuccessor(ghostIndex,action), alpha, beta, depth, ghostIndex+1)
                    if v < value:
                        value = v
                        if value < alpha: return value
                    beta = min(v,beta)

            return value
            
        return maxValue(gameState)[1]

        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        numGhost = gameState.getNumAgents()-1
        maxDepth = self.depth
        INF = 1000000

        def maxValue(gameState: GameState, depth = 1):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), ''
            if depth > maxDepth:
                return self.evaluationFunction(gameState), '' 
            
            actions = gameState.getLegalActions()
            value = -INF
            move = ''
            
            for action in actions:
                v = expectValue(gameState.generateSuccessor(0,action),depth)
                if v > value:
                    value = v
                    move = action

            return value, move

        def expectValue(gameState: GameState, depth, ghostIndex = 1):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            v = 0
            actions = gameState.getLegalActions(ghostIndex)
            if ghostIndex == numGhost:
                for action in actions:
                    v += maxValue(gameState.generateSuccessor(ghostIndex,action), depth+1)[0] / len(actions)
                    
            else:
                for action in actions:
                    v += expectValue(gameState.generateSuccessor(ghostIndex,action), depth, ghostIndex+1) / len(actions)
                    
            return v
        
        return maxValue(gameState)[1]
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    def Manhattan(x1,y1,x2,y2):
        s = abs(x1-x2) + abs(y1-y2) + 1
        return s if s > 1 else 0.001
    
    x,y = currentGameState.getPacmanPosition()
    newGhostStates = currentGameState.getGhostStates()
    newGhostPos = [ghost.getPosition() for ghost in newGhostStates]
    
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    foods = list(currentGameState.getFood())
    foods = [(i,j) for i in range(len(foods)) for j in range(len(foods[i])) if foods[i][j] == True]
    foods_num = len(foods)


    score = 0
    Foods = foods

    # add score of food
    for food in Foods:
        score += 1/Manhattan(food[0],food[1],x,y)
    score -= 0.8*len(Foods)

    # add score of Ghost    
    newGhostPos = [ghost.getPosition() for ghost in newGhostStates]   
    score -= min([1/Manhattan(x,y,i,j) for i,j in newGhostPos])


    # add score of scared time 
    for time in newScaredTimes:
        score += 2*time 
    
    return score

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
