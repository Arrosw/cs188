# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    result_list = [] #返回结果
    close_set = set() #已经遍历过的点
    pre_dict = dict() #上一个点，用于确定路径

    start_state = problem.getStartState()#起始状态

    stack = util.Stack()#dfs用的栈
    stack.push(start_state)

    while (not stack.isEmpty()):
        
        temp = stack.pop()
        close_set.add(temp)

        if problem.isGoalState(temp):
            
            while (temp != start_state):
                result_list.append(pre_dict[temp][1])
                temp = pre_dict[temp][0]

            result_list.reverse()    
            break
            
        successors = problem.getSuccessors(temp)
        for get_successor in successors:
            successor = get_successor[0]
            action = get_successor[1]

            if successor not in close_set : 
                stack.push(successor)
                pre_dict[successor] = (temp,action)

    return result_list

    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    result_list = []
    close_set = set()

    pre_dict = dict()

    queue = util.Queue()
    start_state = problem.getStartState()

    queue.push(start_state)
    close_set.add(start_state)

    while (not queue.isEmpty()):
        temp = queue.pop()
        # close_set.add(temp)

        if problem.isGoalState(temp):
            while temp != start_state:     
                result_list.append(pre_dict[temp][1])
                temp = pre_dict[temp][0]                
            
            result_list.reverse()        
            break

        for get_successor in problem.getSuccessors(temp):
            successor = get_successor[0]
            action = get_successor[1]

            if successor in close_set : continue
        
            queue.push(successor)
            close_set.add(successor)
            pre_dict[successor] = (temp,action)

    return result_list
    
    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    
    priority_queue = util.PriorityQueue()
    close_set = set()
    result_list = []
    pre_dict = dict()

    start_state = problem.getStartState()
    priority_queue.push(start_state,0)
    pre_dict[start_state] = (0,'',0)

    while not priority_queue.isEmpty():
        temp = priority_queue.pop()
        close_set.add(temp)

        if problem.isGoalState(temp):
            while temp != start_state:
                result_list.append(pre_dict[temp][1])
                temp = pre_dict[temp][0]
            
            result_list.reverse()
            break

        pre_cost = pre_dict[temp][2] #if temp in pre_dict else 0
        for get_successor in problem.getSuccessors(temp):
            successor, action, cost = get_successor
            backward_cost = cost + pre_cost
   
            if successor in close_set : continue

            if successor in pre_dict :
                if pre_dict[successor][2] > backward_cost :
                    pre_dict[successor] = (temp,action,backward_cost)
                    priority_queue.update(successor,backward_cost)
                
            else:
                pre_dict[successor] = (temp,action,backward_cost)
                priority_queue.push(successor,backward_cost)

    print(result_list)
    print(len(result_list))
    return result_list    
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    
    #Heuristics take two arguments:
    #a state in the search problem (the main argument), 
    #the problem itself (for reference information).

    result_list = []
    close_set = set()
    pre_dict = dict()
    priority_queue = util.PriorityQueue()

    start_state = problem.getStartState()
    priority_queue.push(start_state,0)
    pre_dict[start_state] = (0,'',0)

    while not priority_queue.isEmpty():
        temp = priority_queue.pop()
        close_set.add(temp)

        if problem.isGoalState(temp):
            while temp != start_state:
                result_list.append(pre_dict[temp][1])
                temp = pre_dict[temp][0]

            result_list.reverse()
            break
            
        pre_cost = pre_dict[temp][2] - heuristic(temp,problem)
        for get_successor in problem.getSuccessors(temp):
            successor, action, cost = get_successor

            if successor in close_set : continue

            backward_cost = pre_cost + cost
            forward_cost = heuristic(successor,problem)
            total_cost = backward_cost + forward_cost
            
            if successor in pre_dict:
                if total_cost < pre_dict[successor][2]:
                    pre_dict[successor] = (temp,action,total_cost)
                    priority_queue.update(successor,total_cost)

            else:
                pre_dict[successor] = (temp,action,total_cost)
                priority_queue.push(successor,total_cost)

    print(result_list)
    print(len(result_list))
    return result_list

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
