
# ```
# token   terrain    cost 
# 🌾       plains     1
# 🌲       forest     3
# ⛰       hills      5
# 🐊       swamp      7
# 🌋       mountains  impassible
# ```
# 

full_world = [
['🌾', '🌾', '🌾', '🌾', '🌾', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🌾', '🌋', '🌋', '🌋', '🌋', '🌋', '🌋', '🌋', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🌾', '🌋', '🌋', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌋', '🌋', '🌋', '⛰', '⛰', '⛰', '🌋', '🌋', '⛰', '⛰'],
['🌾', '🌾', '🌾', '🌾', '⛰', '🌋', '🌋', '🌋', '🌲', '🌲', '🌲', '🌲', '🐊', '🐊', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🌾', '⛰', '⛰', '🌋', '🌋', '⛰', '🌾'],
['🌾', '🌾', '🌾', '⛰', '⛰', '🌋', '🌋', '🌲', '🌲', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🌲', '🌲', '🌲', '🌾', '🌾', '🌾', '⛰', '🌋', '🌋', '🌋', '⛰', '🌾'],
['🌾', '⛰', '⛰', '⛰', '🌋', '🌋', '⛰', '⛰', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '🌋', '⛰', '🌾', '🌾'],
['🌾', '⛰', '⛰', '🌋', '🌋', '⛰', '⛰', '🌾', '🌾', '🌾', '🌾', '⛰', '🌋', '🌋', '🌋', '🐊', '🐊', '🐊', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '🌾', '🌾', '🌾'],
['🌾', '🌾', '⛰', '⛰', '⛰', '⛰', '⛰', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '🌋', '🌋', '🌋', '🐊', '🐊', '🐊', '🌾', '🌾', '⛰', '⛰', '⛰', '🌾', '🌾'],
['🌾', '🌾', '🌾', '⛰', '⛰', '⛰', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '⛰', '🌋', '🌋', '🌾', '🐊', '🐊', '🌾', '🌾', '⛰', '⛰', '⛰', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🌾', '🌾', '⛰', '⛰', '⛰', '🌋', '🌋', '🌋', '🌋', '🌾', '🌾', '🌾', '🐊', '🌾', '⛰', '⛰', '⛰', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🌾', '⛰', '⛰', '🌋', '🌋', '🌋', '⛰', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '🌋', '🌋', '🌋', '⛰', '🌾', '🌾', '🌾'],
['🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '⛰', '🌋', '🌋', '⛰', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🌾', '🌾', '⛰', '🌋', '🌋', '⛰', '🌾', '🌾', '🌾'],
['🐊', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '⛰', '⛰', '🌋', '🌋', '⛰', '🌾', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '🌾', '⛰', '🌋', '⛰', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '⛰', '🌲', '🌲', '⛰', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '⛰', '🌾', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🌾', '🌋', '🌾', '🌾', '🌲', '🌲', '🌲', '🌲', '⛰', '⛰', '⛰', '⛰', '🌾', '🐊', '🐊', '🐊', '🌾', '🌾', '⛰', '🌋', '⛰', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🌋', '🌋', '🌋', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌋', '🌋', '🌋', '⛰', '⛰', '🌾', '🐊', '🌾', '⛰', '🌋', '🌋', '⛰', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌋', '🌋', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌋', '🌋', '🌋', '🌾', '🌾', '🌋', '🌋', '🌋', '🌾', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🌋', '🌋', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌋', '🌋', '🌋', '🌋', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🌋', '🌋', '🌋', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🌾', '🌾', '⛰', '⛰', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🌾', '🌋', '🌋', '🌋', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊'],
['🌾', '🌾', '⛰', '⛰', '⛰', '⛰', '🌋', '🌋', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🌋', '🌾', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊'],
['🌾', '🌾', '🌾', '🌾', '⛰', '⛰', '⛰', '🌋', '🌋', '🌋', '🌲', '🌲', '🌋', '🌋', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊'],
['🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '⛰', '⛰', '🌋', '🌋', '🌋', '🌋', '🌾', '🌾', '🌾', '🌾', '⛰', '⛰', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊'],
['🌾', '⛰', '⛰', '🌾', '🌾', '⛰', '⛰', '⛰', '⛰', '⛰', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '⛰', '🌋', '🌋', '⛰', '⛰', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊'],
['⛰', '🌋', '⛰', '⛰', '⛰', '⛰', '🌾', '🌾', '🌾', '🌾', '🌾', '🌋', '🌋', '🌋', '⛰', '⛰', '🌋', '🌋', '🌾', '🌋', '🌋', '⛰', '⛰', '🐊', '🐊', '🐊', '🐊'],
['⛰', '🌋', '🌋', '🌋', '⛰', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '⛰', '🌋', '🌋', '🌋', '🌋', '⛰', '⛰', '⛰', '⛰', '🌋', '🌋', '🌋', '🐊', '🐊', '🐊', '🐊'],
['⛰', '⛰', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '⛰', '⛰', '⛰', '⛰', '🌾', '🌾', '🌾', '🌾', '⛰', '⛰', '⛰', '🌾', '🌾', '🌾']
]

small_world = [
    ['🌾', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲'],
    ['🌾', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲'],
    ['🌾', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲'],
    ['🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾'],
    ['🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾'],
    ['🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾'],
    ['🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾']
]

MOVES = [(0,-1), (1,0), (0,1), (-1,0)]

COSTS = { '🌾': 1, '🌲': 3, '⛰': 5, '🐊': 7}

from typing import List, Tuple, Dict, Callable
from copy import deepcopy

# ## heuristic
# 
# `heuristic` is a helper function for the A_star algorithm that returns an estimate of the cost to get to the goal 
# state from a given input state.  For the A_star search to work optimally, the heuristic should be admissable and consistent.
# Non-admissable heuristics can also work for some problems if care is taken.  In this case simple Manhattan distance is
# used, which is admissable and consistent as long as none of the movement costs are less than one, and diagonal movement 
# is not permitted.
# 
# **Used by**: [a_star_search](#a_star_search)
# 
# * **state** tuple: the current position that the heuristic will evaluate relative to the goal state.
# * **goal** tuple: the location of the goal state. 
# 
# **returns** tuple: an estimate of the cost to get from 'state' to 'goal'
def heuristic(state: Tuple[int, int], goal: Tuple[int, int]): # you can add formal parameters
    x = abs(state[0]-goal[0])
    y = abs(state[1]-goal[1])
    return x+y


# ## return_path
# 
# `return_path` is a helper function for the A_star algorithm that unwinds nested nodes into a list of states representing the path as a series of transitions (NOT positions, as I originally implemented and fortunately fixed).  It should be run at the completion of the algorithm for path return and preparing for pretty_print_path.
# 
# **Used by**: [a_star_search](#a_star_search)
# 
# * **goal_node** `dict[str: tuple, str: dict]`: the node representing the goal state.  Keys must contain 'position' and 'parent'.
# 
# **returns** `list[tuple]`: an estimate of the cost to get from 'state' to 'goal'
def return_path(goal_node: dict[str: tuple, str: dict])-> list[tuple[int,int]]:
    cur_node = deepcopy(goal_node)
    path = []

    while cur_node['parent'] is not None:
        path.append((cur_node['position'][0] - cur_node['parent']['position'][0], cur_node['position'][1] - cur_node['parent']['position'][1]))
        cur_node = cur_node['parent']
    
    path.reverse()
    return path

# ## successors
# 
# `successors` is a helper function for the A_star algorithm that 
# 
# **Used by**: [a_star_search](#a_star_search)
# 
# * **goal_node** ``: 
# 
# **returns** ``: 
def successors(state, world, moves):
    successors = []
    for move in moves:
        new_state = (state[0] + move[0], state[1] + move[1])
        # Remember (x,y) is world[y][x]
        if 0 <= new_state[0] <= (len(world[0]) - 1) and 0 <= new_state[1] <= (len(world) - 1):
            if world[new_state[1]][new_state[0]] != '🌋':
                successors.append(new_state)
    return successors


# ## frontier_insert
# 
# `frontier_insert` is the function in the A_star algorithm that inserts nodes into and maintains the frontier list.  The implementation of this function essentially determines what type of search is being performed; in this case maintaining a so-called "priority queue" sorted by the total current path cost plus the heuristic cost-- $g(n) + h(n)$ --makes it an A* search.
# 
# This particular implementation sorts by h(n) as a secondary key, with the intention that if there are multiple equal-cost nodes on the frontier, the one closest to the goal will be expanded first, hopefully slightly increasing efficiency in the case of many equal-cost paths.  Implementing a proper priority queue instead of a list that sorts itself every time is a very desirable future project.
# 
# **Used by**: [a_star_search](#a_star_search)
# 
# * **frontier** `list[dict]`: The current list of frontier nodes ready to be explored.
# * **node** `dict`: The node to be inserted into the frontier.
# 
# **returns** `list[dict]`: The updated frontier with the input node inserted.
def frontier_insert(frontier: list[dict], node: dict) -> list[dict]:
    frontier_copy = deepcopy(frontier)
    frontier_copy.append(node)
    frontier_copy.sort(key=lambda x: (x['g'] + x['h'], x['h']))
    return frontier_copy

# ## pretty_print_path
# 
# `pretty_print_path` is a function that accepts a grid world along with a path from `start` to `goal` and returns the cost of that path by looking up the cost of each position reached.  It also prints the world with the transitions traced on it forming a visual path to the goal.
# 
# **note that `pretty_print_path` has side effects (the printing)**
# 
# * **world** List[List[str]]: the world (terrain map) for the path to be printed upon.
# * **path** List[Tuple[int, int]]: the path from start to goal, in offsets.
# * **start** Tuple[int, int]: the starting location for the path.
# * **goal** Tuple[int, int]: the goal location for the path.
# * **costs** Dict[str, int]: the costs for each action.
# 
# **returns** int - The path cost.
def pretty_print_path( world: List[List[str]], path: List[Tuple[int, int]], start: Tuple[int, int], goal: Tuple[int, int], costs: Dict[str, int]) -> int:
    path_cost = 0
    current_node = start
    transitions = {(1,0):'⏩', (-1,0):'⏪', (0,-1):'⏫', (0,1):'⏬', 'goal':'🎁'} #remember, positive y is DOWN
    world_copy = deepcopy(world)
    for move in path:
        world_copy[current_node[1]][current_node[0]] = transitions[move] #remember, (x,y) is world[y][x]
        current_node = (current_node[0] + move[0], current_node[1] + move[1])
        path_cost += costs[world_copy[current_node[1]][current_node[0]]]
    world_copy[current_node[1]][current_node[0]] = transitions['goal']

    for line in world_copy:
        print("".join(line))

    return path_cost


# ## a_star_search
# 
# `a_star_search` uses the 'A*' graph search algorithm to search for an optimal (or near-optimal) path through a given state space.  A* is a type of informed
# search where nodes are expanded in best-first order based on the sum of each node's individual movement cost plus its cost as estimated by a heuristic function 
# (typically represented as $g(n) + h(n)$).  
# 
# **Uses:** [heuristic](#heuristic), [return_path](#return_path), [successors](#successors), [frontier_insert](#frontier_insert)
# 
# * **world** List[List[str]]: the actual context for the navigation problem.
# * **start** Tuple[int, int]: the starting location of the bot, `(x, y)`.
# * **goal** Tuple[int, int]: the desired goal position for the bot, `(x, y)`.
# * **costs** Dict[str, int]: is a `Dict` of costs for each type of terrain in **world**.
# * **moves** List[Tuple[int, int]]: the legal movement model expressed in offsets in **world**.
# * **heuristic** Callable: is a heuristic function, $h(n)$.
# 
# **returns** List[Tuple[int, int]]: the offsets needed to get from start state to the goal as a `List`.
def a_star_search( world: List[List[str]], start: Tuple[int, int], goal: Tuple[int, int], costs: Dict[str, int], moves: List[Tuple[int, int]], heuristic: Callable) -> List[Tuple[int, int]]:
    frontier = [{'position': start, 'parent': None, 'g': 0, 'h': heuristic(start, goal)}]
    explored = []

    while frontier:
        current_state = frontier.pop(0)
        if current_state['position'] == goal:
            return return_path(current_state)
        
        children = successors(state=current_state['position'], world=world, moves=moves)
        for child in children:
            if child not in explored and child not in [node['position'] for node in frontier]:
                child = {'position': child, 'parent': current_state, 'g': current_state['g'] + costs[world[child[1]][child[0]]], 'h': heuristic(child, goal)} # Remember (x,y) is world[y][x]
                frontier = frontier_insert(frontier, child)
        explored.append(current_state['position'])
    
    return None # If no path found