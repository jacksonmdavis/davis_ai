import streamlit as st
import time
from random import choices
from typing import List, Tuple, Dict, Callable
from copy import deepcopy
from a_star_search import a_star_search, heuristic


COSTS = { '🌾': 1, '🌲': 3, '🪨': 5, '🐊': 7}
TERRAIN = ['🌾', '🌲', '🪨', '🐊', '🌋']
MOVES = [(0, 1), (1, 0), (0, -1), (-1, 0)]
CHARACTER = '😀'
CONFUSED = '🤔'
GOAL = '🎁'

########## Function Defs ##########

# generate_world takes in a height and width and returns a randomly generated 2D list of terrain.
#
# Parameters:
#   height: int, the height of the world
#   width: int, the width of the world
#   terrain: list of single-emoji strings, the possible terrain types
#
# Returns:
#   world: 2D list of single-emoji strings, the randomly generated world
def generate_world(height: int, width: int, terrain: list[str]) -> list[list[str]]:
    world = []
    for i in range(height):
        row = choices(terrain, k=width)
        world.append(row)
    return world

# print_join take in a 2D world list and returns a printable grid of the world.
#
# Parameters:
#   world: 2D list of single-emoji strings, the world to be printed
#
# Returns:
#   print_world: a single string, the printable world
def print_join(world: list[list[str]]) -> str:
    print_world = []
    for row in world:
        print_world.append(' '.join(row))
    print_world = '  \n'.join(print_world)
    return print_world

# pretty_print_path is documented in the a_star_search.py file.  Its only change here is that it returns the world 
# with the path printed on it, rather than printing it directly, for streamlit compatibility.
def pretty_print_path_streamlit( world: List[List[str]], path: List[Tuple[int, int]], start: Tuple[int, int], goal: Tuple[int, int], costs: Dict[str, int]) -> int:
    path_cost = 0
    current_node = start
    transitions = {(1,0):'⏩', (-1,0):'⏪', (0,-1):'⏫', (0,1):'⏬', 'goal':'🎁'} #remember, positive y is DOWN
    world_copy = deepcopy(world)
    for move in path:
        world_copy[current_node[1]][current_node[0]] = transitions[move] #remember, (x,y) is world[y][x]
        current_node = (current_node[0] + move[0], current_node[1] + move[1])
        path_cost += costs[world_copy[current_node[1]][current_node[0]]]
    world_copy[current_node[1]][current_node[0]] = transitions['goal']

    return world_copy, path_cost

# print_explanation prints a quick explanation of the A* algorithm.
# There are no parameters or return values; it is just one big side effect.
def print_explanation():
    left_column, right_column = st.columns(2)

    with left_column:
        st.write('A quick intro:')
        st.write('This is an interactive example of the A* path finding algorithm. It is a heuristic search algorithm that finds the shortest path between two points on a grid. \
                 It is commonly used in video games or mapping software, for example, to find the shortest path between two points on a map.  All controls for this example are \
                 on the left sidebar.')

    with right_column:
        st.write(':blue[Path costs:]')
        st.write(':blue[Plains🌾: 1  \n Forest🌲: 3  \n Mountains 🪨: 5  \n Swamp🐊: 7  \n Volcano🌋: impassible]')

########## End Function Defs ##########


########## Streamlit App ##############

# First initialize session state
if 'status' not in st.session_state:
    st.session_state.status = 'initial'
    st.session_state.world = None
    st.session_state.error = ''
    st.session_state.start = None
    st.session_state.goal = None

# This will always be printed
st.header('Welcome to the :blue[A* path finding algorithm!]')
st.write("This is Franklin: 😀  Today is Franklin's birthday!")
st.write('Help Franklin find his way to his birthday present! 🎁')
st.divider()

# From here on everything is state dependent:

# The initial state is to generate a new world
if st.session_state.status == 'initial':

    st.write(f'First, use sidebar to generate a new world')
    st.sidebar.write('First, generate a new world:')

    in_height = st.sidebar.number_input('World height', min_value=5, max_value=30, value=10, step=1)
    in_width = st.sidebar.number_input('World width', min_value=5, max_value=30, value=10, step=1)

    if st.sidebar.button('Generate new world'):
        height, width = in_height, in_width
        st.session_state.world = generate_world(in_height, in_width, terrain=TERRAIN)
        st.session_state.status = 'world_generated'
        st.rerun()

# The 'world_generated' state allows the user to select start and goal locations
if st.session_state.status == 'world_generated':

    height, width = len(st.session_state.world), len(st.session_state.world[0])
    st.write(print_join(st.session_state.world))

    with st.sidebar.form(key='start_locations'):
        st.write('Select start and goal locations:')
        in_start_x = st.number_input('Start x', min_value=0, max_value=width-1, value=0, step=1)
        in_start_y = st.number_input('Start y', min_value=0, max_value=height-1, value=0, step=1)
        goal_start_x = st.number_input('Goal x', min_value=0, max_value=width-1, value=width-1, step=1)
        goal_start_y = st.number_input('Goal y', min_value=0, max_value=height-1, value=height-1, step=1)
        submitted = st.form_submit_button(label='Submit')
        st.write(st.session_state.error)
    
    if submitted:
        if st.session_state.world[in_start_y][in_start_x] == '🌋':
            st.session_state.error = 'Cannot start on a volcano! Try again.'
            st.rerun()
        
        elif st.session_state.world[goal_start_y][goal_start_x] == '🌋':
            st.session_state.error = 'Goal cannot be a volcano! Try again'
            st.rerun()
        
        elif (in_start_x, in_start_y) == (goal_start_x, goal_start_y):
            st.write('Start and goal cannot be the same! Try again')
            st.rerun()

        else:
            st.session_state.start = (in_start_x, in_start_y)
            st.session_state.goal = (goal_start_x, goal_start_y)
            st.session_state.status = 'start_goal_selected'
            st.session_state.error = ''
            st.rerun()

# The 'start_goal_selected' state prints the world with the start and goal locations displayed
if st.session_state.status == 'start_goal_selected':
    world = st.session_state.world
    height, width = len(world), len(world[0])
    start, goal = st.session_state.start, st.session_state.goal
    print_world = deepcopy(world)

    print_world[start[1]][start[0]] = CHARACTER
    print_world[goal[1]][goal[0]] = GOAL
    print_world = print_join(print_world)
    
    st.write(print_world)
    st.write(f'Start: {start}  |  Goal: {goal}')

    st.sidebar.write('Looks like Franklin is ready to go.')
    find_path = st.sidebar.button('Find that path!')
    if find_path:
        st.session_state.status = 'path_find'
        st.session_state.error = ''
        st.rerun()

# The 'path_find' state runs the A* algorithm and prints the path and path cost
if st.session_state.status == 'path_find':
    world = st.session_state.world
    start, goal = st.session_state.start, st.session_state.goal
    print_world = deepcopy(world)

    path = a_star_search(world, st.session_state.start, st.session_state.goal, COSTS, MOVES, heuristic)

    if path == None:
        st.session_state.error = 'No path possible! Unlucky. Try again with a different world:'
        print_world[start[1]][start[0]] = CONFUSED
        print_world[goal[1]][goal[0]] = GOAL
        print_world = print_join(print_world)
        st.write(print_world)
        st.write(f'Start: {st.session_state.start}  |  Goal: {st.session_state.goal}')
        st.write('Unlucky Franklin! 😢  \n Press reset to try again.')

    else:
        print_world, path_cost = pretty_print_path_streamlit(world, path, st.session_state.start, st.session_state.goal, COSTS)
        print_world[start[1]][start[0]] = CHARACTER
        print_world[goal[1]][goal[0]] = GOAL
        print_world = print_join(print_world)

        st.write(print_world)
        st.write(f'Start: {st.session_state.start}  |  Goal: {st.session_state.goal}')
        st.subheader(f'Ideal path found!  Path cost: {path_cost}')
        st.sidebar.write('Good job Franklin! 🎉  \n Press reset to try again.')
        

    st.sidebar.write(st.session_state.error)
    reset = st.sidebar.button('Reset')
    if reset:
        st.session_state.status = 'initial'
        st.session_state.world = None
        st.session_state.error = ''
        st.rerun()

# This will always be printed
st.divider()
print_explanation()

########## End Streamlit App ##############