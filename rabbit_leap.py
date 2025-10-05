from collections import deque


def get_next_states(state):
    states = []
    s = list(state)
    for i, c in enumerate(s):
        if c == 'E': 
            if i + 1 < len(s) and s[i + 1] == '_':
                new = s.copy()
                new[i], new[i + 1] = new[i + 1], new[i]
                states.append(''.join(new))

            if i + 2 < len(s) and s[i + 1] in ('E', 'W') and s[i + 2] == '_':
                new = s.copy()
                new[i], new[i + 2] = new[i + 2], new[i]
                states.append(''.join(new))

        elif c == 'W': 
            if i - 1 >= 0 and s[i - 1] == '_':
                new = s.copy()
                new[i], new[i - 1] = new[i - 1], new[i]
                states.append(''.join(new))

            if i - 2 >= 0 and s[i - 1] in ('E', 'W') and s[i - 2] == '_':
                new = s.copy()
                new[i], new[i - 2] = new[i - 2], new[i]
                states.append(''.join(new))
    return states


# BFS Solution
def bfs(start, goal):
    queue = deque([start])
    parent = {start: None}
    nodes_explored = 0

    while queue:
        state = queue.popleft()
        nodes_explored += 1
        if state == goal:
            path = []
            while state:
                path.append(state)
                state = parent[state]
            return path[::-1], nodes_explored

        for next_state in get_next_states(state):
            if next_state not in parent:
                parent[next_state] = state
                queue.append(next_state)
    return None, nodes_explored

# DFS Solution
def dfs(start, goal):
    stack = [start]
    parent = {start: None}
    visited = set()
    nodes_explored = 0

    while stack:
        state = stack.pop()
        nodes_explored += 1
        if state == goal:
            path = []
            while state:
                path.append(state)
                state = parent[state]
            return path[::-1], nodes_explored

        visited.add(state)
        for next_state in get_next_states(state):
            if next_state not in visited:
                parent[next_state] = state
                stack.append(next_state)
    return None, nodes_explored


if __name__ == "__main__":
    start = "EEE_WWW"
    goal = "WWW_EEE"

    path_bfs, nodes_bfs = bfs(start, goal)
    print("BFS Solution:")
    for step in path_bfs:
        print(step)
    print("\nTotal steps:", len(path_bfs) - 1)
    print("Nodes explored (BFS):", nodes_bfs)

    path_dfs, nodes_dfs = dfs(start, goal)
    print("\nDFS Solution:")
    for step in path_dfs:
        print(step)
    print("\nTotal steps:", len(path_dfs) - 1)
    print("Nodes explored (DFS):", nodes_dfs) 

