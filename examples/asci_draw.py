board = {
    (0, -1): [('Player1', 'Beetle')],
    (-2, 0): [('Player1', 'Queen')],
    (-3, 1): [('Player2', 'Ant')],
    (-3, 2): [('Player2', 'Queen')],
    (-1, -1): [('Player1', 'Beetle')],
    (-2, 3): [('Player2', 'Ant'), ('Player2', 'Beetle')],
    (-2, 2): [('Player1', 'Ant')],
    (-1, 1): [('Player1', 'Spider')],
    (-4, 3): [('Player2', 'Beetle')],
    (-4, 2): [('Player1', 'Grasshopper')],
    (-2, -1): [('Player1', 'Spider')],
    (-2, 4): [('Player2', 'Spider')],
    (-5, 4): [('Player2', 'Grasshopper')],
    (-2, -2): [('Player2', 'Ant')],
    (-1, -3): [('Player1', 'Ant')],
    (-3, -2): [('Player2', 'Spider')],
    (-3, 3): [('Player2', 'Grasshopper')],
    (-3, -1): [('Player1', 'Ant')],
    (0, -2): [('Player1', 'Grasshopper')],
    (-2, 1): [('Player2', 'Grasshopper')]
}

# 1) Find bounding box
xs = [x for (x,y) in board.keys()]
ys = [y for (x,y) in board.keys()]
min_x, max_x = min(xs), max(xs)
min_y, max_y = min(ys), max(ys)

# 2) Build a grid. We'll store strings at each (x,y)
grid = {}
for (x, y), contents in board.items():
    # Use first occupant as a label, or combine them
    label = ""
    for (player, piece) in contents:
        label += f"{player[0]}{piece[0]} "  # e.g. "P1B " for Player1 Beetle
    label = label.strip()
    grid[(x,y)] = label

# 3) Print from top row (max_y) down to bottom (min_y).
# We'll add a small offset to mimic hex staggering
for row_y in range(max_y, min_y - 1, -1):
    row_str = ""
    for col_x in range(min_x, max_x + 1):
        # Add some indentation based on y to stagger columns
        # Just a small trick so that neighbors visually line up better
        indent = " " if (row_y % 2 != 0) else ""
        cell_label = grid.get((col_x,row_y), "..")
        row_str += f"{indent}{cell_label:5s}"  # 5-char fixed width
    print(f"y={row_y:2d} | {row_str}")

print("\nLegend:")
print("  '..' means empty.")
print("  'P1B' = Player1 Beetle, 'P2Q' = Player2 Queen, etc.")

