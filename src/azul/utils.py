import numpy as np


def print_wall(wall: np.ndarray, title: str = "") -> None:
    color_map = {
        0: 'B',
        1: 'Y',
        2: 'R',
        3: 'X',
        4: 'O'
    }
    if np.all(wall == -1):
        print(f"{title} VACIO" if title else "VACIO\n")
        return

    if title:
        print(title + "\n")
    row_sep = "+---" * wall.shape[1] + "+"
    for row in wall:
        print(row_sep)
        row_str = ""
        for val in row:
            if val == -1:
                row_str += "|   "
            else:
                row_str += f"| {color_map[val]} "
        row_str += "|"
        print(row_str)
    print(row_sep)