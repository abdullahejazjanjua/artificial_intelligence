from typing import Optional

class Node:
    def __init__(self, parent: Optional['Node'], board: Optional[list[list[int]]], input_str: Optional[str]):
        self.parent = parent
        if input_str:
            self.board = []
            for _, line in enumerate(filter(None, input_str.splitlines())):
                self.board.append([int(n) for n in line.split()])
        else:
            self.board = board

    def check_constraint(self):


    def available_moves(self):
        pass
