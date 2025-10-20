import copy

from node import Node


class FifteensNode(Node):
    """Extends the Node class to solve the 15 puzzle.

    Parameters
    ----------
    parent : Node, optional
        The parent node. It is optional only if the input_str is provided. Default is None.

    g : int or float, optional
        The cost to reach this node from the start node : g(n).
        In this puzzle it is the number of moves to reach this node from the initial configuration.
        It is optional only if the input_str is provided. Default is 0.

    board : list of lists
        The two-dimensional list that describes the state. It is a 4x4 array of values 0, ..., 15.
        It is optional only if the input_str is provided. Default is None.

    input_str : str
        The input string to be parsed to create the board.
        The argument 'board' will be ignored, if input_str is provided.
        Example: input_str = '1 2 3 4\n5 6 7 8\n9 10 0 11\n13 14 15 12' # 0 represents the empty cell

    Examples
    ----------
    Initialization with an input string (Only the first/root construction call should be formatted like this):
    >>> n = FifteensNode(input_str=initial_state_str)
    >>> print(n)
      5  1  4  8
      7     2 11
      9  3 14 10
      6 13 15 12

    Generating a child node (All the child construction calls should be formatted like this) ::
    >>> n = FifteensNode(parent=p, g=p.g+c, board=updated_board)
    >>> print(n)
      5  1  4  8
      7  2    11
      9  3 14 10
      6 13 15 12

    """

    def __init__(self, parent=None, g=0, board=None, input_str=None):
        # NOTE: You shouldn't modify the constructor
        if input_str:
            self.board = []
            for i, line in enumerate(filter(None, input_str.splitlines())):
                self.board.append([int(n) for n in line.split()])
        else:
            self.board = board

        super(FifteensNode, self).__init__(parent, g)

    def generate_children(self, c = 1):
        """Generates children by trying all 4 possible moves of the empty cell.

        Returns
        -------
            children : list of Nodes
                The list of child nodes.
        """
        children_boards = []

        movements = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        board_size = len(self.board)

        row_idx, col_idx = next((i, j) for i in range(board_size) for j in range(board_size) if self.board[i][j] == 0)

        for (row_increment, col_increment) in movements:
            idx = row_idx + row_increment
            jdx = col_idx + col_increment

            if (0 <= idx < board_size) and (0 <= jdx < board_size):
                updated_board = copy.deepcopy(self.board)
                updated_board[row_idx][col_idx] = updated_board[idx][jdx]
                updated_board[idx][jdx] = 0

                children_boards.append(FifteensNode(parent=self, board=updated_board, g=self.g + 1))
        return children_boards

    def is_goal(self):
        """Decides whether this search state is the final state of the puzzle.

        Returns
        -------
            is_goal : bool
                True if this search state is the goal state, False otherwise.
        """

        return self.board ==  [[1 ,2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]

    def evaluate_heuristic(self):
        """Heuristic function h(n) that estimates the minimum number of moves
        required to reach the goal state from this node.

        Returns
        -------
            h : int or float
                The heuristic value for this state.
        """
        goal_board = [[1 ,2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]
        h = 0
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                element = self.board[i][j]
                if element != 0:
                    goal_position = [(x, y) for x, row in enumerate(goal_board) for y, val in enumerate(row) if val == element][0]
                    h += abs(i - goal_position[0]) + abs(j - goal_position[1])
        return h

    def _get_state(self):
        """Returns an hashable representation of this search state.

        Returns
        -------
            state: tuple
                The hashable representation of the search state
        """
        # NOTE: You shouldn't modify this method.
        return tuple([n for row in self.board for n in row])

    def __str__(self):
        """Returns the string representation of this node.

        Returns
        -------
            state_str : str
                The string representation of the node.
        """
        # NOTE: You shouldn't modify this method.
        sb = []  # String builder
        for row in self.board:
            for i in row:
                sb.append(' ')
                if i == 0:
                    sb.append('  ')
                else:
                    if i < 10:
                        sb.append(' ')
                    sb.append(str(i))
            sb.append('\n')
        return ''.join(sb)


class SuperqueensNode(Node):
    """Extends the Node class to solve the Superqueens problem.

    Parameters
    ----------
    parent : Node, optional
        The parent node. Default is None.

    g : int or float, optional
        The cost to reach this node from the start node : g(n).
        In this problem it is the number of pairs of superqueens that can attack each other in this state configuration.
        Default is 1.

    queen_positions : list of pairs
        The list that stores the x and y positions of the queens in this state configuration.
        Example: [(q1_y,q1_x),(q2_y,q2_x)]. Note that the upper left corner is the origin and y increases downward
        Default is the empty list [].
        ------> x
        |
        |
        v
        y

    n : int
        The size of the board (n x n)

    Examples
    ----------
    Initialization with a board size (Only the first/root construction call should be formatted like this):
    >>> n = SuperqueensNode(n=4)
    >>> print(n)
         .  .  .  .
         .  .  .  .
         .  .  .  .
         .  .  .  .

    Generating a child node (All the child construction calls should be formatted like this):
    >>> n = SuperqueensNode(parent=p, g=p.g+c, queen_positions=updated_queen_positions, n=p.n)
    >>> print(n)
         Q  .  .  .
         .  .  .  .
         .  .  .  .
         .  .  .  .

    """

    def __init__(self, parent=None, g=0, queen_positions=[], n=1):
        # NOTE: You shouldn't modify the constructor
        self.queen_positions = queen_positions
        self.n = n
        super(SuperqueensNode, self).__init__(parent, g)

    def generate_children(self):
        """Generates children by adding a new queen.

        Returns
        -------
            children : list of Nodes
                The list of child nodes.
        """
        # TODO: add your code here
        # You should use self.queen_positions and self.n to produce children.
        # Don't forget to create a new queen_positions list for each child.
        # You can use copy.deepcopy function from the standard library.
        pass

    def is_goal(self):
        """Decides whether all the queens are placed on the board.

        Returns
        -------
            is_goal : bool
                True if all the queens are placed on the board, False otherwise.
        """
        # You should use self.queen_positions and self.n to decide.
        # TODO: add your code here
        pass

    def evaluate_heuristic(self):
        """Heuristic function h(n) that estimates the minimum number of conflicts required to reach the final state.

        Returns
        -------
            h : int or float
                The heuristic value for this state.
        """
        # If you want to design a heuristic for this problem, you should use self.queen_positions and self.n.
        # TODO: add your code here (optional)
        return 0

    def _get_state(self):
        """Returns an hashable representation of this search state.

        Returns
        -------
            state: tuple
                The hashable representation of the search state
        """
        # NOTE: You shouldn't modify this method.
        return tuple(self.queen_positions)

    def __str__(self):
        """Returns the string representation of this node.

        Returns
        -------
            state_str : str
                The string representation of the node.
        """
        # NOTE: You shouldn't modify this method.
        sb = [[' . '] * self.n for i in range(self.n)]  # String builder
        for i, j in self.queen_positions:
            sb[i][j] = ' Q '
        return '\n'.join([''.join(row) for row in sb])
