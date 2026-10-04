from pacman_module.game import Agent
from pacman_module.pacman import Directions


def key(state):
    """
    Returns a hashable key that identifies a Pacman game state.
    """
    pacman_pos = state.getPacmanPosition()
    food_pos = tuple(state.getFood().asList())

    return (pacman_pos, food_pos)


class PacmanAgent(Agent):
    """
    A Pacman agent based on Depth-First-Search.
    """

    def __init__(self, args):
        self.moves = []

    def get_action(self, state):
        """
        Given a pacman game state, returns a legal move.
        """

        if not self.moves:
            self.moves = self.dfs(state)

        try:
            return self.moves.pop(0)
        except IndexError:
            return Directions.STOP

    def dfs(self, state):
        """
        Given a pacman game state,
        returns a list of legal moves to solve the search layout.
        """

        if not state.getFood().asList():
            return []

        stack = [(state, [])]
        visited = set()

        while stack:
            current_state, path = stack.pop()

            current_key = key(current_state)

            if current_key in visited:
                continue

            visited.add(current_key)

            if not current_state.getFood().asList():
                return path

            successors = current_state.generatePacmanSuccessors()

            for successor, action in reversed(successors):
                if action == Directions.STOP:
                    continue

                successor_key = key(successor)

                if successor_key not in visited:
                    stack.append(
                        (successor, path + [action])
                    )

        return []