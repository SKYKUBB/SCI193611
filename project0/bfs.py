from collections import deque

from pacman_module.game import Agent, Directions


def key(state):
    """Return a unique key identifying a Pacman game state."""
    return (
        state.getPacmanPosition(),
        tuple(state.getFood().asList()),
        tuple(state.getCapsules()),
    )


class PacmanAgent(Agent):
    """A Pacman agent based on Breadth-First Search."""

    def __init__(self, args):
        """Initialize the Pacman agent."""
        self.moves = []

    def get_action(self, state):
        """Return a legal move for the current game state."""

        if not self.moves:
            self.moves = self.bfs(state)

        try:
            return self.moves.pop(0)
        except IndexError:
            return Directions.STOP

    def bfs(self, state):
        """Return a list of moves found using Breadth-First Search."""

        frontier = deque([(state, [])])
        visited = {key(state)}

        while frontier:
            current_state, actions = frontier.popleft()

            if current_state.isWin():
                return actions

            for successor, action in current_state.generatePacmanSuccessors():
                if action == Directions.STOP:
                    continue

                state_key = key(successor)

                if state_key not in visited:
                    visited.add(state_key)
                    frontier.append(
                        (successor, actions + [action])
                    )

        return []