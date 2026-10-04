import heapq

from pacman_module.game import Agent, Directions
from pacman_module.util import manhattanDistance


def key(state):
    """Return a unique key identifying a Pacman game state."""
    return (
        state.getPacmanPosition(),
        tuple(state.getFood().asList()),
    )


def heuristic(state):
    """Return a lower-bound estimate of the remaining path cost."""
    pacman_pos = state.getPacmanPosition()
    food = state.getFood().asList()

    if not food:
        return 0

    return max(
        manhattanDistance(pacman_pos, position)
        for position in food
    )


class PacmanAgent(Agent):
    """A Pacman agent based on A* Search."""

    def __init__(self, args):
        self.moves = []

    def get_action(self, state):
        """Return a legal move."""

        if not self.moves:
            self.moves = self.astar(state)

        try:
            return self.moves.pop(0)
        except IndexError:
            return Directions.STOP

    def astar(self, initial_state):
        """Find a path using A* Search."""

        counter = 0
        frontier = []

        heapq.heappush(
            frontier,
            (
                heuristic(initial_state),
                counter,
                initial_state,
                [],
                0,
            ),
        )

        cost_so_far = {key(initial_state): 0}

        while frontier:
            _, _, current_state, actions, cost = heapq.heappop(
                frontier
            )

            current_key = key(current_state)

            if cost > cost_so_far.get(current_key, float("inf")):
                continue

            if current_state.isWin():
                return actions

            for successor, action in (
                current_state.generatePacmanSuccessors()
            ):
                if action == Directions.STOP:
                    continue

                new_cost = cost + 1
                successor_key = key(successor)

                if (
                    successor_key not in cost_so_far
                    or new_cost < cost_so_far[successor_key]
                ):
                    cost_so_far[successor_key] = new_cost
                    counter += 1

                    priority = (
                        new_cost + heuristic(successor)
                    )

                    heapq.heappush(
                        frontier,
                        (
                            priority,
                            counter,
                            successor,
                            actions + [action],
                            new_cost,
                        ),
                    )

        return []