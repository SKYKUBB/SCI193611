from pacman_module.game import Agent, Directions


def key(state):
    """Return a unique key representing the game state."""
    pacman_pos = state.getPacmanPosition()
    ghost_positions = tuple(state.getGhostPositions())
    food_positions = tuple(state.getFood().asList())

    return (pacman_pos, ghost_positions, food_positions)


class PacmanAgent(Agent):
    """Pacman agent using the Minimax algorithm."""

    def __init__(self):
        super().__init__()

    def get_action(self, state):
        """Return the best action according to Minimax."""
        successors = state.generatePacmanSuccessors()

        if not successors:
            return Directions.STOP

        best_score = float("-inf")
        best_action = Directions.STOP
        path = {key(state)}

        for successor, action in successors:
            if action == Directions.STOP:
                continue

            score = self.minimax(successor, 1, path)

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def minimax(self, state, agent_index, path):
        """Return the Minimax value of a state."""
        if state.isWin() or state.isLose():
            return state.getScore()

        state_key = key(state)

        if state_key in path:
            return state.getScore()

        new_path = path | {state_key}

        num_agents = state.getNumAgents()
        next_agent = (agent_index + 1) % num_agents

        if agent_index == 0:
            successors = state.generatePacmanSuccessors()
        else:
            successors = state.generateGhostSuccessors(agent_index)

        successors = [
            (successor, action)
            for successor, action in successors
            if action != Directions.STOP
        ]

        if not successors:
            return state.getScore()

        if agent_index == 0:
            return max(
                self.minimax(successor, next_agent, new_path)
                for successor, _ in successors
            )

        return min(
            self.minimax(successor, next_agent, new_path)
            for successor, _ in successors
        )