from pacman_module.game import Agent, Directions
from pacman_module.util import manhattanDistance


class PacmanAgent(Agent):
    """Pacman agent using depth-limited H-Minimax."""

    def __init__(self, depth=2):
        super().__init__()
        self.depth = int(depth)
        self.history = []
        self.position_counts = {}

    def get_state_key(self, state):
        pacman_pos = state.getPacmanPosition()
        ghost_pos = tuple(state.getGhostPositions())
        food_pos = tuple(state.getFood().asList())

        return (pacman_pos, ghost_pos, food_pos)

    def evaluation_function(self, state):
        if state.isWin():
            return 999999

        if state.isLose():
            return -999999

        pacman_pos = state.getPacmanPosition()
        food = state.getFood().asList()
        ghosts = state.getGhostPositions()

        score = state.getScore()

        # Prefer eating food.
        score -= 80 * len(food)

        if food:
            food_dist = min(
                manhattanDistance(pacman_pos, food_pos)
                for food_pos in food
            )
            score -= 2 * food_dist

        # Avoid ghosts.
        for ghost_pos in ghosts:
            distance = manhattanDistance(
                pacman_pos,
                ghost_pos
            )

            if distance == 0:
                score -= 10000
            elif distance == 1:
                score -= 2000
            elif distance == 2:
                score -= 600
            elif distance == 3:
                score -= 150
            elif distance == 4:
                score -= 50

        # Avoid repeatedly visiting the same position.
        visit_count = self.position_counts.get(
            pacman_pos,
            0
        )
        score -= 300 * visit_count

        return score

    def h_minimax(
        self,
        state,
        depth,
        agent_index,
        alpha,
        beta,
        path
    ):
        if state.isWin() or state.isLose():
            return self.evaluation_function(state)

        if depth >= self.depth:
            return self.evaluation_function(state)

        state_key = self.get_state_key(state)

        # Prevent cycles inside the search tree.
        if state_key in path:
            return self.evaluation_function(state) - 2000

        new_path = path | {state_key}

        num_agents = state.getNumAgents()
        next_agent = (agent_index + 1) % num_agents

        if agent_index == 0:
            successors = state.generatePacmanSuccessors()
        else:
            successors = state.generateGhostSuccessors(
                agent_index
            )

        successors = [
            (successor, action)
            for successor, action in successors
            if action != Directions.STOP
        ]

        if not successors:
            return self.evaluation_function(state)

        next_depth = depth

        if next_agent == 0:
            next_depth += 1

        # Pacman = MAX
        if agent_index == 0:
            value = float("-inf")

            for successor, action in successors:
                child_value = self.h_minimax(
                    successor,
                    next_depth,
                    next_agent,
                    alpha,
                    beta,
                    new_path
                )

                value = max(value, child_value)

                if value >= beta:
                    break

                alpha = max(alpha, value)

            return value

        # Ghost = MIN
        value = float("inf")

        for successor, action in successors:
            child_value = self.h_minimax(
                successor,
                next_depth,
                next_agent,
                alpha,
                beta,
                new_path
            )

            value = min(value, child_value)

            if value <= alpha:
                break

            beta = min(beta, value)

        return value

    def get_action(self, state):
        successors = state.generatePacmanSuccessors()

        successors = [
            (successor, action)
            for successor, action in successors
            if action != Directions.STOP
        ]

        if not successors:
            return Directions.STOP

        current_pos = state.getPacmanPosition()

        self.history.append(current_pos)

        if len(self.history) > 30:
            self.history.pop(0)

        self.position_counts[current_pos] = (
            self.position_counts.get(current_pos, 0) + 1
        )

        path = {self.get_state_key(state)}

        scored_actions = []

        alpha = float("-inf")
        beta = float("inf")

        for successor, action in successors:
            score = self.h_minimax(
                successor,
                0,
                1,
                alpha,
                beta,
                path
            )

            next_pos = successor.getPacmanPosition()

            # Strongly discourage going back to recent positions.
            recent_count = self.history.count(next_pos)

            if recent_count > 0:
                score -= 1000 * recent_count

            # Strong penalty for positions visited many times.
            visits = self.position_counts.get(next_pos, 0)

            score -= 500 * visits

            scored_actions.append(
                (score, action)
            )

            alpha = max(alpha, score)

        scored_actions.sort(
            key=lambda item: item[0],
            reverse=True
        )

        return scored_actions[0][1]