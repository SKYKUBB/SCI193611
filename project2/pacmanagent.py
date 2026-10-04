import numpy as np
from pacman_module.game import Agent
from pacman_module.pacman import Directions
from pacman_module.util import Queue, manhattanDistance


class PacmanAgent(Agent):
    """
    An agent that uses belief state from Bayes Filter to locate and capture ghosts.
    """

    def __init__(self, args):
        super().__init__(args)
        self.args = args

    def get_action(self, state, belief_state=None):
        """
        Given a pacman game state and a belief state from Bayes Filter,
        returns a legal move toward the most likely ghost position.
        """
        legal = state.getLegalActions(0)

        # หลีกเลี่ยงการเลือก STOP ถ้ายังมีทางอื่นให้เดิน
        if Directions.STOP in legal and len(legal) > 1:
            legal.remove(Directions.STOP)

        if not belief_state:
            return np.random.choice(legal) if legal else Directions.STOP

        pacman_pos = state.getPacmanPosition()

        # 1. ค้นหาตำแหน่งที่มีความน่าจะเป็นสูงสุด (MAP Position) ของผีที่ยังไม่ถูกกิน
        targets = []
        for z, b_state in enumerate(belief_state):
            is_eaten = False
            if hasattr(state, "data") and hasattr(state.data, "_eaten"):
                if z + 1 < len(state.data._eaten):
                    is_eaten = state.data._eaten[z + 1]

            if not is_eaten and np.max(b_state) > 0:
                max_idx = np.unravel_index(np.argmax(b_state), b_state.shape)
                targets.append((int(max_idx[0]), int(max_idx[1])))

        if not targets:
            return np.random.choice(legal) if legal else Directions.STOP

        # 2. เลือกตำแหน่งเป้าหมายที่อยู่ใกล้ Pacman มากที่สุดตาม Manhattan Distance
        best_target = min(targets, key=lambda t: manhattanDistance(pacman_pos, t))

        # 3. คำนวณ Action ก้าวแรกด้วย BFS
        return self._bfs_next_action(state, pacman_pos, best_target, legal)

    def _bfs_next_action(self, state, start_pos, target_pos, legal_actions):
        """
        หา Action ก้าวแรกจาก start_pos ไปยัง target_pos โดยใช้ Breadth-First Search (BFS)
        """
        if start_pos == target_pos:
            return legal_actions[0] if legal_actions else Directions.STOP

        walls = state.getWalls()
        queue = Queue()
        visited = set([start_pos])

        action_vectors = {
            Directions.NORTH: (0, 1),
            Directions.SOUTH: (0, -1),
            Directions.EAST: (1, 0),
            Directions.WEST: (-1, 0),
        }

        # กำหนดจุดเริ่มต้นก้าวแรกจาก legal_actions
        for action in legal_actions:
            if action in action_vectors:
                dx, dy = action_vectors[action]
                nxt = (start_pos[0] + dx, start_pos[1] + dy)
                if 0 <= nxt[0] < walls.width and 0 <= nxt[1] < walls.height:
                    if not walls[nxt[0]][nxt[1]]:
                        if nxt == target_pos:
                            return action
                        visited.add(nxt)
                        queue.push((nxt, action))

        # ค้นหาเส้นทาง BFS
        while not queue.isEmpty():
            curr_pos, first_action = queue.pop()
            if curr_pos == target_pos:
                return first_action

            for action, (dx, dy) in action_vectors.items():
                nxt = (curr_pos[0] + dx, curr_pos[1] + dy)
                if 0 <= nxt[0] < walls.width and 0 <= nxt[1] < walls.height:
                    if not walls[nxt[0]][nxt[1]] and nxt not in visited:
                        visited.add(nxt)
                        queue.push((nxt, first_action))

        return legal_actions[0] if legal_actions else Directions.STOP