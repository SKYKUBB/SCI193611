"""
Smart BlindDog Agent
---------------------
Agent เดินสำรวจ grid 2D โดย:
- จำตำแหน่งที่เคยไปแล้ว (visited)
- ห้ามเหยียบตำแหน่งซ้ำ
- ใช้ backtracking เพื่อให้สำรวจครบทุกช่อง (แทนที่จะติดตันกลางทาง)
"""

import random


class SmartBlindDog:
    def __init__(self, rows, cols, start=(0, 0), seed=None):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.current = start
        self.visited = {start}
        self.path = [start]         
        self.stack = [start]        
        self.total_cells = rows * cols
        if seed is not None:
            random.seed(seed)

    def get_valid_moves(self, position):
        """คืนตำแหน่งข้างเคียงที่ยังไม่เคยไ"""
        r, c = position
        candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]  # บน,ล่าง,ซ้าย,ขวา
        valid = [
            pos for pos in candidates
            if 0 <= pos[0] < self.rows and 0 <= pos[1] < self.cols
            and pos not in self.visited
        ]
        return valid

    def step(self):
      
        moves = self.get_valid_moves(self.current)

        if moves:
            # มีทางไปที่ยังไม่เคไป สุ่มเลือกไป
            next_pos = random.choice(moves)
            self.visited.add(next_pos)
            self.current = next_pos
            self.stack.append(next_pos)
            self.path.append(next_pos)
            return True
        else:
            # ตัน ถอยกลับตำแหน่งก่อนหน้า
            self.stack.pop()
            if not self.stack:
                return False  # ไม่มีที่ให้ถอยแล้ว จบ
            self.current = self.stack[-1]
            self.path.append(self.current)  # บันทึกว่าถอยกลับ
            return True

    def run(self, max_steps=10000):
        
        steps = 0
        while len(self.visited) < self.total_cells and steps < max_steps:
            alive = self.step()
            steps += 1
            if not alive:
                break
        return self.visited, self.path

    def print_result(self):
        print(f"Grid ขนาด: {self.rows}x{self.cols} (ทั้งหมด {self.total_cells} ช่อง)")
        print(f"สำรวจได้: {len(self.visited)} ช่อง")
        print(f"สำรวจครบหรือไม่: {'ครบ' if len(self.visited) == self.total_cells else 'ไม่ครบ'}")
        print(f"จำนวนก้าวเดินทั้งหมด (รวม backtrack): {len(self.path) - 1}")
        print()
        self.print_grid()

    def print_grid(self):
       
        order = {}
        idx = 1
        for pos in self.path:
            if pos not in order:
                order[pos] = idx
                idx += 1

        for r in range(self.rows):
            row_str = ""
            for c in range(self.cols):
                if (r, c) == self.start:
                    cell = "S".rjust(3)
                elif (r, c) in order:
                    cell = str(order[(r, c)]).rjust(3)
                else:
                    cell = "-".rjust(3)
                row_str += cell
            print(row_str)


if __name__ == "__main__":
    dog = SmartBlindDog(rows=5, cols=5, start=(0, 0), seed=42)
    dog.run()
    dog.print_result()