import collections
import random

from settings import (
    block_size,
    height,
    max_hunger,
    obstacle_count,
    width,
    world_size,
)
from vision import dxdy_four, look_direction


class SnakeEnv:
    def __init__(self):
        self.direction = None
        self.snake_body = None
        self.food = None
        self.obstacles = None
        self.hunger = None
        self.eaten = None
        self.steps = None

    def reset(self, seed=None):
        if seed is None:
            seed = random.randint(-2**63, 2**63 - 1)
        random.seed(seed)

        self.direction = random.randint(0, 3)
        snake_head_initial = (
            random.randint(2, width // block_size - 3) * block_size,
            random.randint(2, height // block_size - 3) * block_size,
        )
        self.snake_body = collections.deque([snake_head_initial])
        self.snake_body.append(
            (
                snake_head_initial[0] + block_size * -dxdy_four(self.direction)[0],
                snake_head_initial[1] + block_size * -dxdy_four(self.direction)[1],
            )
        )
        self.snake_body.append(
            (
                snake_head_initial[0]
                + block_size * 2 * -dxdy_four(self.direction)[0],
                snake_head_initial[1]
                + block_size * 2 * -dxdy_four(self.direction)[1],
            )
        )

        self.food = self._spawn_food()
        self.obstacles = self._spawn_obstacles()
        self.hunger = max_hunger
        self.eaten = 0
        self.steps = 0

        return seed

    def _spawn_food(self):
        food = (
            random.randint(0, width // block_size - 1) * block_size,
            random.randint(0, height // block_size - 1) * block_size,
        )
        while food in self.snake_body:
            food = (
                random.randint(0, width // block_size - 1) * block_size,
                random.randint(0, height // block_size - 1) * block_size,
            )
        return food

    def _spawn_obstacles(self):
        obstacles = set()
        attempts = 0
        max_attempts = obstacle_count * 20
        while len(obstacles) < obstacle_count and attempts < max_attempts:
            attempts += 1
            position = (
                random.randint(0, width // block_size - 1) * block_size,
                random.randint(0, height // block_size - 1) * block_size,
            )
            if position in self.snake_body or position == self.food:
                continue
            obstacles.add(position)
        return obstacles

    def get_inputs(self):
        inputs = 42 * [0]
        inputs[self.direction] = 1

        body_len = len(self.snake_body)
        tail_x, tail_y = self.snake_body[body_len - 1]
        tail2_x, tail2_y = self.snake_body[body_len - 2]
        if tail_x == tail2_x:
            if tail_y > tail2_y:
                inputs[6] = 1
            else:
                inputs[4] = 1
        else:
            if tail_x > tail2_x:
                inputs[5] = 1
            else:
                inputs[7] = 1

        for i in range(0, 8):
            vision = look_direction(i, self.snake_body, self.food, self.obstacles)
            for j in range(0, 4):
                inputs[8 + i + j * 8] = vision[j]

        food_dx = (self.food[0] - self.snake_body[0][0]) / width
        food_dy = (self.food[1] - self.snake_body[0][1]) / height
        inputs[40] = food_dx
        inputs[41] = food_dy
        return inputs

    def _distance_to_food(self):
        return abs(self.food[0] - self.snake_body[0][0]) + abs(
            self.food[1] - self.snake_body[0][1]
        )

    def step(self, direction):
        self.direction = direction
        reward = 1.0
        prev_distance = self._distance_to_food()

        if direction == 0:
            self.snake_body.appendleft(
                (self.snake_body[0][0], self.snake_body[0][1] + block_size)
            )
        elif direction == 1:
            self.snake_body.appendleft(
                (self.snake_body[0][0] - block_size, self.snake_body[0][1])
            )
        elif direction == 2:
            self.snake_body.appendleft(
                (self.snake_body[0][0], self.snake_body[0][1] - block_size)
            )
        elif direction == 3:
            self.snake_body.appendleft(
                (self.snake_body[0][0] + block_size, self.snake_body[0][1])
            )

        head = self.snake_body[0]
        if head[0] < 0 or head[0] >= width or head[1] < 0 or head[1] >= height:
            return reward, True
        if head in self.obstacles:
            return reward, True
        if self.snake_body.count(head) > 1:
            return reward, True

        if head == self.food:
            self.hunger = max_hunger
            self.eaten += 1
            reward += 100.0
            if len(self.snake_body) == world_size:
                return reward, True
            self.food = self._spawn_food()
        else:
            self.hunger -= 1
            self.snake_body.pop()

        if self.hunger <= 0:
            return reward, True

        new_distance = self._distance_to_food()
        if new_distance < prev_distance:
            reward += 1.0
        elif new_distance > prev_distance:
            reward -= 1.0

        self.steps += 1
        return reward, False
