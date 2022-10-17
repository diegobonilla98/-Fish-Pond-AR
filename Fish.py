import time
from scipy.spatial.distance import euclidean
import cv2
import numpy as np
import glob


def paste_image(canvas, image, pos, angle):
    w, h = image.shape[1], image.shape[0]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img_rot = cv2.warpAffine(image[:, :, :3], M, (w, h))
    mask_rot = cv2.warpAffine(cv2.cvtColor(image[:, :, 3], cv2.COLOR_GRAY2BGR), M, (w, h), borderMode=cv2.BORDER_CONSTANT)
    cX, cY = int(pos[0]), int(pos[1])
    piece_of_background = canvas[cY - h // 2: cY + h // 2, cX - w // 2: cX + w // 2, :].copy()
    img_combined = np.uint8(np.where(mask_rot == 255, img_rot, piece_of_background))
    canvas[cY - h // 2: cY + h // 2, cX - w // 2: cX + w // 2, :] = img_combined
    return canvas


def remap(x, in_min, in_max, out_min, out_max):
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def angle_between(p1, p2):
    p3 = np.array([p1[0], p1[1] - 1])
    v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    v2 = np.array([p3[0] - p1[0], p3[1] - p1[1]])
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.rad2deg(np.arccos(dot_product))


class Fish:
    def __init__(self, canvas_size, personality=None):
        if personality is None:
            personality = np.random.randint(0, 3)
        self.personality = personality

        self.WIDTH, self.HEIGHT = canvas_size
        self.pos = np.array((np.random.randint(0, self.WIDTH), np.random.randint(0, self.HEIGHT)), np.float32)

        self.size = 50 * (personality + 1)
        self.speed_mult = 5 * (personality + 1)
        self.speed = np.array([0., 0.])
        self.color = [(255, 176, 58), (98, 181, 255), (116, 116, 248)][personality]

        sprites_path = glob.glob('./fish_sprite/*.png')
        self.sprites = np.array([cv2.resize(cv2.imread(p, cv2.IMREAD_UNCHANGED), (self.size, self.size)) for p in sprites_path])
        self.sprites[:, :, :, 0] = self.color[0]
        self.sprites[:, :, :, 1] = self.color[1]
        self.sprites[:, :, :, 2] = self.color[2]

        self.sine_offset = 0
        self.angle_movement = np.random.randint(0, 360)

        self.sine_offset_anim = 0
        self.sprite_idx = 0

        self.timer_to_go = np.random.rand() * 5.
        self.last_time_to_go = 0
        self.natural_attraction_point = np.array((np.random.randint(0, self.WIDTH), np.random.randint(0, self.HEIGHT)))

    def display(self, canvas):
        return paste_image(canvas, self.sprites[self.sprite_idx], (int(self.pos[0]), int(self.pos[1])), self.angle_movement)

    def update(self, center_of_attraction):
        if center_of_attraction is None:
            center_of_attraction = self.natural_attraction_point
        else:
            self.natural_attraction_point += np.array((np.random.randint(-self.WIDTH // 10, self.WIDTH // 10), np.random.randint(-self.HEIGHT // 10, self.HEIGHT // 10)))
            self.natural_attraction_point[0] = np.clip(self.natural_attraction_point[0], 0, self.WIDTH)
            self.natural_attraction_point[1] = np.clip(self.natural_attraction_point[1], 0, self.HEIGHT)

        new_direction = center_of_attraction - self.pos
        new_direction_norm = np.array((new_direction[0] / self.WIDTH * self.speed_mult, new_direction[1] / self.HEIGHT * self.speed_mult))
        self.speed = new_direction_norm
        self.pos += self.speed

        self.angle_movement = angle_between(self.pos, center_of_attraction)

        self.sine_offset_anim += np.linalg.norm(self.speed) * (0.1 - 0.02 * self.personality)
        self.sprite_idx = int(remap(np.sin(self.sine_offset_anim), -1, 1, 0, len(self.sprites)))
        self.sine_offset += np.linalg.norm(self.speed) * 0.03
        self.pos += np.sin(self.sine_offset) * np.linalg.norm(self.speed) * 0.1

        if abs(time.time() - self.last_time_to_go) > self.timer_to_go:
            self.last_time_to_go = time.time()
            self.timer_to_go = np.random.rand() * 5.
            self.natural_attraction_point += np.array((np.random.randint(-self.WIDTH // 10, self.WIDTH // 10), np.random.randint(-self.HEIGHT // 10, self.HEIGHT // 10)))
            self.natural_attraction_point[0] = np.clip(self.natural_attraction_point[0], 0, self.WIDTH)
            self.natural_attraction_point[1] = np.clip(self.natural_attraction_point[1], 0, self.HEIGHT)

        if self.pos[0] - self.size // 2 < 0:
            self.pos[0] = self.size // 2
        if self.pos[0] + self.size // 2 > self.WIDTH:
            self.pos[0] = self.WIDTH - self.size // 2
        if self.pos[1] - self.size // 2 < 0:
            self.pos[1] = self.size // 2
        if self.pos[1] + self.size // 2 > self.HEIGHT:
            self.pos[1] = self.HEIGHT - self.size // 2
