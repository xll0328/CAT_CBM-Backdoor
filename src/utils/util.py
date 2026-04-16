import pickle
import os
import PIL.Image as Image
import random
import numpy as np
from os.path import join


def read_data(data_path: str, split: str = None):
    if split:
        data_path = join(data_path, split + '.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


def check_dir(directory):
    cwd = os.getcwd()
    path = join(cwd, directory)
    if not os.path.exists(path):
        os.makedirs(path)


def one_hot(label, N_classes):
    one_hot_label = [0] * N_classes
    one_hot_label[label] = 1
    return one_hot_label


def generate_colored_patch(size: tuple):
    patch = Image.new('RGB', size)
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 165, 0),
    ]
    pixels = patch.load()
    for i in range(size[0]):
        for j in range(size[1]):
            pixels[i, j] = random.choice(colors)
    return patch


def generate_signal_trigger(size=(256, 256), frequency=5, amplitude=50):
    width, height = size
    signal_trigger = Image.new('RGB', size)
    pixels = signal_trigger.load()
    for x in range(width):
        for y in range(height):
            value = int((np.sin(2 * np.pi * frequency * x / width) + 1) / 2 * amplitude + (255 - amplitude))
            pixels[x, y] = (value, value, value)
    return signal_trigger


def add_patched_trigger(image_path: str = None, patch: Image = None, position: tuple = (0, 0), save_path: str = None):
    image = Image.open(image_path)
    image = image.convert('RGB').resize((256, 256))
    if patch is None:
        patch = Image.new('RGB', size=((20, 20)), color=(0, 0, 0))
    image.paste(patch)
    if save_path:
        image.save(save_path)
    return image


def add_blended_trigger(image_path: str, trigger_path: str, position=(0, 0), save_path: str = None, alpha=0.5):
    image = Image.open(image_path).convert('RGBA').resize((256, 256))
    trigger = Image.open(trigger_path).convert('RGBA').resize((256, 256))
    blended_trigger = Image.new('RGBA', trigger.size)
    for x in range(trigger.size[0]):
        for y in range(trigger.size[1]):
            r, g, b, a = trigger.getpixel((x, y))
            blended_trigger.putpixel((x, y), (r, g, b, int(a * alpha)))
    image.paste(blended_trigger, position, blended_trigger)
    if save_path:
        image.save(save_path)
    return image


def add_signal_trigger(image_path, trigger_image, position: tuple = (0, 0), save_path: str = None, alpha: float = 0.5):
    image = Image.open(image_path).convert('RGBA')
    trigger_image = trigger_image.convert('RGBA')
    blended_trigger = Image.new('RGBA', trigger_image.size)
    for x in range(trigger_image.size[0]):
        for y in range(trigger_image.size[1]):
            r, g, b, a = trigger_image.getpixel((x, y))
            blended_trigger.putpixel((x, y), (r, g, b, int(a * alpha)))
    image.paste(blended_trigger, position, blended_trigger)
    if save_path:
        image.save(save_path)
    return image
