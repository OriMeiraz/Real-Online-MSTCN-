from os.path import exists
from PIL import Image


class FrameGenerator:
    def __init__(self, path: str, size_of_index=5, start='img', end='jpg') -> None:
        self.t = 1
        self.path = path
        self.size_of_index = size_of_index
        self.end = end
        self.start = start
        self.batch_size = 1

    def next(self):
        temp_path = f'{self.path}/{self.start}_{str(self.t).zfill(self.size_of_index)}.{self.end}'
        if exists(temp_path):
            self.t += 1
            return Image.open(temp_path)
