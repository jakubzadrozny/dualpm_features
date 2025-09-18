from abc import ABC, abstractmethod


class ExtractorWrapper(ABC):
    def __init__(self, device):
        self.device = device

    def extract(self, img):
        raise NotImplementedError
