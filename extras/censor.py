import os
import numpy as np
import torch
from transformers import CLIPConfig, CLIPImageProcessor

class Censor:
    def __init__(self):
        self.clip_image_processor: CLIPImageProcessor | None = None
        self.load_device = torch.device('cpu')

    def init(self):
        if self.clip_image_processor is None:
            self.clip_image_processor = CLIPImageProcessor()

    def censor(self, images: list | np.ndarray) -> list | np.ndarray:
        self.init()

        single = False
        if not isinstance(images, (list, np.ndarray)):
            images = [images]
            single = True

        # Apenas retorna as imagens originais, sem aplicar censura
        checked_images = [np.array(image, dtype=np.uint8) for image in images]

        if single:
            checked_images = checked_images[0]

        return checked_images

default_censor = Censor().censor
