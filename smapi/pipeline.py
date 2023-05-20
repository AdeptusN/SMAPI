from .config import Config
import logging
import os
import requests
import shutil

from PIL import Image

from typing import Dict

import torch
from torchvision import transforms
from basicsr.archs.rrdbnet_arch import RRDBNet

from realesrgan import RealESRGANer

from adeptus_models.unet_autoencoder import UNetAutoencoder
from adeptus_models.segmentation import UNet
import adeptus_modules.transforms as custom_transforms

from .utils.download import download
from .utils.image_processing import prepare_image_for_segmentation, clean_image_by_mask
from .utils.models import load_model



class SmapiPipeline:
    """
    Class for SM model
    """
    def __init__(self, logger=None):
        self.config = Config()
        if not logger:
            self.logger = logging.Logger(__name__)
            self.logger.setLevel(logging.INFO)

            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(name)-12s: %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            
        else:
            self.logger=logger

        self._load_models()

    def _load_models(self):
        WEIGHTS_DIR = self.config.WEIGHTS_DIR
        if WEIGHTS_DIR is None or not os.access(WEIGHTS_DIR, os.F_OK):            
            self.logger.info("Can't access WEIGHTS_DIR. Working in default weight directory")
            WEIGHTS_DIR = os.path.join(os.getcwd(), 'weights')
        
        self.logger.info(f"Working in {WEIGHTS_DIR}")

        try:
            self.logger.info("Loading weights")
            self._load_models_from_files(WEIGHTS_DIR)
        except FileNotFoundError as e:
            self.logger.warning("Didn't found weights files. Downloading from storage")
            if not self._download_weights(WEIGHTS_DIR):
                self.logger.error("Failed to download weights")
                return
            self._load_models_from_files(WEIGHTS_DIR)




    def _load_models_from_files(self, weights_dir):    
        # prepare models
        segmentation_model = UNet(in_channels=3, out_channels=1)
        encoder_decoder_model = UNetAutoencoder(in_channels=6, out_channels=3)
        unsampler = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        
        # load weights
        self.segmentation = load_model(segmentation_model,
                os.path.join(weights_dir, "segmentation.pt"))
        self.logger.info('Segmentation model loaded')

        self.encoder_decoder = load_model(encoder_decoder_model,
                os.path.join(weights_dir, "encoder_decoder.pt"))
        self.logger.info('Encoder-decoder model loaded')

        self.unsampler =  RealESRGANer(
            scale=4,
            model_path=os.path.join(weights_dir, "unsampler.pth"),
            model=unsampler,
        )
        self.logger.info('RealESRGAN model loaded')

    def _download_weights(self , weights_dir: str): 
        models_to_download = ["segmentation.pt", "encoder_decoder.pt",
        "unsampler.pth"]
        bucket = "adeptus-cloud-storage"
        url_template = "https://storage.yandexcloud.net/{bucket}/{key}"
        for key in models_to_download:
            url = url_template.format(bucket=bucket, key=key)
            file_path = os.path.join(weights_dir, key)
            self.logger.info(f'Downloading {key}...')
            download(url, file_path)
        return True

    def process(self, human_image: Image, clothes_image: Image, upscale=True) -> Image:
        """
            Params:
                human_image: Source image of human
                clothes_image: Source image of clothes
                models: Dictitionary of all pretrained models.
                        Keys:
                        - "segmentation"
                        - "encoder_decoder"
            Return:
                Human image with new clothes.
        """
        # TODO: make models preload check

        transform_human = transforms.Compose([
            transforms.Resize((256, 192)),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.25, 0.25, 0.25]
                )
        ])

        transform_clothes = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 192)),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.25, 0.25, 0.25]
                )
        ])

        transform_segmented = transforms.Compose([
            custom_transforms.ThresholdTransform(threshold=0.5),
        ])

        human = prepare_image_for_segmentation(human_image, transform=transform_human)

        segmented = self.segmentation(human)
        segmented = transform_segmented(segmented)

        without_body = clean_image_by_mask(human.squeeze(0), segmented.squeeze(0))

        clothes = transform_clothes(clothes_image)

        encoder_decoder_input = torch.cat((without_body.unsqueeze(0), clothes.unsqueeze(0)), axis=1)

        encoded_image = self.encoder_decoder(encoder_decoder_input)

        if upscale:
            super_resoltion, _ = self.unsampler.enhance(encoded_image.squeeze(0).permute(1, 2, 0).detach().numpy())
            result_image = Image.fromarray(super_resolution)     
        else:
            t = transforms.ToPILImage()
            result_image = t(encoded_image[0])
        return result_image

