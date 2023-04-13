import math
import os
import sys
import traceback
import cv2
from tqdm import tqdm

import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageChops
import torch
import torch.nn.functional as F

from modules import devices, sd_samplers
from modules.generation_parameters_copypaste import create_override_settings_dict
from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images
from modules.shared import opts, state
import modules.shared as shared
import modules.processing as processing
from modules.ui import plaintext_to_html
import modules.images as images
import modules.scripts

def batch(tasklist, func, max_batch_size, desc=None):
    tasklist = np.array(tasklist)
    with torch.no_grad():
        with tqdm(total=len(tasklist), desc=desc) as pbar:
            rest = tasklist

            results = []
            while len(rest) > 0:
                minibatch, rest = np.split(rest, [max_batch_size])

                result = func(minibatch)
                if result is not None:
                    results += [*result]

                pbar.update(len(minibatch))

    return results

def process_batch(p:StableDiffusionProcessingImg2Img, images, *args):
    processing.fix_seed(p)

    state.job_count = np.ceil(len(images) / p.batch_size)
    total_size = p.height + 2*p.inpaint_full_res_padding
    mask_size = p.height
    mask_image = np.ones((total_size, total_size))
    mask_image = cv2.circle(mask_image, (int(total_size/2), int(total_size/2)), int(mask_size/2), 0, -1)
    # p.image_mask = Image.fromarray(mask_image, 'L')
    p.image_mask = None

    tasklist = np.array(images)
    rest = tasklist
    results = []

    while len(rest) > 0:
        if state.interrupted:
            break
        
        minibatch, rest = np.split(rest, [p.batch_size])
        
        p.init_images = [Image.fromarray(face) for face in minibatch]
        proc = modules.scripts.scripts_img2img.run(p, *args)
        if proc is None:
            proc = process_images(p)
        # proc = process_images(p)
        results += proc.images
        state.job_no += 1
    
    return results

class SoftErosion(torch.nn.Module):
    def __init__(self, kernel_size=15, threshold=0.6, iterations=1):
        super(SoftErosion, self).__init__()
        r = kernel_size // 2
        self.padding = r
        self.iterations = iterations
        self.threshold = threshold

        # Create kernel
        y_indices, x_indices = torch.meshgrid(torch.arange(0., kernel_size), torch.arange(0., kernel_size))
        dist = torch.sqrt((x_indices - r) ** 2 + (y_indices - r) ** 2)
        kernel = dist.max() - dist
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, *kernel.shape)
        self.register_buffer('weight', kernel)

    def forward(self, x):
        x = x.float()
        for i in range(self.iterations - 1):
            x = torch.min(x, F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding))
        x = F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding)

        mask = x >= self.threshold
        x[mask] = 1.0
        x[~mask] /= x[~mask].max()

        return x