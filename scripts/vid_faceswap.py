from pathlib import Path
import gradio as gr
import cv2
from PIL import Image, ImageDraw
import numpy as np
import random
from datetime import datetime
import pkg_resources
import torch
import torchgeometry as tg
from tqdm import tqdm

import modules
from modules import script_callbacks
from modules.ui import create_sampler_and_steps_selection, create_seed_inputs, create_refresh_button
from modules import shared, sd_samplers
from modules.processing import StableDiffusionProcessingImg2Img, get_fixed_seed
from modules.devices import torch_gc

from scripts.video import video_reader, video_writer
from scripts.frame import Frame
from scripts.face import FaceDetector_get, FaceDetector_release
from scripts.batch import process_batch, batch

def rifed(frames, fps, target_fps):
    from rife_ncnn_vulkan_python import Rife
    rife = Rife(gpuid=0, model='rife-v4.6')
    idxes = np.linspace(0, len(frames) - 1, int(np.round(target_fps / fps * len(frames))))
    np.array(fps / target_fps * len(frames)) / ((target_fps - 1) / fps)
    new_frames = []
    for idx in tqdm(idxes, desc='Interpolating frames'):
        lb, ub = int(np.floor(idx)), int(np.ceil(idx))
        if lb == ub:
            frame = frames[int(idx)]
        else:
            frame = rife.process(Image.fromarray(frames[lb]), Image.fromarray(frames[ub]), timestep=idx-lb)
        new_frames.append(np.array(frame))
    return new_frames

def process_video(
        video_input,
        max_fps,
        target_fps,
        batch_size,
        size,
        alpha,
        prompt_styles,
        txt_pos_prompt,
        txt_neg_prompt,
        steps,
        sampler_index,
        cfg_scale,
        denoising_strength,
        seed,
        subseed,
        subseed_strength,
        seed_resize_from_h,
        seed_resize_from_w,
        seed_checkbox,
        *args
):
    torch_gc()

    frames_data, fps, audio = video_reader(video_input, max_fps=max_fps if max_fps > 0 else None)
    dets = FaceDetector_get(frames_data)
    frames = [Frame(i, frame_data, det) for i, (frame_data, det) in enumerate(zip(frames_data, dets))]

    faces = [face for frame in frames for face in frame.faces]
    FaceDetector_release()
    print(f'{len(faces)} faces found.')
    p = StableDiffusionProcessingImg2Img(
        prompt=txt_pos_prompt,
        negative_prompt=txt_neg_prompt,
        styles=prompt_styles,
        width=size,
        height=size,
        init_images=None,
        batch_size=batch_size,
        cfg_scale=cfg_scale,
        denoising_strength=denoising_strength,
        inpaint_full_res=1,
        resize_mode=0,
        image_cfg_scale=1.5,
        inpainting_mask_invert=0,
        inpainting_fill=1,
        inpaint_full_res_padding=32,
        mask_blur=4,
        do_not_save_samples=True,
        do_not_save_grid=True,
        seed=get_fixed_seed(seed),
        steps=steps,
        sampler_name=sd_samplers.samplers_for_img2img[sampler_index].name,
        seed_enable_extras=seed_checkbox,
        subseed=get_fixed_seed(subseed),
        subseed_strength=subseed_strength,
        seed_resize_from_h=seed_resize_from_h,
        seed_resize_from_w=seed_resize_from_w,
    )
    p.scripts = modules.scripts.scripts_txt2img
    p.script_args = args
    
    alpha = 1 / (1+alpha)
    padding = int((size-alpha*size)*0.5)

    mask = np.zeros((size, size))
    c = (size) // 2
    mask = cv2.circle(mask, (c, c), padding, 255, -1).astype(np.uint8)
    p.image_mask = Image.fromarray(mask, 'L')
    mask = np.array(mask, dtype=np.float32)
    mask = cv2.blur(mask, (padding, padding))
    mask = mask[...,None]

    crops = [face.crop(size, alpha, padding) for face in faces]
    swapped_faces = process_batch(p, crops, *args)
    p.close()
    
    for face, swapped_face in zip(faces, swapped_faces):
        face.swapped = swapped_face

    def batchwarp(minibatch):
        target_crops = [np.array(source.swapped) for source in minibatch]
        target_crops_t = torch.from_numpy(np.array(target_crops).transpose(0, 3, 1, 2)).float().cuda()
        M_t = torch.from_numpy(np.array([face.iM(alpha,padding) for face in minibatch])).float().cuda()
        warped_crops = tg.warp_affine(target_crops_t, M_t, frames[0].shape[:2]).cpu().detach().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
        masks = torch.from_numpy(np.array([mask] * len(target_crops)).transpose(0, 3, 1, 2)).float().cuda()
        warped_masks = tg.warp_affine(masks, M_t, frames[0].shape[:2]).cpu().detach().numpy().transpose(0, 2, 3, 1).astype(np.float32)

        for face, warped, wraped_mask in zip(minibatch, warped_crops, warped_masks):
            face.warped = warped
            face.mask = wraped_mask
    batch(faces, batchwarp, max_batch_size=50, desc='Warping Swaps')

    swapped_frames = []
    for frame in tqdm(frames):
        if len([face for face in frame.faces if face in faces])==0:
            swapped_frames.append(frame.data)
        
        else:
            mask = np.add.reduce([
                face.mask
                for face in frame.faces
                if face in faces
            ])
            swapped = np.add.reduce([
                face.warped * face.mask
                for face in frame.faces
                if face in faces
            ], axis=0) / np.maximum(1, mask)
            image_mask = np.minimum(1, mask)
            swapped_frame = (swapped + (1 - image_mask) * frame.data).astype(np.uint8)
            swapped_frames.append(swapped_frame)

    if target_fps > fps:
        swapped_frames = rifed(swapped_frames, fps, target_fps)
        fps = target_fps

    video_file = f'{shared.opts.outdir_samples}/{datetime.now():%Y%m%d%H%M%S%f} {Path(video_input).stem}.mp4'
    video_writer(video_file, swapped_frames, fps, audio)
    return video_file

def add_tab():
    with gr.Blocks() as ui:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='compact'):
                btn_faceswap = gr.Button(value='Faceswap')
                
                with gr.Row():
                    prompt_styles = gr.Dropdown(label="Styles", elem_id='vid_fs_styles', choices=[k for k, v in shared.prompt_styles.styles.items()], value=[], multiselect=True)
                    create_refresh_button(prompt_styles, shared.prompt_styles.reload, lambda: {"choices": [k for k, v in shared.prompt_styles.styles.items()]}, f"refresh_vid_fs_styles")

                with gr.Row():
                    with gr.Column():
                        txt_pos_prompt = gr.TextArea(label='Positive prompt')
                        txt_neg_prompt = gr.TextArea(label='Negative prompt')
                    with gr.Column():
                        video_input = gr.Video()
                        video_input.style(height=380)
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            rife_enabled = 'rife-ncnn-vulkan-python' in {pkg.key for pkg in pkg_resources.working_set}
                            max_fps = gr.Slider(label='Maximum FPS (0 = no limit)', minimum=0, maximum=60, step=1, value=15)
                            target_fps = gr.Slider(label='Target FPS (0 = no interpolation)', minimum=0, maximum=60, step=1, value=60 if rife_enabled else 0, interactive=rife_enabled)
                    batch_size = gr.Slider(label='Batch Size', minimum=1, maximum=64, step=1, value=8)
                with gr.Row():
                    size = gr.Slider(label='Size', minimum=512, maximum=1024, step=8, value=512)
                    alpha = gr.Slider(label='% Region Increase', minimum=0, maximum=3, step=0.1, value=1)

                steps, sampler_index = create_sampler_and_steps_selection(modules.sd_samplers.samplers, 'vid-faceswap')
                cfg_scale = gr.Slider(label='CFG Scale', minimum=1, maximum=30, step=0.5, value=7)
                denoising_strength = gr.Slider(label='Denoising Strength', minimum=0, maximum=1, step=0.01, value=0.2)
                seed, _, subseed, _, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox = create_seed_inputs('txt2img')

                with gr.Group(elem_id="vid_faceswap_script_container"):
                    custom_inputs = modules.scripts.scripts_img2img.setup_ui()
            
            video_output = gr.Video(interactive=False)
        
        btn_faceswap.click(fn=process_video, inputs=[
            video_input,
            max_fps,
            target_fps,
            batch_size,
            size,
            alpha,
            prompt_styles,
            txt_pos_prompt,
            txt_neg_prompt,
            steps,
            sampler_index,
            cfg_scale,
            denoising_strength,
            seed,
            subseed,
            subseed_strength,
            seed_resize_from_h,
            seed_resize_from_w,
            seed_checkbox,
            ] + custom_inputs, outputs=[
                video_output
            ])

    return [(ui, "Vid Faceswap", 'vid-faceswap')]

script_callbacks.on_ui_tabs(add_tab)
