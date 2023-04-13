from tqdm import tqdm
import cv2
from pathlib import Path
from webp import WebPData, WebPAnimDecoder, WebPAnimDecoderOptions, WebPColorMode, mimread
from moviepy.editor import VideoFileClip, AudioFileClip, ImageSequenceClip

def video_reader(file, max_fps=None):
    if Path(file).suffix.lower() == '.webp':
        return _webp_reader(file, max_fps)
    else:
        return _ffmpeg_reader(file, max_fps)
    
def _webp_reader(file, max_fps):
    with open(file, 'rb') as f:
        webp_data = WebPData.from_buffer(f.read())
        dec_opts = WebPAnimDecoderOptions.new(use_threads=True, color_mode=WebPColorMode.RGB)
        dec = WebPAnimDecoder.new(webp_data, dec_opts)
        eps = 1e-7
        
        frames_data = list(dec.frames())

    frames = [arr for arr, _ in frames_data]
    fps = 1000 * len(frames_data) / frames_data[-1][1]
    if max_fps is not None and fps > max_fps:
        frames = list(mimread(file, fps=max_fps))
        fps = max_fps

    return frames, fps, None

def _ffmpeg_reader(file, max_fps):
    with VideoFileClip(file) as clip:
        target_fps = min(max_fps, clip.fps) if max_fps is not None else clip.fps
        frames = list(clip.iter_frames(target_fps))
        audio = AudioFileClip(file) if clip.audio is not None else None
        
    return frames, target_fps, audio

def video_writer(file, frames, fps, audio):
    clip = ImageSequenceClip(frames, fps=fps)
    clip = clip.set_audio(audio)
    clip.write_videofile(file, audio_codec='aac', fps=clip.fps)