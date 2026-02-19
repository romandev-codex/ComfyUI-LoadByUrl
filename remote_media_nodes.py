import requests
from io import BytesIO
from PIL import Image, ImageOps
import torch
import numpy as np
import av


# 🖼️ LoadImageByUrl
class LoadImageByUrl:
    """
    Loads an image from a remote URL and returns it as a Torch tensor.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "https://example.com/image.png", "multiline": False}),
                "max_width": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "max_height": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "WIDTH", "HEIGHT",)
    FUNCTION = "load_image"
    CATEGORY = "Remhes/Remote"
    OUTPUT_NODE = True

    def load_image(self, url, max_width=0, max_height=0):
        if not url or not url.strip():
            return (None, None, None)
        
        response = requests.get(url, stream=True)
        response.raise_for_status()

        image = Image.open(BytesIO(response.content))
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        
        # Apply max_width and max_height constraints
        current_width, current_height = image.size
        new_width = current_width
        new_height = current_height
        
        # Calculate scaling for width constraint
        if max_width > 0 and current_width > max_width:
            width_scale = max_width / current_width
            new_width = max_width
            new_height = int(current_height * width_scale)
        
        # Calculate scaling for height constraint
        if max_height > 0 and new_height > max_height:
            height_scale = max_height / new_height
            new_height = max_height
            new_width = int(new_width * height_scale)
        
        # Resize if needed
        if new_width != current_width or new_height != current_height:
            image = image.resize((new_width, new_height), Image.LANCZOS)
        
        arr = np.array(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W, 3)
        height, width = tensor.shape[1], tensor.shape[2]

        return (tensor, width, height)

    @classmethod
    def IS_CHANGED(cls, url, max_width, max_height):
        return url


# 🖼️ LoadImagesByUrl
class LoadImagesByUrl:
    """
    Loads multiple images from remote URLs and returns them as separate Torch tensors.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "https://example.com/media.png", "multiline": False}),
                "max_width": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "max_height": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "url2": ("STRING", {"default": "", "multiline": False}),
                "url3": ("STRING", {"default": "", "multiline": False}),
                "url4": ("STRING", {"default": "", "multiline": False}),
                "url5": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "INT", "INT",)
    RETURN_NAMES = ("image1", "image2", "image3", "image4", "image5", "WIDTH", "HEIGHT",)
    FUNCTION = "load_images"
    CATEGORY = "Remhes/Remote"
    OUTPUT_NODE = True

    def load_images(self, url, max_width, max_height, url2, url3, url4, url5):
        image_loader = LoadImageByUrl()
        images = []
        first_width = None
        first_height = None
        
        # Load all URLs
        for idx, img_url in enumerate([url, url2, url3, url4, url5]):
            if img_url and img_url.strip():
                try:
                    img_tensor, width, height = image_loader.load_image(img_url, max_width, max_height)
                    images.append(img_tensor)
                    # Store dimensions only for first image
                    if idx == 0:
                        first_width = width
                        first_height = height
                except Exception as e:
                    # If loading fails, return None
                    images.append(None)
            else:
                # Empty URL - return None
                images.append(None)
        
        return tuple(images + [first_width, first_height])

    @classmethod
    def IS_CHANGED(cls, url, max_width, max_height, url2, url3, url4, url5):
        return f"{url}|{url2}|{url3}|{url4}|{url5}"


# 🎥 LoadVideoByUrl
class LoadVideoByUrl:
    """
    Loads a video from a remote URL and returns decoded frames and FPS.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "https://example.com/video.mp4", "multiline": False}),
                "max_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1}),
                "fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 120.0, "step": 0.1}),
                "skip_first_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1}),
                "max_width": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "max_height": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "IMAGE", "IMAGE", "INT", "INT", "INT", "AUDIO",)
    RETURN_NAMES = ("IMAGES", "FPS", "FIRST_FRAME", "LAST_FRAME", "WIDTH", "HEIGHT", "FRAMES", "AUDIO",)
    FUNCTION = "load_video"
    CATEGORY = "Remhes/Remote"
    OUTPUT_NODE = True

    def _resize_frame(self, img, max_width, max_height):
        """Helper method to resize a frame based on max_width and max_height"""
        from PIL import Image
        pil_img = Image.fromarray(img)
        
        current_width = pil_img.width
        current_height = pil_img.height
        
        new_width = current_width
        new_height = current_height
        
        # Calculate scaling for width constraint
        if max_width > 0 and current_width > max_width:
            width_scale = max_width / current_width
            new_width = max_width
            new_height = int(current_height * width_scale)
        
        # Calculate scaling for height constraint
        if max_height > 0 and new_height > max_height:
            height_scale = max_height / new_height
            new_height = max_height
            new_width = int(new_width * height_scale)
        
        # No resizing needed
        if new_width == current_width and new_height == current_height:
            return img
        
        pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
        resized = np.array(pil_img)
        del pil_img
        return resized

    def load_video(self, url, max_seconds, select_every_nth, fps, skip_first_seconds, max_width, max_height):
        if not url or not url.strip():
            return (None, None, None, None, None, None, 0, None)
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        buffer = BytesIO(response.content)

        container = av.open(buffer)
        video_stream = next(s for s in container.streams if s.type == "video")
        video_fps = float(video_stream.average_rate or 25)
        frames = []
        last_frame_data = None
        
        # Find audio stream
        audio_stream = None
        audio_frames = []
        sample_rate = 0
        for stream in container.streams:
            if stream.type == "audio":
                audio_stream = stream
                sample_rate = stream.sample_rate
                break

        # Calculate frame interval based on fps
        # If fps is 0 or >= video fps, include all frames
        if fps <= 0 or fps >= video_fps:
            frame_interval = 1.0
        else:
            frame_interval = video_fps / fps

        skip_first_frames = int(video_fps * max(skip_first_seconds, 0.0))
        max_allowed_frames = int(video_fps * max(max_seconds, 0.0)) if max_seconds > 0 else 0

        frame_count = 0
        next_frame_index = 0.0
        
        # Decode audio if available
        if audio_stream:
            container.seek(0)
            for frame in container.decode(audio_stream):
                audio_frames.append(frame.to_ndarray())
            container.seek(0)
        
        for i, frame in enumerate(container.decode(video_stream)):
            img = frame.to_ndarray(format="rgb24")

            if i < skip_first_frames:
                continue

            # Stop decoding past the requested time window (relative to skip_first_seconds)
            if max_allowed_frames > 0 and (i - skip_first_frames) >= max_allowed_frames:
                break
            
            # Store as last_frame_data (will be added at end if not already included)
            last_frame_data = (img, i)
            
            # Keep one frame every N frames (1 = keep all frames)
            if select_every_nth > 1 and (i % select_every_nth) != 0:
                continue

            # Determine if this frame should be included based on fps logic
            should_include = i >= int(next_frame_index)
            
            if not should_include:
                continue
                
            frame_count += 1
            next_frame_index += frame_interval

            img = self._resize_frame(img, max_width, max_height)
            tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
            frames.append(tensor)

            del img
            del tensor

        container.close()
        
        # Add last frame if it wasn't already included and we have frames
        if last_frame_data and len(frames) > 0:
            last_img, last_idx = last_frame_data
            # Check if the last frame was already processed
            if last_idx >= int(next_frame_index - frame_interval):
                # Last frame was already included, skip
                pass
            else:
                # Add the last frame
                img = self._resize_frame(last_img, max_width, max_height)
                tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
                frames.append(tensor)
                del img
                del tensor

        if not frames:
            raise ValueError("No frames decoded from video.")

        video_tensor = torch.cat(frames, dim=0)  # (frames, H, W, 3)
        first_frame = frames[0]
        last_frame = frames[-1]
        height, width = video_tensor.shape[1], video_tensor.shape[2]
        
        # Process audio to ComfyUI AUDIO format: {"waveform": (B, C, S), "sample_rate": int}
        audio_output = None
        if audio_frames:
            normalized_frames = []
            for frame_np in audio_frames:
                if frame_np.ndim == 1:
                    frame_np = frame_np[np.newaxis, :]
                elif frame_np.ndim == 2:
                    if frame_np.shape[1] <= 8 and frame_np.shape[0] > frame_np.shape[1]:
                        frame_np = frame_np.T
                else:
                    continue

                if np.issubdtype(frame_np.dtype, np.integer):
                    max_val = float(np.iinfo(frame_np.dtype).max)
                    if max_val > 0:
                        frame_np = frame_np.astype(np.float32) / max_val
                    else:
                        frame_np = frame_np.astype(np.float32)
                else:
                    frame_np = frame_np.astype(np.float32)

                normalized_frames.append(frame_np)

            if normalized_frames:
                audio_np = np.concatenate(normalized_frames, axis=1)
                skip_audio_samples = int((sample_rate or 44100) * max(skip_first_seconds, 0.0))
                if skip_audio_samples > 0:
                    if skip_audio_samples >= audio_np.shape[1]:
                        audio_np = None
                    else:
                        audio_np = audio_np[:, skip_audio_samples:]

                if audio_np is not None and max_seconds > 0:
                    max_audio_samples = int((sample_rate or 44100) * max(max_seconds, 0.0))
                    if max_audio_samples <= 0:
                        audio_np = None
                    elif audio_np.shape[1] > max_audio_samples:
                        audio_np = audio_np[:, :max_audio_samples]

                if audio_np is None or audio_np.shape[1] == 0:
                    audio_output = None
                else:
                    audio_waveform = torch.from_numpy(audio_np).unsqueeze(0)
                    audio_output = {
                        "waveform": audio_waveform,
                        "sample_rate": int(sample_rate or 44100),
                    }
        
        frames_count = int(video_tensor.shape[0])
        return video_tensor, float(video_fps), first_frame, last_frame, width, height, frames_count, audio_output

    @classmethod
    def IS_CHANGED(cls, url, max_seconds, select_every_nth, fps, skip_first_seconds, max_width, max_height):
        return url


# 🔊 LoadAudioByUrl
class LoadAudioByUrl:
    """
    Loads audio from a remote URL and returns it as ComfyUI AUDIO.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "https://example.com/audio.mp3", "multiline": False}),
                "max_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1}),
                "skip_first_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("AUDIO", "INT", "INT",)
    RETURN_NAMES = ("AUDIO", "SAMPLE_RATE", "SAMPLES",)
    FUNCTION = "load_audio"
    CATEGORY = "Remhes/Remote"
    OUTPUT_NODE = True

    def load_audio(self, url, max_seconds, skip_first_seconds):
        if not url or not url.strip():
            return (None, 0, 0)

        response = requests.get(url, stream=True)
        response.raise_for_status()

        audio_frames = []
        sample_rate = 0
        channels = 1

        with av.open(BytesIO(response.content)) as container:
            audio_stream = next((s for s in container.streams if s.type == "audio"), None)
            if audio_stream is None:
                raise ValueError("No audio stream found in media.")

            sample_rate = int(audio_stream.sample_rate or 44100)
            channels = int(audio_stream.channels or 1)

            for frame in container.decode(audio_stream):
                frame_np = frame.to_ndarray()

                if frame_np.ndim == 1:
                    frame_np = frame_np[np.newaxis, :]
                elif frame_np.ndim == 2:
                    if frame_np.shape[0] != channels and frame_np.shape[1] == channels:
                        frame_np = frame_np.T
                    elif frame_np.shape[0] != channels:
                        frame_np = frame_np.reshape(-1, channels).T
                else:
                    continue

                if np.issubdtype(frame_np.dtype, np.integer):
                    info = np.iinfo(frame_np.dtype)
                    scale = max(abs(float(info.min)), abs(float(info.max)))
                    if scale > 0:
                        frame_np = frame_np.astype(np.float32) / scale
                    else:
                        frame_np = frame_np.astype(np.float32)
                else:
                    frame_np = frame_np.astype(np.float32)

                audio_frames.append(frame_np)

        if not audio_frames:
            raise ValueError("Decoded zero audio frames.")

        audio_np = np.concatenate(audio_frames, axis=1)

        skip_samples = int(sample_rate * max(skip_first_seconds, 0.0))
        if skip_samples > 0:
            if skip_samples >= audio_np.shape[1]:
                raise ValueError("skip_first_seconds exceeds audio duration.")
            audio_np = audio_np[:, skip_samples:]

        if max_seconds > 0:
            max_samples = int(sample_rate * max(max_seconds, 0.0))
            if max_samples <= 0:
                raise ValueError("max_seconds produced an invalid sample count.")
            if audio_np.shape[1] > max_samples:
                audio_np = audio_np[:, :max_samples]

        if audio_np.shape[1] == 0:
            raise ValueError("No audio data after trimming.")

        waveform = torch.from_numpy(audio_np).unsqueeze(0).contiguous()  # (1, C, S)
        audio_output = {
            "waveform": waveform,
            "sample_rate": sample_rate,
        }

        return (audio_output, int(sample_rate), int(audio_np.shape[1]))

    @classmethod
    def IS_CHANGED(cls, url, max_seconds, skip_first_seconds):
        return url


# --- Register nodes with ComfyUI ---
NODE_CLASS_MAPPINGS = {
    "LoadImageByUrl": LoadImageByUrl,
    "LoadImagesByUrl": LoadImagesByUrl,
    "LoadVideoByUrl": LoadVideoByUrl,
    "LoadAudioByUrl": LoadAudioByUrl,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageByUrl": "🖼️ Load Image by URL",
    "LoadImagesByUrl": "🖼️ Load Images by URL",
    "LoadVideoByUrl": "🎥 Load Video by URL",
    "LoadAudioByUrl": "🔊 Load Audio by URL",
}
