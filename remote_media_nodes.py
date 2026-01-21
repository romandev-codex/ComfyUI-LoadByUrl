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
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_image"
    CATEGORY = "Remhes/Remote"

    def load_image(self, url):
        response = requests.get(url, stream=True)
        response.raise_for_status()

        image = Image.open(BytesIO(response.content))
        image = ImageOps.exif_transpose(image)
        arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W, 3)

        return (tensor,)

    @classmethod
    def IS_CHANGED(cls, url):
        return url


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
                "max_frames": ("INT", {"default": 32, "min": 0, "max": 10000, "step": 1}),
                "frame_skip": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "force_width": ("INT", {"default": 100, "min": 0, "max": 1000, "step": 1}),
                "force_height": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT",)
    RETURN_NAMES = ("IMAGES", "FPS",)
    FUNCTION = "load_video"
    CATEGORY = "Remhes/Remote"

    def load_video(self, url, max_frames, frame_skip, force_width, force_height):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        buffer = BytesIO(response.content)

        container = av.open(buffer)
        video_stream = next(s for s in container.streams if s.type == "video")
        fps = float(video_stream.average_rate or 25)
        frames = []

        frame_count = 0
        for i, frame in enumerate(container.decode(video_stream)):
            if frame_skip > 0 and i % (frame_skip + 1) != 0:
                continue
            if max_frames > 0 and frame_count >= max_frames:
                break
            frame_count += 1
            img = frame.to_ndarray(format="rgb24")

            # Resize if force_width or force_height is specified
            if force_width > 0 or force_height > 0:
                from PIL import Image
                pil_img = Image.fromarray(img)
                
                if force_width > 0 and force_height > 0:
                    # Both dimensions specified
                    new_width, new_height = force_width, force_height
                elif force_width > 0:
                    # Only width specified, maintain aspect ratio
                    aspect_ratio = pil_img.height / pil_img.width
                    new_width = force_width
                    new_height = int(force_width * aspect_ratio)
                else:
                    # Only height specified, maintain aspect ratio
                    aspect_ratio = pil_img.width / pil_img.height
                    new_height = force_height
                    new_width = int(force_height * aspect_ratio)
                
                pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
                img = np.array(pil_img)
                del pil_img  # Clean up PIL image

            tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
            frames.append(tensor)

            # Clean up temporary variables
            del img
            del tensor

        container.close()

        if not frames:
            raise ValueError("No frames decoded from video.")

        video_tensor = torch.cat(frames, dim=0)  # (frames, H, W, 3)
        return video_tensor, fps

    @classmethod
    def IS_CHANGED(cls, url, max_frames, frame_skip, force_width, force_height):
        return url


# --- Register both nodes with ComfyUI ---
NODE_CLASS_MAPPINGS = {
    "LoadImageByUrl": LoadImageByUrl,
    "LoadVideoByUrl": LoadVideoByUrl,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageByUrl": "🖼️ Load Image by URL",
    "LoadVideoByUrl": "🎥 Load Video by URL",
}
