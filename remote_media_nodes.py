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

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_image"
    CATEGORY = "Remhes/Remote"
    OUTPUT_NODE = True

    def load_image(self, url, max_width=0, max_height=0):
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

        return (tensor,)

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

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("image1", "image2", "image3", "image4", "image5",)
    FUNCTION = "load_images"
    CATEGORY = "Remhes/Remote"
    OUTPUT_NODE = True

    def load_images(self, url, max_width, max_height, url2, url3, url4, url5):
        image_loader = LoadImageByUrl()
        images = []
        
        # Load all URLs
        for img_url in [url, url2, url3, url4, url5]:
            if img_url and img_url.strip():
                try:
                    img_tensor = image_loader.load_image(img_url, max_width, max_height)[0]
                    images.append(img_tensor)
                except Exception as e:
                    # If loading fails, return None
                    images.append(None)
            else:
                # Empty URL - return None
                images.append(None)
        
        return tuple(images)

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
                "max_frames": ("INT", {"default": 32, "min": 0, "max": 10000, "step": 1}),
                "fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 120.0, "step": 0.1}),
                "max_width": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "max_height": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("IMAGES", "FPS", "FIRST_FRAME", "LAST_FRAME",)
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

    def load_video(self, url, max_frames, fps, max_width, max_height):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        buffer = BytesIO(response.content)

        container = av.open(buffer)
        video_stream = next(s for s in container.streams if s.type == "video")
        video_fps = float(video_stream.average_rate or 25)
        frames = []
        last_frame_data = None

        # Calculate frame interval based on fps
        # If fps is 0 or >= video fps, include all frames
        if fps <= 0 or fps >= video_fps:
            frame_interval = 1.0
        else:
            frame_interval = video_fps / fps

        frame_count = 0
        next_frame_index = 0.0
        
        for i, frame in enumerate(container.decode(video_stream)):
            img = frame.to_ndarray(format="rgb24")
            
            # Store as last_frame_data (will be added at end if not already included)
            last_frame_data = (img, i)
            
            # Determine if this frame should be included based on fps logic
            should_include = i >= int(next_frame_index)
            
            if not should_include:
                continue
                
            if max_frames > 0 and frame_count >= max_frames:
                break
            
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
        return video_tensor, float(fps), first_frame, last_frame

    @classmethod
    def IS_CHANGED(cls, url, max_frames, fps, max_width, max_height):
        return url


# --- Register nodes with ComfyUI ---
NODE_CLASS_MAPPINGS = {
    "LoadImageByUrl": LoadImageByUrl,
    "LoadImagesByUrl": LoadImagesByUrl,
    "LoadVideoByUrl": LoadVideoByUrl,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageByUrl": "🖼️ Load Image by URL",
    "LoadImagesByUrl": "🖼️ Load Images by URL",
    "LoadVideoByUrl": "🎥 Load Video by URL",
}
