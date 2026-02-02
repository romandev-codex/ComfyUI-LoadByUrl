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
                "max_width": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "max_height": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("IMAGES", "FPS", "FIRST_FRAME", "LAST_FRAME",)
    FUNCTION = "load_video"
    CATEGORY = "Remhes/Remote"

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

    def load_video(self, url, max_frames, frame_skip, max_width, max_height):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        buffer = BytesIO(response.content)

        container = av.open(buffer)
        video_stream = next(s for s in container.streams if s.type == "video")
        fps = float(video_stream.average_rate or 25)
        frames = []
        last_frame_data = None

        frame_count = 0
        for i, frame in enumerate(container.decode(video_stream)):
            is_first = i == 0
            should_include = is_first or (frame_skip == 0) or (i % (frame_skip + 1) == 0)
            
            img = frame.to_ndarray(format="rgb24")
            
            # Store as last_frame_data (will be added at end if not already included)
            last_frame_data = (img, i)
            
            # Skip middle frames based on frame_skip logic
            if not should_include:
                continue
                
            if max_frames > 0 and frame_count >= max_frames:
                break
            frame_count += 1

            img = self._resize_frame(img, max_width, max_height)
            tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
            frames.append(tensor)

            del img
            del tensor

        container.close()
        
        # Add last frame if it wasn't already included
        if last_frame_data and len(frames) > 0:
            last_img, last_idx = last_frame_data
            last_processed_idx = last_idx if should_include else -1
            if frame_skip > 0 and last_processed_idx != last_idx:
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
        return video_tensor, fps, first_frame, last_frame

    @classmethod
    def IS_CHANGED(cls, url, max_frames, frame_skip, max_width, max_height):
        return url


# 🌐 LoadByUrl (Auto-detect Image or Video)
class LoadByUrl:
    """
    Automatically detects if the URL is an image or video and loads it accordingly.
    Returns unified output format: IMAGES, FPS, FIRST_FRAME, LAST_FRAME
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "https://example.com/media.png", "multiline": False}),
                "max_frames": ("INT", {"default": 32, "min": 0, "max": 10000, "step": 1}),
                "frame_skip": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "max_width": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "max_height": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "url1": ("STRING", {"default": "", "multiline": False}),
                "url2": ("STRING", {"default": "", "multiline": False}),
                "url3": ("STRING", {"default": "", "multiline": False}),
                "url4": ("STRING", {"default": "", "multiline": False}),
                "url5": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("IMAGES", "FPS", "FIRST_FRAME", "LAST_FRAME", "image1", "image2", "image3", "image4", "image5",)
    FUNCTION = "load_media"
    CATEGORY = "Remhes/Remote"

    def _is_video_url(self, url, content_type):
        """Detect if URL points to a video based on extension and content-type"""
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v', '.mpg', '.mpeg')
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif')
        
        url_lower = url.lower()
        
        # Check content-type header first
        if content_type:
            if content_type.startswith('video/'):
                return True
            if content_type.startswith('image/'):
                return False
        
        # Check file extension
        if any(url_lower.endswith(ext) for ext in video_extensions):
            return True
        if any(url_lower.endswith(ext) for ext in image_extensions):
            return False
        
        # Default to image if uncertain
        return False

    def load_media(self, url, max_frames, frame_skip, max_width, max_height, url1, url2, url3, url4, url5):
        # Fetch headers to detect media type
        response = requests.head(url, allow_redirects=True)
        content_type = response.headers.get('content-type', '').lower()
        
        is_video = self._is_video_url(url, content_type)
        
        if is_video:
            # Use LoadVideoByUrl logic
            video_loader = LoadVideoByUrl()
            video_tensor, fps, first_frame, last_frame = video_loader.load_video(url, max_frames, frame_skip, max_width, max_height)
        else:
            # Load as image and format as video output
            image_loader = LoadImageByUrl()
            image_tensor = image_loader.load_image(url)[0]
            
            # For images: FPS = 0, and first/last frame are the same
            fps = 0.0
            first_frame = image_tensor
            last_frame = image_tensor
            video_tensor = image_tensor
        
        # Load additional images (url1-url5) - works only for images
        image_loader = LoadImageByUrl()
        additional_images = []
        
        for img_url in [url1, url2, url3, url4, url5]:
            if img_url and img_url.strip():
                try:
                    img_tensor = image_loader.load_image(img_url)[0]
                    additional_images.append(img_tensor)
                except Exception as e:
                    # If loading fails, return None
                    additional_images.append(None)
            else:
                # Empty URL - return None
                additional_images.append(None)
        
        return (video_tensor, fps, first_frame, last_frame, *additional_images)

    @classmethod
    def IS_CHANGED(cls, url, max_frames, frame_skip, max_width, max_height, url1, url2, url3, url4, url5):
        return f"{url}|{url1}|{url2}|{url3}|{url4}|{url5}"


# --- Register both nodes with ComfyUI ---
NODE_CLASS_MAPPINGS = {
    "LoadImageByUrl": LoadImageByUrl,
    "LoadVideoByUrl": LoadVideoByUrl,
    "LoadByUrl": LoadByUrl,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageByUrl": "🖼️ Load Image by URL",
    "LoadVideoByUrl": "🎥 Load Video by URL",
    "LoadByUrl": "🌐 Load by URL (Auto-detect)",
}
