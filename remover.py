#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import fitz  # PyMuPDF
import argparse
import os
import math
import logging
import cv2
import numpy as np
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass
from tqdm import tqdm
from PIL import Image
import io
import ffmpeg
import tempfile
import shutil

# Add ffmpeg to PATH if installed via winget (Windows only).
# On Linux/macOS, ffmpeg is expected to be on $PATH already (apt install ffmpeg
# / brew install ffmpeg).
if os.name == "nt":
    _ffmpeg_winget_path = os.path.expanduser("~/AppData/Local/Microsoft/WinGet/Packages")
    for _dir in os.listdir(_ffmpeg_winget_path) if os.path.isdir(_ffmpeg_winget_path) else []:
        if "FFmpeg" in _dir:
            _bin_path = os.path.join(_ffmpeg_winget_path, _dir)
            for _root, _dirs, _files in os.walk(_bin_path):
                if "ffmpeg.exe" in _files:
                    os.environ["PATH"] = _root + os.pathsep + os.environ.get("PATH", "")
                    break

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class WatermarkConfig:
    """Configuration for watermark detection and removal."""
    # Search area: Increased to cover 'NotebookLM' + Logo
    # The 'touching edge' logic prevents removing slide content that enters this area.
    search_margin_x: int = 350 
    search_margin_y: int = 70
    
    # Pixel scanning settings
    pixel_threshold: int = 30
    
    # Vector detection limits
    vector_max_width: int = 150
    vector_max_height: int = 40
    
    # Padding for the patch area
    patch_padding: int = 10
    
    # PDF Image Quality
    pdf_dpi_scale: float = 2.0 

    # Inpainting settings
    inpaint_radius: int = 3

class WatermarkRemover:
    def __init__(self, config: WatermarkConfig = WatermarkConfig()):
        self.config = config

    @staticmethod
    def hex_to_rgb(hex_str: Optional[str]) -> Optional[Tuple[float, float, float]]:
        """Converts hex color string to normalized RGB tuple (0.0 - 1.0)."""
        if not hex_str: return None
        hex_str = hex_str.lstrip('#')
        try:
            return tuple(int(hex_str[i:i+2], 16)/255.0 for i in (0, 2, 4))
        except ValueError: return None

    def _clean_image_array(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Core logic: Takes a BGR numpy array (image chunk), detects the watermark 
        using local contrast (blur difference), and inpaints it.
        """
        try:
            # 1. Median Blur to estimate 'background' without thin text
            # Reduced kernel size slightly to be less aggressive on nearby shapes
            blurred = cv2.medianBlur(img_bgr, 15)
            
            # 2. Difference between Original and Blur
            diff = cv2.absdiff(img_bgr, blurred)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            
            # 3. Threshold to get mask of text/sharp details
            # Lowered threshold to capture faint shadows (prevent black spots)
            _, mask = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)
            
            # 4. Filter Mask: Remove large continuous blobs that touch the left/top edge
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            h, w = mask.shape
            
            new_mask = np.zeros_like(mask)
            
            for i in range(1, num_labels): # Skip background (0)
                x, y, stat_w, stat_h, area = stats[i]
                
                if stat_w > w * 0.9 or stat_h > h * 0.9:
                    continue
                if x == 0 or y == 0:
                    continue
                    
                new_mask[labels == i] = 255
            
            mask = new_mask
            
            # 5. Dilate to cover anti-aliasing artifacts and shadows
            # Increased iterations to fully consume text rims
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(mask, kernel, iterations=3)
            
            if cv2.countNonZero(mask) == 0:
                return img_bgr

            # 6. Inpaint
            # Switched to TELEA as it handles small spot removal with fewer artifacts than NS
            cleaned = cv2.inpaint(img_bgr, mask, self.config.inpaint_radius, cv2.INPAINT_TELEA)
            
            return cleaned
        except Exception as e:
            logger.warning(f"Inpainting failed: {e}")
            return img_bgr

    # --- PDF Processing Methods ---

    def process_pdf(self, input_path: str, output_path: str, preview: bool = False) -> bool:
        """
        Processes a PDF by 'patching' the watermark area.
        Instead of a solid color, it rasterizes the corner, cleans it with AI/Inpainting,
        and pastes it back over the original watermark.
        """
        try:
            doc = fitz.open(input_path)
        except Exception as e:
            logger.error(f"Could not open file {input_path}: {e}")
            return False

        filename = os.path.basename(input_path)
        pbar = tqdm(enumerate(doc), total=len(doc), desc=f"Processing {filename} (PDF)", unit="page")
        
        pages_modified = 0

        for i, page in pbar:
            if preview and i > 0: break
            
            w, h = page.rect.width, page.rect.height
            
            # Define search area (bottom right corner)
            x0 = w - self.config.search_margin_x
            y0 = h - self.config.search_margin_y
            
            if x0 < 0: x0 = 0
            if y0 < 0: y0 = 0
            
            search_rect = fitz.Rect(x0, y0, w, h)
            
            # Heuristic check to skip clean pages
            has_watermark = False
            if page.search_for("NotebookLM", clip=search_rect):
                has_watermark = True
            else:
                drawings = page.get_drawings()
                for s in drawings:
                    if s["rect"].intersects(search_rect) and s["rect"].width < 150:
                        has_watermark = True
                        break
            
            # Always attempt patch if heuristic matches, or we could force it.
            # Given the improved "smart" cleaning which ignores non-watermark content,
            # we can be aggressive.
            
            # 1. Capture high-res image of the corner
            mat = fitz.Matrix(self.config.pdf_dpi_scale, self.config.pdf_dpi_scale)
            pix = page.get_pixmap(clip=search_rect, matrix=mat)
            
            img_data = np.frombuffer(pix.samples, dtype=np.uint8)
            
            if pix.n == 4:
                img_data = img_data.reshape(pix.h, pix.w, 4)
                img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img_data = img_data.reshape(pix.h, pix.w, 3)
                img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
            else:
                continue

            cleaned_bgr = self._clean_image_array(img_bgr)
            
            cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cleaned_rgb)
            
            img_byte_arr = io.BytesIO()
            pil_img.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            
            page.insert_image(search_rect, stream=img_bytes)
            pages_modified += 1

        try:
            doc.save(output_path, garbage=3, deflate=True, clean=True)
            doc.close()
            logger.info(f"Saved {output_path} ({pages_modified} pages patched)")
            return True
        except Exception as e:
            logger.error(f"Error saving PDF {output_path}: {e}")
            return False

    # --- Video Processing Methods ---

    def process_video(self, input_path: str, output_path: str, preview: bool = False) -> bool:
        """
        Processes a video file frame by frame, removing watermarks and preserving audio.
        """
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                logger.error(f"Could not open video {input_path}")
                return False

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if preview:
                preview_seconds = 5
                total_frames = min(total_frames, int(fps * preview_seconds))

            filename = os.path.basename(input_path)

            temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
            os.close(temp_fd)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

            if not out.isOpened():
                logger.error(f"Could not create video writer for {temp_path}")
                cap.release()
                return False

            margin_x = self.config.search_margin_x
            margin_y = self.config.search_margin_y
            x_start = max(0, width - margin_x)
            y_start = max(0, height - margin_y)

            pbar = tqdm(total=total_frames, desc=f"Processing {filename} (Video)", unit="frame")

            frame_count = 0
            while frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                roi = frame[y_start:height, x_start:width]
                cleaned_roi = self._clean_image_array(roi)
                frame[y_start:height, x_start:width] = cleaned_roi

                out.write(frame)
                frame_count += 1
                pbar.update(1)

            pbar.close()
            cap.release()
            out.release()

            try:
                input_stream = ffmpeg.input(input_path)
                video_stream = ffmpeg.input(temp_path)

                output = ffmpeg.output(
                    video_stream.video,
                    input_stream.audio,
                    output_path,
                    vcodec='copy',
                    acodec='copy'
                )
                output = output.overwrite_output()
                output.run(quiet=True)

                os.remove(temp_path)
                logger.info(f"Saved cleaned video to {output_path}")
                return True

            except Exception as e:
                logger.warning(f"Could not merge audio (ffmpeg may not be installed): {e}")
                if os.path.exists(temp_path):
                    shutil.move(temp_path, output_path)
                    logger.info(f"Saved cleaned video (no audio) to {output_path}")
                    return True
                return False

        except Exception as e:
            logger.error(f"Error processing video {input_path}: {e}")
            return False

    def trim_video(self, input_path: str, output_path: str, start_time: float, end_time: float) -> bool:
        """Trims a video to the specified time range using ffmpeg."""
        try:
            duration = end_time - start_time
            (
                ffmpeg
                .input(input_path, ss=start_time, t=duration)
                .output(output_path, c='copy')
                .overwrite_output()
                .run(quiet=True)
            )
            logger.info(f"Trimmed video saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error trimming video {input_path}: {e}")
            return False

    def get_video_info(self, input_path: str) -> Optional[dict]:
        """Gets video metadata (duration, fps, dimensions)."""
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                return None

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0

            cap.release()
            return {
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "duration": duration
            }
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return None

    def add_image_overlay(self, video_path: str, image_path: str, output_path: str,
                          x: int, y: int, scale: float = 1.0,
                          start_time: float = 0, end_time: float = None) -> bool:
        """Adds an image overlay to a video at specified position."""
        try:
            video_info = self.get_video_info(video_path)
            if not video_info:
                return False

            if end_time is None:
                end_time = video_info['duration']

            video = ffmpeg.input(video_path)
            image = ffmpeg.input(image_path)

            if scale != 1.0:
                image = image.filter('scale', f'iw*{scale}', f'ih*{scale}')

            overlay = video.overlay(image, x=x, y=y,
                                    enable=f'between(t,{start_time},{end_time})')

            output = ffmpeg.output(overlay, video.audio, output_path,
                                   vcodec='libx264', acodec='copy')
            output.overwrite_output().run(quiet=True)

            logger.info(f"Added image overlay, saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error adding image overlay: {e}")
            return False

    def merge_videos(self, video_paths: List[str], output_path: str) -> bool:
        """Merges multiple videos into one."""
        try:
            if len(video_paths) < 2:
                logger.error("Need at least 2 videos to merge")
                return False

            # Create concat file
            concat_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
            for path in video_paths:
                abs_path = os.path.abspath(path).replace('\\', '/')
                concat_file.write(f"file '{abs_path}'\n")
            concat_file.close()

            (
                ffmpeg
                .input(concat_file.name, format='concat', safe=0)
                .output(output_path, c='copy')
                .overwrite_output()
                .run(quiet=True)
            )

            os.unlink(concat_file.name)
            logger.info(f"Merged {len(video_paths)} videos to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error merging videos: {e}")
            return False

    def extract_frame(self, video_path: str, output_path: str, time: float) -> bool:
        """Extracts a single frame from video at specified time."""
        try:
            (
                ffmpeg
                .input(video_path, ss=time)
                .output(output_path, vframes=1)
                .overwrite_output()
                .run(quiet=True)
            )
            logger.info(f"Extracted frame at {time}s to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error extracting frame: {e}")
            return False

    def change_speed(self, input_path: str, output_path: str, speed: float) -> bool:
        """Changes video playback speed. speed > 1 = faster, < 1 = slower."""
        try:
            video = ffmpeg.input(input_path)

            # Video speed filter
            v = video.video.filter('setpts', f'{1/speed}*PTS')

            # Audio speed filter (atempo only supports 0.5 to 2.0)
            a = video.audio
            if 0.5 <= speed <= 2.0:
                a = a.filter('atempo', speed)
            elif speed > 2.0:
                # Chain atempo filters for speeds > 2
                temp_speed = speed
                while temp_speed > 2.0:
                    a = a.filter('atempo', 2.0)
                    temp_speed /= 2.0
                if temp_speed > 1.0:
                    a = a.filter('atempo', temp_speed)
            else:
                # Chain atempo filters for speeds < 0.5
                temp_speed = speed
                while temp_speed < 0.5:
                    a = a.filter('atempo', 0.5)
                    temp_speed *= 2.0
                if temp_speed < 1.0:
                    a = a.filter('atempo', temp_speed)

            output = ffmpeg.output(v, a, output_path, vcodec='libx264', acodec='aac')
            output.overwrite_output().run(quiet=True)

            logger.info(f"Changed speed to {speed}x, saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error changing speed: {e}")
            return False

    def export_video(self, input_path: str, output_path: str,
                     format: str = 'mp4', quality: str = 'medium',
                     resolution: str = None) -> bool:
        """Exports video with specified format and quality."""
        try:
            quality_presets = {
                'low': {'crf': 28, 'preset': 'fast'},
                'medium': {'crf': 23, 'preset': 'medium'},
                'high': {'crf': 18, 'preset': 'slow'},
                'best': {'crf': 15, 'preset': 'veryslow'}
            }

            preset = quality_presets.get(quality, quality_presets['medium'])
            video = ffmpeg.input(input_path)

            if resolution:
                w, h = resolution.split('x')
                video = video.filter('scale', w, h)

            codec_map = {
                'mp4': {'vcodec': 'libx264', 'acodec': 'aac'},
                'webm': {'vcodec': 'libvpx-vp9', 'acodec': 'libopus'},
                'avi': {'vcodec': 'mpeg4', 'acodec': 'mp3'},
                'mov': {'vcodec': 'libx264', 'acodec': 'aac'},
                'mkv': {'vcodec': 'libx264', 'acodec': 'aac'}
            }

            codecs = codec_map.get(format, codec_map['mp4'])

            output = ffmpeg.output(
                video, output_path,
                vcodec=codecs['vcodec'],
                acodec=codecs['acodec'],
                crf=preset['crf'],
                preset=preset['preset']
            )
            output.overwrite_output().run(quiet=True)

            logger.info(f"Exported to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting video: {e}")
            return False

    # --- Image Processing Methods ---

    def process_image(self, input_path: str, output_path: str) -> bool:
        """Processes an image using Gradient-Aware Inpainting, preserving Alpha."""
        try:
            # Load with Alpha if present
            img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                return False

            h, w = img.shape[:2]
            
            # Handle Alpha Channel
            has_alpha = False
            if len(img.shape) == 3 and img.shape[2] == 4:
                has_alpha = True
                b, g, r, a = cv2.split(img)
                img_bgr = cv2.merge([b, g, r])
            else:
                img_bgr = img

            # ROI: Bottom Right
            margin_x = self.config.search_margin_x
            margin_y = self.config.search_margin_y
            x_start = max(0, w - margin_x)
            y_start = max(0, h - margin_y)
            
            roi_bgr = img_bgr[y_start:h, x_start:w]
            
            # Clean ROI
            cleaned_roi = self._clean_image_array(roi_bgr)
            
            # Restore
            img_bgr[y_start:h, x_start:w] = cleaned_roi
            
            # Merge Alpha back if needed
            if has_alpha:
                img_final = cv2.merge([*cv2.split(img_bgr), a])
            else:
                img_final = img_bgr
            
            cv2.imwrite(output_path, img_final)
            logger.info(f"Saved cleaned image to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error processing image {input_path}: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="NotebookLM Watermark Remover (Smart Inpainting)")
    parser.add_argument("path", help="File (PDF/PNG/JPG/MP4) or directory")
    parser.add_argument("-o", "--output", help="Output path")
    parser.add_argument("--preview", action="store_true", help="Process only first page (PDF) or first 5 seconds (video)")

    args = parser.parse_args()
    remover = WatermarkRemover()

    tasks = []
    image_exts = ('.png', '.jpg', '.jpeg', '.webp')
    video_exts = ('.mp4', '.mov', '.avi', '.mkv')
    supported_exts = ('.pdf',) + image_exts + video_exts
    
    if os.path.isdir(args.path):
        files = sorted([f for f in os.listdir(args.path) if f.lower().endswith(supported_exts)])
        tasks = [os.path.join(args.path, f) for f in files]
        logger.info(f"Found {len(tasks)} supported files.")
    elif os.path.isfile(args.path) and args.path.lower().endswith(supported_exts):
        tasks = [args.path]
    else:
        logger.error("Invalid path.")
        return

    for input_path in tasks:
        ext = os.path.splitext(input_path)[1].lower()
        if args.output and len(tasks) == 1:
            out_path = args.output
        else:
            base, _ = os.path.splitext(input_path)
            out_path = f"{base}_cleaned{ext}"
            
        if ext == '.pdf':
            remover.process_pdf(input_path, out_path, preview=args.preview)
        elif ext in video_exts:
            remover.process_video(input_path, out_path, preview=args.preview)
        else:
            remover.process_image(input_path, out_path)

if __name__ == "__main__":
    main()
