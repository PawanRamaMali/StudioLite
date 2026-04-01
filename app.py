#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import tempfile
import os
import ffmpeg
from audio_recorder_streamlit import audio_recorder
from remover import WatermarkRemover
from youtube_uploader import (
    YouTubeUploader, check_dependencies, check_client_secrets, get_category_list
)
from transcriber import (
    Transcriber, TranscriptionConfig, check_whisperx_installed,
    get_language_list, get_language_code, get_device,
    MODEL_SIZES, COMPUTE_TYPES, OUTPUT_FORMATS
)
from reelforge import (
    load_config, save_config, list_local_models, list_ollama_models,
    rf_generate_full, check_ollama_connection, check_backend_status,
    get_gguf_models, download_model, RECOMMENDED_MODELS, check_llamacpp_available,
    select_backend, get_backend,
    # Enhanced features
    RECOMMENDED_IMAGE_MODELS, IMAGE_STYLE_PRESETS, SUBTITLE_STYLES,
    MOTION_EFFECTS, COLOR_FILTERS, download_image_model, ASPECT_RATIOS,
    list_background_music, get_music_dir,
)

st.set_page_config(page_title="StudioLite - Video Editor", layout="wide")

remover = WatermarkRemover()

# Initialize session state
if "video_cache" not in st.session_state:
    st.session_state.video_cache = {}
if "current_tool" not in st.session_state:
    st.session_state.current_tool = None
if "youtube_uploader" not in st.session_state:
    st.session_state.youtube_uploader = None

# Sidebar navigation
st.sidebar.title("StudioLite")
tool = st.sidebar.radio(
    "Select Tool",
    ["Remove Watermark", "Trim / Cut", "Add Image Overlay", "Change Speed",
     "Merge Videos", "Extract Frame", "Export Video", "Upscale Video",
     "Video Editor", "Audio Studio", "Transcribe", "View & Publish",
     "ReelForge", "Video Generator", "Story Mode", "Logs", "Settings"]
)

# Clear cache when switching tools
if st.session_state.current_tool != tool:
    st.session_state.video_cache = {}
    st.session_state.current_tool = tool

st.sidebar.markdown("---")
st.sidebar.markdown("**Supported formats:** MP4, MOV, AVI, MKV")


def save_uploaded_file(uploaded_file, cache_key="input"):
    """Save uploaded file to temp location and cache bytes."""
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    file_bytes = uploaded_file.read()

    # Cache the bytes
    st.session_state.video_cache[cache_key] = file_bytes

    # Write to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(file_bytes)
        return tmp.name


def get_cached_video(cache_key):
    """Get cached video bytes."""
    return st.session_state.video_cache.get(cache_key)


def cache_video_file(path, cache_key):
    """Read video file and cache its bytes."""
    with open(path, "rb") as f:
        video_bytes = f.read()
    st.session_state.video_cache[cache_key] = video_bytes
    return video_bytes


def display_video_info(video_info):
    """Display video metadata in columns."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Duration", f"{video_info['duration']:.1f}s")
    col2.metric("FPS", f"{video_info['fps']:.1f}")
    col3.metric("Resolution", f"{video_info['width']}x{video_info['height']}")
    col4.metric("Frames", video_info['frame_count'])


def display_cached_video(cache_key, label=None):
    """Display video from cache."""
    video_bytes = get_cached_video(cache_key)
    if video_bytes:
        if label:
            st.subheader(label)
        st.video(video_bytes)
        return True
    return False


def _is_h264_mp4(file_path):
    """Check if file is already an H.264 MP4 (browser-compatible)."""
    try:
        probe = ffmpeg.probe(file_path)
        video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
        if video_stream and video_stream.get('codec_name') == 'h264' and file_path.lower().endswith('.mp4'):
            return True
    except Exception:
        pass
    return False


def display_video_browser_compatible(input_path, source_cache_key, label=None):
    """
    Display video with automatic conversion to H.264 for browser compatibility.

    Args:
        input_path: Path to the input video file
        source_cache_key: Cache key for the source video bytes
        label: Optional label to display above the video
    """
    converted_key = f"{source_cache_key}_converted"

    if get_cached_video(converted_key) is None:
        if _is_h264_mp4(input_path):
            # Already browser-compatible, use original bytes directly
            st.session_state.video_cache[converted_key] = get_cached_video(source_cache_key)
        else:
            # Convert to H.264 MP4 for browser compatibility (fast preset for preview)
            with st.spinner("Preparing video for preview..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    preview_path = tmp.name
                if remover.export_video(input_path, preview_path, "mp4", "low", None):
                    cache_video_file(preview_path, converted_key)
                    os.unlink(preview_path)
                else:
                    # Fallback to original if conversion fails
                    st.session_state.video_cache[converted_key] = get_cached_video(source_cache_key)

    if label:
        st.subheader(label)
    st.video(get_cached_video(converted_key))


def download_button(cache_key, filename, label="Download Video"):
    """Create download button for cached video."""
    video_bytes = get_cached_video(cache_key)
    if video_bytes:
        st.download_button(
            label=label,
            data=video_bytes,
            file_name=filename,
            mime="video/mp4"
        )


# =====================
# REMOVE WATERMARK
# =====================
if tool == "Remove Watermark":
    st.title("Remove Watermark")
    st.markdown("Remove NotebookLM watermarks from videos, PDFs, and images using AI inpainting.")

    uploaded_file = st.file_uploader(
        "Upload a file",
        type=["mp4", "mov", "avi", "mkv", "pdf", "png", "jpg", "jpeg", "webp"],
        key="watermark_upload"
    )

    if uploaded_file:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        is_video = file_ext in [".mp4", ".mov", ".avi", ".mkv"]

        # Check if we need to save the file (new upload)
        if get_cached_video("input") is None:
            input_path = save_uploaded_file(uploaded_file, "input")
        else:
            # Recreate temp file from cache
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(get_cached_video("input"))
                input_path = tmp.name

        if is_video:
            display_video_browser_compatible(input_path, "input", "Original Video")

            video_info = remover.get_video_info(input_path)
            if video_info:
                display_video_info(video_info)

                if st.button("Remove Watermark", type="primary"):
                    with st.spinner("Removing watermark (this may take a while)..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                            output_path = tmp.name

                        if remover.process_video(input_path, output_path):
                            st.success("Done!")
                            cache_video_file(output_path, "result")
                            os.unlink(output_path)
                        else:
                            st.error("Failed to process video")

                # Display cached result
                if display_cached_video("result", "Result"):
                    download_button("result", f"{os.path.splitext(uploaded_file.name)[0]}_cleaned.mp4")

        elif file_ext == ".pdf":
            preview_mode = st.checkbox("Preview mode (first page only)")
            if st.button("Remove Watermark", type="primary"):
                with st.spinner("Processing PDF..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        output_path = tmp.name
                    if remover.process_pdf(input_path, output_path, preview=preview_mode):
                        st.success("Done!")
                        with open(output_path, "rb") as f:
                            st.session_state.video_cache["pdf_result"] = f.read()
                        os.unlink(output_path)
                    else:
                        st.error("Failed to process PDF")

            if "pdf_result" in st.session_state.video_cache:
                st.download_button("Download PDF", st.session_state.video_cache["pdf_result"],
                                   f"{os.path.splitext(uploaded_file.name)[0]}_cleaned.pdf",
                                   "application/pdf")
        else:
            st.image(input_path)
            if st.button("Remove Watermark", type="primary"):
                with st.spinner("Processing image..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                        output_path = tmp.name
                    if remover.process_image(input_path, output_path):
                        st.success("Done!")
                        st.image(output_path)
                        with open(output_path, "rb") as f:
                            st.session_state.video_cache["img_result"] = f.read()
                        os.unlink(output_path)
                    else:
                        st.error("Failed to process image")

            if "img_result" in st.session_state.video_cache:
                st.download_button("Download Image", st.session_state.video_cache["img_result"],
                                   f"{os.path.splitext(uploaded_file.name)[0]}_cleaned{file_ext}",
                                   "image/png")

        os.unlink(input_path)


# =====================
# TRIM / CUT
# =====================
elif tool == "Trim / Cut":
    st.title("Trim / Cut Video")
    st.markdown("Cut a portion of your video by selecting start and end times.")

    uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"], key="trim_upload")

    if uploaded_file:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()

        if get_cached_video("input") is None:
            input_path = save_uploaded_file(uploaded_file, "input")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(get_cached_video("input"))
                input_path = tmp.name

        display_video_browser_compatible(input_path, "input", "Original Video")

        video_info = remover.get_video_info(input_path)
        if video_info:
            display_video_info(video_info)
            duration = video_info['duration']

            st.subheader("Select Range")
            trim_range = st.slider(
                "Time range",
                min_value=0.0,
                max_value=duration,
                value=(0.0, duration),
                step=0.1,
                format="%.1fs"
            )
            start_time, end_time = trim_range
            st.write(f"Duration: {end_time - start_time:.1f}s")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Preview"):
                    with st.spinner("Creating preview..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                            preview_path = tmp.name
                        if remover.trim_video(input_path, preview_path, start_time, end_time):
                            cache_video_file(preview_path, "preview")
                            os.unlink(preview_path)

            with col2:
                if st.button("Export Trimmed Video", type="primary"):
                    with st.spinner("Trimming..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                            output_path = tmp.name
                        if remover.trim_video(input_path, output_path, start_time, end_time):
                            st.success("Done!")
                            cache_video_file(output_path, "result")
                            os.unlink(output_path)

            # Display cached preview
            display_cached_video("preview", "Preview")

            # Display cached result
            if display_cached_video("result", "Result"):
                download_button("result", f"{os.path.splitext(uploaded_file.name)[0]}_trimmed.mp4")

        os.unlink(input_path)


# =====================
# ADD IMAGE OVERLAY
# =====================
elif tool == "Add Image Overlay":
    st.title("Add Image Overlay")
    st.markdown("Add an image (logo, watermark) to your video.")

    col_video, col_image = st.columns(2)

    with col_video:
        video_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"], key="overlay_video")

    with col_image:
        image_file = st.file_uploader("Upload image to overlay", type=["png", "jpg", "jpeg"], key="overlay_image")

    if video_file and image_file:
        video_ext = os.path.splitext(video_file.name)[1].lower()
        image_ext = os.path.splitext(image_file.name)[1].lower()

        if get_cached_video("input") is None:
            video_path = save_uploaded_file(video_file, "input")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=video_ext) as tmp:
                tmp.write(get_cached_video("input"))
                video_path = tmp.name

        if get_cached_video("overlay_img") is None:
            image_bytes = image_file.read()
            st.session_state.video_cache["overlay_img"] = image_bytes
            with tempfile.NamedTemporaryFile(delete=False, suffix=image_ext) as tmp:
                tmp.write(image_bytes)
                image_path = tmp.name
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=image_ext) as tmp:
                tmp.write(get_cached_video("overlay_img"))
                image_path = tmp.name

        display_video_browser_compatible(video_path, "input", "Original Video")

        video_info = remover.get_video_info(video_path)
        if video_info:
            display_video_info(video_info)

            st.subheader("Overlay Settings")
            col1, col2, col3 = st.columns(3)

            with col1:
                x_pos = st.number_input("X Position", min_value=0, max_value=video_info['width'], value=10)
                y_pos = st.number_input("Y Position", min_value=0, max_value=video_info['height'], value=10)

            with col2:
                scale = st.slider("Scale", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

            with col3:
                start_time = st.number_input("Start time (s)", min_value=0.0,
                                             max_value=video_info['duration'], value=0.0)
                end_time = st.number_input("End time (s)", min_value=0.0,
                                           max_value=video_info['duration'], value=video_info['duration'])

            st.image(image_path, caption="Overlay image", width=200)

            if st.button("Add Overlay", type="primary"):
                with st.spinner("Adding overlay..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                        output_path = tmp.name
                    if remover.add_image_overlay(video_path, image_path, output_path,
                                                  int(x_pos), int(y_pos), scale, start_time, end_time):
                        st.success("Done!")
                        cache_video_file(output_path, "result")
                        os.unlink(output_path)
                    else:
                        st.error("Failed to add overlay")

            # Display cached result
            if display_cached_video("result", "Result"):
                download_button("result", f"{os.path.splitext(video_file.name)[0]}_overlay.mp4")

        os.unlink(video_path)
        os.unlink(image_path)


# =====================
# CHANGE SPEED
# =====================
elif tool == "Change Speed":
    st.title("Change Video Speed")
    st.markdown("Speed up or slow down your video.")

    uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"], key="speed_upload")

    if uploaded_file:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()

        if get_cached_video("input") is None:
            input_path = save_uploaded_file(uploaded_file, "input")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(get_cached_video("input"))
                input_path = tmp.name

        display_video_browser_compatible(input_path, "input", "Original Video")

        video_info = remover.get_video_info(input_path)
        if video_info:
            display_video_info(video_info)

            st.subheader("Speed Settings")
            speed = st.slider("Speed multiplier", min_value=0.25, max_value=4.0, value=1.0, step=0.25)

            new_duration = video_info['duration'] / speed
            st.write(f"New duration: {new_duration:.1f}s (original: {video_info['duration']:.1f}s)")

            if st.button("Apply Speed Change", type="primary"):
                with st.spinner("Processing..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                        output_path = tmp.name
                    if remover.change_speed(input_path, output_path, speed):
                        st.success("Done!")
                        cache_video_file(output_path, "result")
                        os.unlink(output_path)
                    else:
                        st.error("Failed to change speed")

            # Display cached result
            if display_cached_video("result", "Result"):
                download_button("result", f"{os.path.splitext(uploaded_file.name)[0]}_{speed}x.mp4")

        os.unlink(input_path)


# =====================
# MERGE VIDEOS
# =====================
elif tool == "Merge Videos":
    st.title("Merge Videos")
    st.markdown("Combine multiple videos into one. Videos should have the same resolution and codec.")

    uploaded_files = st.file_uploader(
        "Upload videos (select multiple)",
        type=["mp4", "mov", "avi", "mkv"],
        accept_multiple_files=True,
        key="merge_upload"
    )

    if uploaded_files and len(uploaded_files) >= 2:
        video_paths = []
        for i, f in enumerate(uploaded_files):
            file_ext = os.path.splitext(f.name)[1].lower()
            cache_key = f"merge_{i}"

            if get_cached_video(cache_key) is None:
                file_bytes = f.read()
                st.session_state.video_cache[cache_key] = file_bytes
            else:
                file_bytes = get_cached_video(cache_key)

            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(file_bytes)
                video_paths.append(tmp.name)

            st.write(f"**{i+1}. {f.name}**")
            video_info = remover.get_video_info(video_paths[-1])
            if video_info:
                st.write(f"   Duration: {video_info['duration']:.1f}s, Resolution: {video_info['width']}x{video_info['height']}")

        if st.button("Merge Videos", type="primary"):
            with st.spinner("Merging videos..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    output_path = tmp.name
                if remover.merge_videos(video_paths, output_path):
                    st.success("Done!")
                    cache_video_file(output_path, "result")
                    os.unlink(output_path)
                else:
                    st.error("Failed to merge videos. Ensure videos have compatible formats.")

        # Display cached result
        if display_cached_video("result", "Result"):
            download_button("result", "merged_video.mp4")

        for path in video_paths:
            os.unlink(path)

    elif uploaded_files:
        st.warning("Please upload at least 2 videos to merge.")


# =====================
# EXTRACT FRAME
# =====================
elif tool == "Extract Frame":
    st.title("Extract Frame")
    st.markdown("Extract a single frame from your video as an image.")

    uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"], key="frame_upload")

    if uploaded_file:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()

        if get_cached_video("input") is None:
            input_path = save_uploaded_file(uploaded_file, "input")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(get_cached_video("input"))
                input_path = tmp.name

        display_video_browser_compatible(input_path, "input", "Video")

        video_info = remover.get_video_info(input_path)
        if video_info:
            display_video_info(video_info)

            time_point = st.slider("Select time", min_value=0.0,
                                   max_value=video_info['duration'], value=0.0, step=0.1, format="%.1fs")

            if st.button("Extract Frame", type="primary"):
                with st.spinner("Extracting..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                        output_path = tmp.name
                    if remover.extract_frame(input_path, output_path, time_point):
                        st.success("Done!")
                        with open(output_path, "rb") as f:
                            st.session_state.video_cache["frame_result"] = f.read()
                        st.session_state.video_cache["frame_time"] = time_point
                        os.unlink(output_path)
                    else:
                        st.error("Failed to extract frame")

            # Display cached frame
            if "frame_result" in st.session_state.video_cache:
                st.subheader("Extracted Frame")
                st.image(st.session_state.video_cache["frame_result"])
                frame_time = st.session_state.video_cache.get("frame_time", 0)
                st.download_button("Download Frame", st.session_state.video_cache["frame_result"],
                                   f"frame_{frame_time:.1f}s.png", "image/png")

        os.unlink(input_path)


# =====================
# EXPORT VIDEO
# =====================
elif tool == "Export Video":
    st.title("Export Video")
    st.markdown("Convert video to different formats and quality settings.")

    uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"], key="export_upload")

    if uploaded_file:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()

        if get_cached_video("input") is None:
            input_path = save_uploaded_file(uploaded_file, "input")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(get_cached_video("input"))
                input_path = tmp.name

        display_video_browser_compatible(input_path, "input", "Original Video")

        video_info = remover.get_video_info(input_path)
        if video_info:
            display_video_info(video_info)

            st.subheader("Export Settings")
            col1, col2, col3 = st.columns(3)

            with col1:
                output_format = st.selectbox("Format", ["mp4", "webm", "avi", "mov", "mkv"])

            with col2:
                quality = st.selectbox("Quality", ["low", "medium", "high", "best"], index=1)

            with col3:
                resolution_options = ["Original", "1920x1080", "1280x720", "854x480", "640x360"]
                resolution = st.selectbox("Resolution", resolution_options)

            if st.button("Export", type="primary"):
                with st.spinner("Exporting..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_format}") as tmp:
                        output_path = tmp.name

                    res = None if resolution == "Original" else resolution

                    if remover.export_video(input_path, output_path, output_format, quality, res):
                        st.success("Done!")
                        with open(output_path, "rb") as f:
                            st.session_state.video_cache["export_result"] = f.read()
                        st.session_state.video_cache["export_format"] = output_format
                        st.session_state.video_cache["export_name"] = uploaded_file.name
                        os.unlink(output_path)
                    else:
                        st.error("Failed to export video")

            # Display cached export result
            if "export_result" in st.session_state.video_cache:
                st.subheader("Exported Video")
                export_format = st.session_state.video_cache.get("export_format", "mp4")
                export_name = st.session_state.video_cache.get("export_name", "video")
                st.download_button(
                    "Download Video",
                    st.session_state.video_cache["export_result"],
                    f"{os.path.splitext(export_name)[0]}_exported.{export_format}",
                    f"video/{export_format}"
                )

        os.unlink(input_path)


# =====================
# UPSCALE VIDEO
# =====================
elif tool == "Upscale Video":
    st.title("Upscale Video")
    st.markdown("Enhance video resolution up to 4K using AI super-resolution")

    from upscaler import UPSCALE_PRESETS, upscale_video, get_video_info, check_upscaler_available

    avail, backends = check_upscaler_available()
    if "realesrgan" in backends:
        st.success("Real-ESRGAN available (best quality)")
    else:
        st.info("Using Lanczos interpolation. Install `realesrgan` + `basicsr` for better quality.")

    up_col1, up_col2 = st.columns([1, 1.5])
    with up_col1:
        up_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi", "mkv"], key="up_file")
        up_preset = st.selectbox(
            "Upscale Preset",
            list(UPSCALE_PRESETS.keys()),
            format_func=lambda x: f"{UPSCALE_PRESETS[x]['name']} - {UPSCALE_PRESETS[x]['description']}",
        )
        preset_info = UPSCALE_PRESETS[up_preset]
        st.caption(f"Scale: {preset_info['scale']}x | Method: {preset_info['method']}")

        up_go = st.button("Upscale", type="primary", use_container_width=True)

    with up_col2:
        if up_file and up_go:
            # Save uploaded file
            up_dir = os.path.join(os.path.dirname(__file__), ".mp")
            os.makedirs(up_dir, exist_ok=True)
            up_path = os.path.join(up_dir, f"upload_{up_file.name}")
            with open(up_path, "wb") as f:
                f.write(up_file.read())

            info = get_video_info(up_path)
            st.caption(f"Input: {info['width']}x{info['height']} | {info['frame_count']} frames | {info['duration']:.1f}s")

            progress = st.progress(0, text="Starting upscale...")
            def up_cb(cur, total, msg):
                progress.progress(min(cur / max(total, 1), 0.99), text=msg)

            try:
                result_path = upscale_video(
                    up_path, scale=preset_info["scale"],
                    method=preset_info["method"], progress_callback=up_cb,
                )
                progress.progress(1.0, text="Complete!")
                st.video(result_path)
                new_info = get_video_info(result_path)
                st.success(f"Upscaled: {info['width']}x{info['height']} → {new_info['width']}x{new_info['height']}")
                with open(result_path, "rb") as rf:
                    st.download_button("Download Upscaled Video", rf.read(),
                                       file_name=f"upscaled_{up_file.name}", mime="video/mp4",
                                       use_container_width=True)
            except Exception as e:
                st.error(f"Upscale failed: {e}")
        elif up_file:
            up_dir = os.path.join(os.path.dirname(__file__), ".mp")
            os.makedirs(up_dir, exist_ok=True)
            up_path = os.path.join(up_dir, f"upload_{up_file.name}")
            with open(up_path, "wb") as f:
                f.write(up_file.read())
            st.video(up_path)
            info = get_video_info(up_path)
            st.caption(f"{info['width']}x{info['height']} | {info['frame_count']} frames | {info['duration']:.1f}s")


# =====================
# VIDEO EDITOR
# =====================
elif tool == "Video Editor":
    st.title("Video Editor")
    st.markdown("Edit videos with AI-powered tools: remove objects, apply filters, and more")

    from video_editor import EDIT_OPERATIONS, STYLE_FILTERS, edit_video_region, apply_style_transfer

    ve_col1, ve_col2 = st.columns([1, 1.5])
    with ve_col1:
        ve_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"], key="ve_file")
        ve_tab_mode = st.radio("Mode", ["Style Filters", "Region Edit"], horizontal=True, key="ve_mode")

        if ve_tab_mode == "Style Filters":
            ve_style = st.selectbox("Filter", STYLE_FILTERS,
                                     format_func=lambda x: x.replace("_", " ").title())
            ve_apply_style = st.button("Apply Filter", type="primary", use_container_width=True)
        else:
            ve_op = st.selectbox("Operation", list(EDIT_OPERATIONS.keys()),
                                  format_func=lambda x: EDIT_OPERATIONS[x])
            st.markdown("**Region (pixels)**")
            rc1, rc2 = st.columns(2)
            with rc1:
                ve_x = st.number_input("X", 0, 2000, 50, key="ve_x")
                ve_w = st.number_input("Width", 10, 2000, 200, key="ve_w")
            with rc2:
                ve_y = st.number_input("Y", 0, 2000, 50, key="ve_y")
                ve_h = st.number_input("Height", 10, 2000, 200, key="ve_h")
            ve_apply_region = st.button("Apply Edit", type="primary", use_container_width=True)

    with ve_col2:
        if ve_file:
            ve_dir = os.path.join(os.path.dirname(__file__), ".mp")
            os.makedirs(ve_dir, exist_ok=True)
            ve_path = os.path.join(ve_dir, f"edit_{ve_file.name}")
            with open(ve_path, "wb") as f:
                f.write(ve_file.read())

            if ve_tab_mode == "Style Filters" and 've_apply_style' in dir() and ve_apply_style:
                prog = st.progress(0, text="Applying filter...")
                def ve_cb(c, t, m):
                    prog.progress(min(c/max(t,1), 0.99), text=m)
                try:
                    result = apply_style_transfer(ve_path, ve_style, ve_cb)
                    prog.progress(1.0, text="Done!")
                    st.video(result)
                    with open(result, "rb") as rf:
                        st.download_button("Download", rf.read(), file_name=f"edited_{ve_file.name}",
                                           mime="video/mp4", use_container_width=True)
                except Exception as e:
                    st.error(f"Failed: {e}")

            elif ve_tab_mode == "Region Edit" and 've_apply_region' in dir() and ve_apply_region:
                prog = st.progress(0, text="Editing region...")
                def ve_cb2(c, t, m):
                    prog.progress(min(c/max(t,1), 0.99), text=m)
                try:
                    result = edit_video_region(ve_path, ve_op, ve_x, ve_y, ve_w, ve_h,
                                               progress_callback=ve_cb2)
                    prog.progress(1.0, text="Done!")
                    st.video(result)
                    with open(result, "rb") as rf:
                        st.download_button("Download", rf.read(), file_name=f"edited_{ve_file.name}",
                                           mime="video/mp4", use_container_width=True)
                except Exception as e:
                    st.error(f"Failed: {e}")
            else:
                st.video(ve_path)


# =====================
# AUDIO STUDIO
# =====================
elif tool == "Audio Studio":
    st.title("Audio Studio")
    st.markdown("Sound effects, voice isolation, audio mixing, and more")

    from audio_studio import (
        SFX_LIBRARY, generate_sfx_procedural, isolate_voice,
        normalize_audio, mix_audio_tracks, add_fade, extract_audio,
    )

    as_mode = st.radio("Tool", ["Sound Effects", "Voice Isolation", "Extract Audio", "Normalize"], horizontal=True)

    if as_mode == "Sound Effects":
        st.markdown("### Generate Sound Effects")
        sfx_col1, sfx_col2 = st.columns([1, 1.5])
        with sfx_col1:
            sfx_type = st.selectbox("Effect", list(SFX_LIBRARY.keys()),
                                     format_func=lambda x: f"{x.replace('_',' ').title()} - {SFX_LIBRARY[x]}")
            sfx_dur = st.slider("Duration (seconds)", 0.5, 10.0, 2.0, 0.5)
            sfx_go = st.button("Generate SFX", type="primary", use_container_width=True)
        with sfx_col2:
            if sfx_go:
                with st.spinner("Generating..."):
                    path = generate_sfx_procedural(sfx_type, sfx_dur)
                    st.audio(path)
                    with open(path, "rb") as f:
                        st.download_button("Download SFX", f.read(),
                                           file_name=f"sfx_{sfx_type}.wav", mime="audio/wav",
                                           use_container_width=True)

    elif as_mode == "Voice Isolation":
        st.markdown("### Separate Vocals from Background")
        iso_file = st.file_uploader("Upload Audio/Video", type=["mp3", "wav", "mp4", "mov"], key="iso_file")
        if iso_file and st.button("Isolate Voice", type="primary"):
            iso_dir = os.path.join(os.path.dirname(__file__), ".mp")
            os.makedirs(iso_dir, exist_ok=True)
            iso_path = os.path.join(iso_dir, f"iso_{iso_file.name}")
            with open(iso_path, "wb") as f:
                f.write(iso_file.read())
            with st.spinner("Separating vocals..."):
                result = isolate_voice(iso_path)
            if "error" not in result:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Vocals**")
                    if os.path.exists(result.get("vocals", "")):
                        st.audio(result["vocals"])
                with c2:
                    st.markdown("**Background**")
                    if os.path.exists(result.get("background", "")):
                        st.audio(result["background"])
                st.caption(f"Method: {result.get('method', 'unknown')}")
            else:
                st.error(result["error"])

    elif as_mode == "Extract Audio":
        st.markdown("### Extract Audio from Video")
        ext_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi", "mkv"], key="ext_file")
        if ext_file and st.button("Extract", type="primary"):
            ext_dir = os.path.join(os.path.dirname(__file__), ".mp")
            os.makedirs(ext_dir, exist_ok=True)
            ext_path = os.path.join(ext_dir, f"ext_{ext_file.name}")
            with open(ext_path, "wb") as f:
                f.write(ext_file.read())
            with st.spinner("Extracting audio..."):
                result = extract_audio(ext_path)
            if result:
                st.audio(result)
                with open(result, "rb") as f:
                    st.download_button("Download Audio", f.read(),
                                       file_name="extracted_audio.wav", mime="audio/wav",
                                       use_container_width=True)
            else:
                st.warning("No audio track found in this video.")

    elif as_mode == "Normalize":
        st.markdown("### Normalize Audio Volume")
        norm_file = st.file_uploader("Upload Audio", type=["mp3", "wav"], key="norm_file")
        norm_db = st.slider("Target dB", -10.0, 0.0, -3.0, 0.5)
        if norm_file and st.button("Normalize", type="primary"):
            norm_dir = os.path.join(os.path.dirname(__file__), ".mp")
            os.makedirs(norm_dir, exist_ok=True)
            norm_path = os.path.join(norm_dir, f"norm_{norm_file.name}")
            with open(norm_path, "wb") as f:
                f.write(norm_file.read())
            result = normalize_audio(norm_path, target_db=norm_db)
            st.audio(result)
            with open(result, "rb") as f:
                st.download_button("Download", f.read(),
                                   file_name=f"normalized_{norm_file.name}", mime="audio/wav",
                                   use_container_width=True)


# =====================
# TRANSCRIBE
# =====================
elif tool == "Transcribe":
    st.title("Transcribe Audio/Video")
    st.markdown("Extract text from video or audio using AI speech recognition (WhisperX).")

    # Check if WhisperX is installed
    whisperx_ok, whisperx_msg = check_whisperx_installed()
    if not whisperx_ok:
        st.error(whisperx_msg)
        st.markdown("""
        **Installation:**
        ```bash
        pip install whisperx torch torchaudio
        ```

        For GPU acceleration (CUDA), install PyTorch with CUDA support:
        ```bash
        pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
        ```
        """)
        st.stop()

    st.info(f"Device: **{get_device().upper()}** | {whisperx_msg}")

    # Input source selection
    input_mode = st.radio("Input Source", ["Upload File", "Record Microphone"], horizontal=True)

    input_path = None
    is_video = False
    source_name = None

    if input_mode == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload video or audio",
            type=["mp4", "mov", "avi", "mkv", "mp3", "wav", "m4a", "flac", "ogg", "webm"],
            key="transcribe_upload"
        )

        if uploaded_file:
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            is_video = file_ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]
            source_name = uploaded_file.name

            if get_cached_video("input") is None:
                input_path = save_uploaded_file(uploaded_file, "input")
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                    tmp.write(get_cached_video("input"))
                    input_path = tmp.name

            file_size_mb = len(get_cached_video("input")) / (1024 * 1024)
            st.success(f"Loaded: **{uploaded_file.name}** ({file_size_mb:.1f} MB)")

    else:
        st.markdown("Click the microphone icon to start recording. Click again to stop.")
        audio_bytes = audio_recorder(
            text="",
            recording_color="#e74c3c",
            neutral_color="#6c757d",
            icon_size="2x",
            pause_threshold=60.0,
        )

        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            source_name = "microphone_recording.wav"

            # Save recorded audio to temp file and cache
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                input_path = tmp.name
            st.session_state.video_cache["input"] = audio_bytes

            file_size_mb = len(audio_bytes) / (1024 * 1024)
            st.success(f"Recorded: **{file_size_mb:.2f} MB** of audio")

    if input_path:
        # Transcription Settings FIRST
        st.subheader("Transcription Settings")

        col1, col2 = st.columns(2)

        with col1:
            model_size = st.selectbox(
                "Model Size",
                MODEL_SIZES,
                index=1,  # "base" as default
                help="tiny=fastest, large-v3=most accurate. 'base' is good balance."
            )

            language = st.selectbox(
                "Language",
                get_language_list(),
                index=0,
                help="Select the language of the audio, or Auto-detect"
            )

        with col2:
            compute_type = st.selectbox(
                "Compute Type",
                COMPUTE_TYPES,
                index=1 if get_device() == "cuda" else 0,  # float16 for GPU, int8 for CPU
                help="int8=fastest, float16=GPU optimized, float32=most accurate"
            )

            batch_size = st.slider(
                "Batch Size",
                min_value=1,
                max_value=32,
                value=16,
                help="Higher=faster but uses more VRAM. Reduce if out of memory."
            )

        col3, col4 = st.columns(2)

        with col3:
            translate = st.checkbox(
                "Translate to English",
                value=False,
                help="Translate non-English audio to English"
            )

        with col4:
            pass  # Reserved for future options

        # Output format selection
        st.subheader("Output Format")
        output_cols = st.columns(5)
        output_formats = []

        with output_cols[0]:
            if st.checkbox("Text (.txt)", value=True):
                output_formats.append("txt")
        with output_cols[1]:
            if st.checkbox("SRT Subtitles", value=False):
                output_formats.append("srt")
        with output_cols[2]:
            if st.checkbox("VTT Subtitles", value=False):
                output_formats.append("vtt")
        with output_cols[3]:
            if st.checkbox("JSON", value=False):
                output_formats.append("json")
        with output_cols[4]:
            if st.checkbox("TSV", value=False):
                output_formats.append("tsv")

        st.markdown("---")

        # Action buttons
        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            # Optional Preview button
            if is_video:
                if st.button("Preview Video", type="secondary"):
                    st.session_state.video_cache["show_preview"] = True

        with btn_col2:
            transcribe_clicked = st.button("Transcribe", type="primary")

        # Show video preview only when requested
        if is_video and st.session_state.video_cache.get("show_preview"):
            st.markdown("---")
            display_video_browser_compatible(input_path, "input", "Video Preview")
            video_info = remover.get_video_info(input_path)
            if video_info:
                display_video_info(video_info)
        elif not is_video:
            # Always show audio player (it's lightweight)
            st.audio(get_cached_video("input"))

        # Transcribe button action
        if transcribe_clicked:
            if not output_formats:
                st.warning("Please select at least one output format.")
            else:
                # Create config
                config = TranscriptionConfig(
                    model_size=model_size,
                    language=get_language_code(language),
                    compute_type=compute_type,
                    batch_size=batch_size,
                    translate_to_english=translate
                )

                transcriber = Transcriber(config)

                # Progress display
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_status(msg):
                    status_text.text(msg)

                with st.spinner("Transcribing..."):
                    update_status("Loading model...")
                    progress_bar.progress(10)

                    result = transcriber.transcribe(
                        input_path,
                        output_formats=output_formats,
                        progress_callback=update_status
                    )

                    progress_bar.progress(100)

                if result["success"]:
                    st.success("Transcription complete!")

                    # Store result in session state
                    st.session_state.video_cache["transcription_text"] = result["text"]
                    st.session_state.video_cache["transcription_segments"] = result["segments"]
                    st.session_state.video_cache["transcription_language"] = result["language"]

                    # Display detected language
                    if result["language"]:
                        st.info(f"Detected language: **{result['language']}**")

                else:
                    st.error(f"Transcription failed: {result['error']}")

        # Display transcription results
        if "transcription_text" in st.session_state.video_cache:
            st.markdown("---")
            st.subheader("Transcription Result")

            text = st.session_state.video_cache["transcription_text"]
            segments = st.session_state.video_cache.get("transcription_segments", [])

            # Text area with transcription
            st.text_area("Transcribed Text", text, height=200)

            # Download buttons
            st.subheader("Download")
            download_cols = st.columns(3)

            with download_cols[0]:
                st.download_button(
                    "Download Text (.txt)",
                    text,
                    f"{os.path.splitext(source_name)[0]}_transcription.txt",
                    "text/plain"
                )

            # Generate and offer SRT download
            if segments:
                transcriber_temp = Transcriber()
                srt_content = transcriber_temp.generate_srt(segments)
                vtt_content = transcriber_temp.generate_vtt(segments)

                with download_cols[1]:
                    st.download_button(
                        "Download SRT",
                        srt_content,
                        f"{os.path.splitext(source_name)[0]}.srt",
                        "text/plain"
                    )

                with download_cols[2]:
                    st.download_button(
                        "Download VTT",
                        vtt_content,
                        f"{os.path.splitext(source_name)[0]}.vtt",
                        "text/plain"
                    )

        os.unlink(input_path)
    else:
        st.info("Upload a file or record from your microphone to transcribe.")


# =====================
# VIEW & PUBLISH
# =====================
elif tool == "View & Publish":
    st.title("View & Publish")
    st.markdown("Preview your video and publish directly to YouTube.")

    # Video upload section
    uploaded_file = st.file_uploader(
        "Upload video to view/publish",
        type=["mp4", "mov", "avi", "mkv"],
        key="publish_upload"
    )

    if uploaded_file:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()

        if get_cached_video("input") is None:
            input_path = save_uploaded_file(uploaded_file, "input")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(get_cached_video("input"))
                input_path = tmp.name

        # Video Preview Section
        display_video_browser_compatible(input_path, "input", "Video Preview")

        video_info = remover.get_video_info(input_path)
        if video_info:
            display_video_info(video_info)

        st.markdown("---")

        # YouTube Upload Section
        st.subheader("Publish to YouTube")

        # Check dependencies
        deps_ok, deps_msg = check_dependencies()
        if not deps_ok:
            st.error(deps_msg)
            st.stop()

        # Check for client secrets
        secrets_ok, secrets_msg = check_client_secrets()
        if not secrets_ok:
            st.warning(secrets_msg)
            st.markdown("""
            **Setup Instructions:**
            1. Go to [Google Cloud Console](https://console.cloud.google.com/)
            2. Create a new project or select an existing one
            3. Enable **YouTube Data API v3**
            4. Go to **Credentials** > **Create Credentials** > **OAuth client ID**
            5. Select **Desktop app** as application type
            6. Download the JSON file and save as `client_secrets.json` in the project folder
            """)
            st.stop()

        # Initialize uploader
        if st.session_state.youtube_uploader is None:
            st.session_state.youtube_uploader = YouTubeUploader()

        uploader = st.session_state.youtube_uploader

        # Authentication status
        if uploader.is_authenticated():
            st.success("Connected to YouTube")

            # Logout option
            if st.button("Disconnect YouTube Account"):
                uploader.logout()
                st.session_state.youtube_uploader = None
                st.rerun()

            st.markdown("---")

            # Video metadata form
            st.subheader("Video Details")

            title = st.text_input(
                "Title",
                value=os.path.splitext(uploaded_file.name)[0],
                max_chars=100
            )

            description = st.text_area(
                "Description",
                placeholder="Enter video description...",
                max_chars=5000,
                height=150
            )

            col1, col2 = st.columns(2)

            with col1:
                category = st.selectbox("Category", get_category_list(), index=7)  # Entertainment

            with col2:
                privacy = st.selectbox(
                    "Privacy",
                    ["Private", "Unlisted", "Public"],
                    index=0,
                    help="Private: Only you can view. Unlisted: Anyone with link. Public: Everyone."
                )

            tags_input = st.text_input(
                "Tags (comma-separated)",
                placeholder="tag1, tag2, tag3"
            )
            tags = [t.strip() for t in tags_input.split(",") if t.strip()] if tags_input else []

            st.markdown("---")

            # Upload button
            if st.button("Upload to YouTube", type="primary"):
                with st.spinner("Uploading to YouTube..."):
                    # Create temp file for upload
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                        tmp.write(get_cached_video("input"))
                        upload_path = tmp.name

                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def update_progress(percent):
                        progress_bar.progress(percent)
                        status_text.text(f"Uploading: {percent}%")

                    result = uploader.upload_video(
                        video_path=upload_path,
                        title=title,
                        description=description,
                        category=category,
                        tags=tags,
                        privacy_status=privacy.lower(),
                        progress_callback=update_progress
                    )

                    os.unlink(upload_path)

                    if result["success"]:
                        progress_bar.progress(100)
                        status_text.text("Upload complete!")
                        st.success("Video uploaded successfully!")
                        st.markdown(f"**Video URL:** [{result['url']}]({result['url']})")
                        st.balloons()
                    else:
                        st.error(f"Upload failed: {result['error']}")

        else:
            # Not authenticated - show auth flow
            st.info("Connect your YouTube account to upload videos.")

            st.markdown("""
            **How it works:**
            1. Click the button below
            2. A browser window will open for Google authorization
            3. Sign in and allow access to your YouTube account
            4. Return here after authorization completes
            """)

            if st.button("Connect YouTube Account", type="primary"):
                with st.spinner("Opening browser for authorization..."):
                    if uploader.authenticate_with_local_server(port=8085):
                        st.success("Successfully connected to YouTube!")
                        st.rerun()
                    else:
                        st.error("Authorization failed. Please try again.")

        os.unlink(input_path)
    else:
        st.info("Upload a video to preview and publish to YouTube.")

# ============================================================
# ReelForge — AI Short Video Generator
# ============================================================
elif tool == "ReelForge":
    st.title("ReelForge")
    st.markdown("Generate AI-powered short videos with synced speech and visuals")

    # Init session state
    if "rf_result" not in st.session_state:
        st.session_state.rf_result = None

    cfg = load_config()
    backend = cfg.get("llm_backend", "llamacpp")
    select_backend(backend)

    # Status check (compact)
    backend_ok, backend_msg = check_backend_status()
    if not backend_ok:
        st.error(f"LLM not ready: {backend_msg}. Go to Settings to configure.")
        st.stop()

    # Main layout: Input on left, Output on right
    input_col, output_col = st.columns([1, 2])

    with input_col:
        st.markdown("### Create Video")

        # Basic settings
        rf_topic = st.text_input("Topic", placeholder="What is your video about?")
        rf_num_images = st.slider("Number of Scenes", 2, 6, 3, help="Each scene = 1 image + 1 spoken sentence")

        # Aspect Ratio selector
        st.markdown("**Video Format**")
        aspect_ratio_labels = {
            "9:16": "Portrait (9:16)",
            "16:9": "Landscape (16:9)",
            "1:1": "Square (1:1)",
            "4:5": "Instagram (4:5)",
        }
        rf_aspect_ratio = st.radio(
            "Aspect Ratio",
            options=list(ASPECT_RATIOS.keys()),
            format_func=lambda x: aspect_ratio_labels.get(x, x),
            horizontal=True,
            label_visibility="collapsed",
        )

        # Collapsible settings
        with st.expander("Style & Effects", expanded=False):
            rf_image_style = st.selectbox("Visual Style", list(IMAGE_STYLE_PRESETS.keys()), index=0)
            rf_subtitle_style = st.selectbox("Text Style", list(SUBTITLE_STYLES.keys()), index=1)
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                rf_ken_burns = st.selectbox("Motion", MOTION_EFFECTS, index=0)
            with col_e2:
                rf_color_filter = st.selectbox("Filter", COLOR_FILTERS, index=0)

        with st.expander("Audio Settings", expanded=False):
            rf_music_enabled = st.checkbox("Add Background Music", value=cfg.get("background_music_enabled", False))
            if rf_music_enabled:
                rf_music_volume = st.slider(
                    "Music Volume",
                    min_value=0.05,
                    max_value=0.5,
                    value=cfg.get("background_music_volume", 0.15),
                    step=0.05,
                    help="Lower values keep music subtle behind narration"
                )
                # Show available music files
                available_music = list_background_music()
                if available_music:
                    music_options = ["Random"] + available_music
                    rf_music_file = st.selectbox("Music Track", music_options, index=0)
                    if rf_music_file == "Random":
                        rf_music_path = None
                    else:
                        rf_music_path = str(get_music_dir() / rf_music_file)
                else:
                    st.info("Add .mp3 or .wav files to the 'music' folder for background music")
                    rf_music_path = None
            else:
                rf_music_volume = 0.15
                rf_music_path = None

        with st.expander("Advanced", expanded=False):
            rf_language = st.text_input("Language", value="English")
            rf_provider = st.selectbox("Image Provider", ["sdxl_turbo", "nanobanana2"], index=0)
            rf_model = None
            if rf_provider == "sdxl_turbo":
                models = list_local_models()
                if models:
                    rf_model = st.selectbox("Model", models)
            rf_image_steps = st.slider("Quality (Steps)", 4, 50, 8)
            rf_image_guidance = st.slider("Prompt Strength", 1.0, 15.0, 2.0)

        st.markdown("---")
        generate_clicked = st.button("Generate Video", type="primary", use_container_width=True)

    with output_col:
        if generate_clicked and rf_topic:
            st.session_state.rf_result = None
            progress_bar = st.progress(0, text="Starting...")

            def on_progress(step, total, msg):
                progress_bar.progress(step / total, text=msg)

            try:
                result = rf_generate_full(
                    topic=rf_topic,
                    language=rf_language,
                    sentence_count=rf_num_images,
                    image_provider=rf_provider,
                    sdxl_model=rf_model,
                    progress_callback=on_progress,
                    image_style=rf_image_style,
                    image_steps=rf_image_steps,
                    image_guidance=rf_image_guidance,
                    subtitle_style=rf_subtitle_style,
                    ken_burns_effect=rf_ken_burns,
                    transition="none",
                    color_filter=rf_color_filter,
                    num_images=rf_num_images,
                    aspect_ratio=rf_aspect_ratio,
                    music_enabled=rf_music_enabled,
                    music_path=rf_music_path,
                    music_volume=rf_music_volume,
                )
                st.session_state.rf_result = result
                progress_bar.progress(1.0, text="Complete!")
            except Exception as e:
                st.error(f"Generation failed: {e}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())

        elif generate_clicked:
            st.warning("Please enter a topic for your video.")

        # Display results
        result = st.session_state.rf_result
        if result:
            # Video first (most important)
            st.markdown("### Generated Video")
            st.video(result["video_path"])

            # Download button
            with open(result["video_path"], "rb") as vf:
                st.download_button(
                    "Download Video",
                    vf.read(),
                    file_name=f"reelforge_{rf_topic.replace(' ', '_')[:20]}.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )

            # Scene breakdown - compact grid layout
            with st.expander("Scene Breakdown", expanded=True):
                scenes = result.get("scenes", [])
                if scenes:
                    # Create columns for scene grid (3 per row)
                    cols_per_row = 3
                    for row_start in range(0, len(scenes), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for col_idx, scene_idx in enumerate(range(row_start, min(row_start + cols_per_row, len(scenes)))):
                            scene = scenes[scene_idx]
                            duration = scene.get("duration", 0)
                            narration = scene.get("narration", scene.get("text", ""))

                            with cols[col_idx]:
                                # Scene thumbnail
                                if scene.get("image_path"):
                                    st.image(scene["image_path"], use_container_width=True)
                                # Scene info
                                st.caption(f"**{scene_idx+1}.** {duration:.1f}s")
                                st.markdown(f'"{narration[:60]}..."' if len(narration) > 60 else f'"{narration}"')

            # Metadata
            with st.expander("Video Metadata"):
                st.text_input("Title", result["title"], disabled=True)
                st.text_area("Description", result["description"], disabled=True, height=80)
                if result.get("total_duration"):
                    st.metric("Total Duration", f"{result['total_duration']:.1f}s")

# ============================================================
# VIDEO GENERATOR PAGE
# ============================================================
elif tool == "Video Generator":
    st.title("Video Generator")
    st.markdown("Generate real videos using Wan 2.1, LTX-Video, or CogVideoX diffusion models")

    from videogen import VideoGenerator, check_system_requirements, VideoGenConfig, get_available_engines
    from presets import (
        STYLE_PRESETS, SHOT_TEMPLATES, LIGHTING_TEMPLATES, CAMERA_MOVEMENTS,
        apply_preset, apply_shot, apply_lighting, apply_camera_movement, build_full_prompt,
    )
    from uuid import uuid4

    # System requirements check
    ok, msg, vram_info = check_system_requirements()

    if not ok:
        st.error(f"System requirements not met: {msg}")
        st.info("Video generation requires a CUDA-capable GPU with at least 6GB VRAM.")
        st.stop()

    # Display GPU info
    st.info(f"GPU: {vram_info['gpu_name']} | VRAM: {vram_info['free_vram']:.1f}GB available")

    # Initialize session state
    if "vg_result" not in st.session_state:
        st.session_state.vg_result = None
    if "vg_generating" not in st.session_state:
        st.session_state.vg_generating = False

    # Two-column layout
    input_col, output_col = st.columns([1, 1.5])

    with input_col:
        # Engine selection
        vram = vram_info["free_vram"]
        engines = get_available_engines()

        # Recommend Wan for most cases, HunyuanVideo for high VRAM
        default_engine_idx = 0  # Wan is recommended for most VRAM levels
        engine_options = ["Wan 2.1 (Recommended)", "LTX-Video (Fast)", "CogVideoX"]
        # Add HunyuanVideo for high VRAM GPUs (24GB+)
        if vram >= 24:
            engine_options.insert(1, "HunyuanVideo (High VRAM)")
        vg_engine = st.radio(
            "Video Engine",
            engine_options,
            index=default_engine_idx,
            horizontal=True,
            help="Wan 2.1: Best quality, 8GB+. HunyuanVideo: Best for 24GB+. LTX-Video: Fast. CogVideoX: 8GB+"
        )
        # Parse engine key
        if "Wan" in vg_engine:
            engine_key = "wan"
        elif "Hunyuan" in vg_engine:
            engine_key = "hunyuan"
        elif "LTX" in vg_engine:
            engine_key = "ltx"
        else:
            engine_key = "cogvideox"

        # Mode selection
        vg_mode = st.radio(
            "Generation Mode",
            ["Text to Video", "Image to Video", "Extend Video"],
            horizontal=True,
            help="Choose how to generate your video"
        )

        # Sample prompts for inspiration
        sample_prompts = {
            "Custom (write your own)": "",
            "Cinematic Nature": "A majestic eagle soaring through a dramatic mountain landscape at golden hour, with snow-capped peaks in the background, volumetric clouds, cinematic camera tracking shot following the bird, ultra detailed feathers catching the warm sunlight, 4K quality",
            "Urban Time-lapse": "Busy city intersection at night transitioning from dusk to night, car light trails forming streams of red and white, neon signs flickering on, pedestrians moving in fast motion, reflections on wet pavement after rain, dynamic timelapse style",
            "Ocean Waves": "Crystal clear turquoise ocean waves gently rolling onto a pristine white sand beach, aerial drone shot slowly descending, palm trees swaying in the breeze, golden sunset light reflecting off the water surface, peaceful and serene atmosphere",
            "Dancing Flames": "Mesmerizing campfire flames dancing in slow motion against a dark night sky, sparks floating upward like tiny stars, warm orange and red colors, close-up macro shot with shallow depth of field, cozy autumn forest setting",
            "Space Journey": "Spaceship traveling through a colorful nebula in deep space, passing by giant gas planets with swirling storms, stars streaking past as the ship accelerates to warp speed, epic sci-fi cinematography, lens flares and particle effects",
            "Underwater World": "Vibrant coral reef teeming with tropical fish of all colors, camera slowly gliding through the underwater paradise, sunbeams piercing through the crystal clear water, a sea turtle gracefully swimming past, National Geographic documentary style",
            "Forest Magic": "Enchanted forest path covered in morning mist, golden sunlight filtering through ancient oak trees, fireflies and magical particles floating in the air, a deer walking peacefully through the scene, fantasy fairy tale atmosphere",
            "Robot Future": "Humanoid robot in a futuristic laboratory slowly awakening, LED lights flickering on across its chrome body, camera orbiting around as it takes its first steps, holographic displays in the background, Blade Runner aesthetic",
            "Coffee Art": "Hot espresso being poured into a ceramic cup in extreme slow motion, creating intricate latte art patterns, steam rising elegantly, macro lens capturing every droplet and swirl, warm cafe lighting, ASMR satisfying visuals",
            "Storm Power": "Dramatic thunderstorm over open plains, lightning bolts striking the ground in the distance, dark ominous clouds rolling across the sky, wind bending tall grass, cinematic wide shot capturing nature's raw power, moody atmosphere"
        }

        selected_sample = st.selectbox(
            "Sample Prompts (optional)",
            list(sample_prompts.keys()),
            index=0,
            help="Select a sample prompt for inspiration or write your own"
        )

        # Get the prompt value
        default_prompt = sample_prompts[selected_sample]

        # Prompt input
        vg_prompt = st.text_area(
            "Describe the video",
            value=default_prompt,
            placeholder="A panda playing guitar in a bamboo forest, cinematic lighting, smooth camera movement...",
            height=120,
            help="Describe what should happen in the video. Be specific about motion, camera movement, and style."
        )

        # Style & Camera presets
        with st.expander("Style, Shot & Camera Presets", expanded=False):
            preset_col1, preset_col2 = st.columns(2)
            with preset_col1:
                vg_style = st.selectbox(
                    "Visual Style",
                    ["None"] + list(STYLE_PRESETS.keys()),
                    format_func=lambda x: STYLE_PRESETS[x]["name"] if x in STYLE_PRESETS else "None",
                    key="vg_style_preset",
                )
                vg_shot = st.selectbox(
                    "Shot Type",
                    ["None"] + list(SHOT_TEMPLATES.keys()),
                    format_func=lambda x: SHOT_TEMPLATES[x]["name"] if x in SHOT_TEMPLATES else "None",
                    key="vg_shot_preset",
                )
            with preset_col2:
                vg_lighting = st.selectbox(
                    "Lighting",
                    ["None"] + list(LIGHTING_TEMPLATES.keys()),
                    format_func=lambda x: LIGHTING_TEMPLATES[x]["name"] if x in LIGHTING_TEMPLATES else "None",
                    key="vg_lighting_preset",
                )
                vg_camera = st.selectbox(
                    "Camera Movement",
                    list(CAMERA_MOVEMENTS.keys()),
                    format_func=lambda x: CAMERA_MOVEMENTS[x]["name"],
                    key="vg_camera_preset",
                )
            # Preview toggle
            vg_preview_mode = st.checkbox("Draft Preview (fast, low-res)", value=False, key="vg_preview")

        # Mode-specific inputs
        vg_image_file = None
        vg_video_file = None

        if vg_mode == "Image to Video":
            vg_image_file = st.file_uploader(
                "Upload image to animate",
                type=["png", "jpg", "jpeg"],
                help="Upload a still image to animate"
            )
            if vg_image_file:
                st.image(vg_image_file, caption="Input image", use_container_width=True)

        elif vg_mode == "Extend Video":
            vg_video_file = st.file_uploader(
                "Upload video to extend",
                type=["mp4", "mov", "avi"],
                help="Upload a video to extend or transform"
            )
            if vg_video_file:
                st.video(vg_video_file)

        # Settings expander
        with st.expander("Advanced Settings", expanded=False):
            # Engine-specific model selection
            if engine_key == "wan":
                # Wan 2.1/2.2 settings
                wan_model_options = ["1.3B (8GB VRAM)", "14B (16GB VRAM)", "2.2-14B (24GB VRAM)"]
                wan_model_keys = ["1.3b", "14b", "2.2-14b"]
                if vram >= 24:
                    default_wan_idx = 2
                elif vram >= 16:
                    default_wan_idx = 1
                else:
                    default_wan_idx = 0

                vg_wan_model_choice = st.selectbox(
                    "Wan Model",
                    wan_model_options,
                    index=default_wan_idx,
                    help="1.3B: Fast, 8GB VRAM. 14B: Best quality, needs 16GB+. 2.2-14B: Latest version."
                )
                vg_wan_model = wan_model_keys[wan_model_options.index(vg_wan_model_choice)]

                # Resolution for Wan
                wan_res_options = ["480p (Fast)", "720p (Better Quality)"]
                wan_res_keys = ["480p", "720p"]
                default_res_idx = 1 if vram >= 16 else 0
                vg_wan_resolution_choice = st.selectbox(
                    "Resolution",
                    wan_res_options,
                    index=default_res_idx,
                    help="720p requires more VRAM but produces sharper video"
                )
                vg_wan_resolution = wan_res_keys[wan_res_options.index(vg_wan_resolution_choice)]

                # Frame count for Wan
                vg_num_frames = st.select_slider(
                    "Video Length",
                    options=[49, 81, 121],
                    value=81,
                    format_func=lambda x: f"{x} frames (~{x/24:.1f}s at 24fps)",
                    help="Wan supports up to 121 frames"
                )

                # Guidance scale
                vg_guidance = st.slider(
                    "Prompt Strength",
                    min_value=1.0,
                    max_value=10.0,
                    value=5.0,
                    step=0.5,
                    help="Higher values follow the prompt more closely"
                )

                vg_steps = st.slider(
                    "Inference Steps",
                    min_value=20,
                    max_value=50,
                    value=30,
                    help="More steps = better quality but slower"
                )

                vg_model_variant = None
                vg_use_quant = False
                vg_ltx_model = None

            elif engine_key == "ltx":
                ltx_model_options = ["Base (10GB)", "Distilled - Fast (10GB)", "0.9.7 Dev (16GB)", "0.9.8 13B (24GB)"]
                ltx_model_keys = ["base", "distilled", "0.9.7", "0.9.8"]
                if vram >= 24:
                    default_ltx_idx = 3
                elif vram >= 16:
                    default_ltx_idx = 2
                else:
                    default_ltx_idx = 1

                vg_ltx_model_choice = st.selectbox(
                    "LTX Model",
                    ltx_model_options,
                    index=default_ltx_idx,
                    help="Larger models produce better quality. Distilled is fastest."
                )
                vg_ltx_model = ltx_model_keys[ltx_model_options.index(vg_ltx_model_choice)]

                # Frame count for LTX
                vg_num_frames = st.select_slider(
                    "Video Length",
                    options=[81, 121, 161],
                    value=161,
                    format_func=lambda x: f"{x} frames (~{x/24:.1f}s at 24fps)",
                    help="LTX supports up to 161 frames at 24-30 FPS"
                )

                # Guidance scale - different for distilled
                is_distilled = vg_ltx_model == "distilled"
                vg_guidance = st.slider(
                    "Prompt Strength",
                    min_value=1.0,
                    max_value=10.0,
                    value=1.0 if is_distilled else 5.0,
                    step=0.5,
                    help="Distilled models work best with guidance=1.0"
                )

                # Steps - fewer for distilled
                vg_steps = st.slider(
                    "Inference Steps",
                    min_value=4 if is_distilled else 20,
                    max_value=10 if is_distilled else 50,
                    value=8 if is_distilled else 50,
                    help="Distilled models need only 4-10 steps"
                )

                vg_model_variant = None
                vg_use_quant = False
                vg_hunyuan_model = None
                vg_hunyuan_resolution = None

            elif engine_key == "hunyuan":
                # HunyuanVideo settings
                hunyuan_model_options = ["1.0 (24GB VRAM)", "1.5 (28GB VRAM)"]
                hunyuan_model_keys = ["1.0", "1.5"]
                default_hunyuan_idx = 1 if vram >= 28 else 0

                vg_hunyuan_model_choice = st.selectbox(
                    "HunyuanVideo Model",
                    hunyuan_model_options,
                    index=default_hunyuan_idx,
                    help="1.5 is newer with better quality but needs more VRAM"
                )
                vg_hunyuan_model = hunyuan_model_keys[hunyuan_model_options.index(vg_hunyuan_model_choice)]

                # Resolution for HunyuanVideo
                hunyuan_res_options = ["540p (Fast)", "720p (Balanced)", "1080p (Best Quality)"]
                hunyuan_res_keys = ["540p", "720p", "1080p"]
                if vram >= 48:
                    default_hunyuan_res_idx = 2
                elif vram >= 32:
                    default_hunyuan_res_idx = 1
                else:
                    default_hunyuan_res_idx = 0

                vg_hunyuan_resolution_choice = st.selectbox(
                    "Resolution",
                    hunyuan_res_options,
                    index=default_hunyuan_res_idx,
                    help="Higher resolution needs more VRAM"
                )
                vg_hunyuan_resolution = hunyuan_res_keys[hunyuan_res_options.index(vg_hunyuan_resolution_choice)]

                # Frame count for HunyuanVideo
                vg_num_frames = st.select_slider(
                    "Video Length",
                    options=[49, 81, 129],
                    value=81,
                    format_func=lambda x: f"{x} frames (~{x/24:.1f}s at 24fps)",
                    help="HunyuanVideo supports up to 129 frames"
                )

                # Guidance scale
                vg_guidance = st.slider(
                    "Prompt Strength",
                    min_value=1.0,
                    max_value=10.0,
                    value=6.0,
                    step=0.5,
                    help="Higher values follow the prompt more closely"
                )

                vg_steps = st.slider(
                    "Inference Steps",
                    min_value=20,
                    max_value=50,
                    value=30,
                    help="More steps = better quality but slower"
                )

                vg_model_variant = None
                vg_use_quant = False
                vg_ltx_model = None
                vg_wan_model = None
                vg_wan_resolution = None

            else:
                # CogVideoX settings
                model_options = ["2B (8GB VRAM)", "5B (16GB+ VRAM)"]
                default_model_idx = 0 if vram < 16 else 1
                vg_model_choice = st.selectbox(
                    "CogVideoX Model",
                    model_options,
                    index=default_model_idx,
                    help="Larger models produce better quality but need more VRAM"
                )
                vg_model_variant = "2b" if "2B" in vg_model_choice else "5b"

                # Frame count for CogVideoX
                vg_num_frames = st.select_slider(
                    "Video Length",
                    options=[49, 81],
                    value=49,
                    format_func=lambda x: f"{x} frames (~{x/8:.0f}s at 8fps)",
                    help="More frames = longer video but slower generation"
                )

                # Guidance scale
                vg_guidance = st.slider(
                    "Prompt Strength",
                    min_value=1.0,
                    max_value=15.0,
                    value=6.0,
                    step=0.5,
                    help="Higher values follow the prompt more closely"
                )

                vg_steps = 50

                # Quantization
                vg_use_quant = st.checkbox(
                    "Use INT8 Quantization (lower VRAM)",
                    value=vram < 16,
                    help="Reduces VRAM usage with minimal quality loss"
                )

                vg_ltx_model = None

            # Common advanced settings
            st.markdown("---")
            st.markdown("**Common Settings**")

            # Negative prompt
            vg_negative_prompt = st.text_input(
                "Negative Prompt (what to avoid)",
                value="low quality, blurry, distorted, disfigured, watermark",
                help="Things to avoid in the video"
            )

            # FPS slider
            vg_fps = st.slider(
                "Output FPS",
                min_value=8,
                max_value=30,
                value=24 if engine_key != "cogvideox" else 8,
                help="Frames per second for the output video"
            )

            # CPU offload toggle
            vg_cpu_offload = st.checkbox(
                "Enable CPU Offload",
                value=vram < 24,
                help="Offload model parts to CPU to save VRAM (slower but uses less GPU memory)"
            )

            # Seed (common)
            vg_use_seed = st.checkbox("Use fixed seed (reproducible results)")
            vg_seed = None
            if vg_use_seed:
                vg_seed = st.number_input("Seed", min_value=0, max_value=2**32-1, value=42)

        # Generate button
        vg_generate_btn = st.button(
            "Generate Video",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.vg_generating
        )

    with output_col:
        if vg_generate_btn and vg_prompt:
            st.session_state.vg_generating = True
            st.session_state.vg_result = None

            # Show what's being generated
            engine_names = {"wan": "Wan 2.1", "hunyuan": "HunyuanVideo", "ltx": "LTX-Video", "cogvideox": "CogVideoX"}
            engine_name = engine_names.get(engine_key, "Video Model")
            st.info(f"Generating {vg_num_frames} frames using **{engine_name}**...")

            # Progress indicators
            progress_bar = st.progress(0, text="Initializing...")
            status_text = st.empty()

            def on_progress(step, total, msg):
                progress_bar.progress(min(step / total, 1.0), text=msg)
                status_text.text(f"Step {step}/{total}: {msg}")

            # Configure generator based on selected engine
            if engine_key == "wan":
                config = VideoGenConfig(
                    engine="wan",
                    wan_model=vg_wan_model,
                    wan_resolution=vg_wan_resolution,
                    num_frames=vg_num_frames,
                    num_inference_steps=vg_steps,
                    guidance_scale=vg_guidance,
                    fps=vg_fps,
                    seed=vg_seed,
                    enable_cpu_offload=vg_cpu_offload,
                )
            elif engine_key == "ltx":
                config = VideoGenConfig(
                    engine="ltx",
                    ltx_model=vg_ltx_model,
                    num_frames=vg_num_frames,
                    num_inference_steps=vg_steps,
                    guidance_scale=vg_guidance,
                    width=768 if vram >= 16 else 704,
                    height=512,
                    fps=vg_fps,
                    seed=vg_seed,
                    enable_cpu_offload=vg_cpu_offload,
                )
            elif engine_key == "hunyuan":
                config = VideoGenConfig(
                    engine="hunyuan",
                    hunyuan_model=vg_hunyuan_model,
                    hunyuan_resolution=vg_hunyuan_resolution,
                    num_frames=vg_num_frames,
                    num_inference_steps=vg_steps,
                    guidance_scale=vg_guidance,
                    fps=vg_fps,
                    seed=vg_seed,
                    enable_cpu_offload=vg_cpu_offload,
                )
            else:
                config = VideoGenConfig(
                    engine="cogvideox",
                    model_variant=vg_model_variant,
                    num_frames=vg_num_frames,
                    num_inference_steps=vg_steps,
                    guidance_scale=vg_guidance,
                    quantization="int8" if vg_use_quant else "none",
                    fps=vg_fps,
                    seed=vg_seed,
                    enable_cpu_offload=vg_cpu_offload,
                )

            # Apply presets to prompt
            _final_prompt = vg_prompt
            _final_neg = vg_negative_prompt
            _style_key = vg_style if vg_style != "None" else None
            _shot_key = vg_shot if vg_shot != "None" else None
            _light_key = vg_lighting if vg_lighting != "None" else None
            _cam_key = vg_camera if vg_camera != "static" else None

            if any([_style_key, _shot_key, _light_key, _cam_key]):
                _final_prompt, _neg_add, _g_override = build_full_prompt(
                    vg_prompt, style=_style_key, shot=_shot_key,
                    lighting=_light_key, camera=_cam_key,
                )
                if _neg_add:
                    _final_neg = f"{vg_negative_prompt}, {_neg_add}" if vg_negative_prompt else _neg_add

            # Apply preview mode (reduced settings)
            if vg_preview_mode:
                config.num_frames = 17
                config.num_inference_steps = max(6, config.num_inference_steps // 4)
                config.guidance_scale = min(config.guidance_scale, 4.0)

            generator = VideoGenerator(config)
            result = None

            try:
                if vg_mode == "Text to Video":
                    on_progress(0, 5, "Loading text-to-video pipeline...")
                    result = generator.generate_text2video(
                        _final_prompt,
                        negative_prompt=_final_neg,
                        progress_callback=on_progress
                    )

                elif vg_mode == "Image to Video" and vg_image_file:
                    on_progress(0, 5, "Processing uploaded image...")
                    # Save uploaded image
                    img_dir = os.path.join(os.path.dirname(__file__), ".mp")
                    os.makedirs(img_dir, exist_ok=True)
                    img_path = os.path.join(img_dir, f"upload_{uuid4()}.png")
                    with open(img_path, "wb") as f:
                        f.write(vg_image_file.read())

                    result = generator.generate_image2video(
                        img_path,
                        vg_prompt,
                        progress_callback=on_progress
                    )

                elif vg_mode == "Extend Video" and vg_video_file:
                    on_progress(0, 5, "Processing uploaded video...")
                    # Save uploaded video
                    vid_dir = os.path.join(os.path.dirname(__file__), ".mp")
                    os.makedirs(vid_dir, exist_ok=True)
                    vid_path = os.path.join(vid_dir, f"upload_{uuid4()}.mp4")
                    with open(vid_path, "wb") as f:
                        f.write(vg_video_file.read())

                    result = generator.generate_video2video(
                        vid_path,
                        vg_prompt,
                        progress_callback=on_progress
                    )
                else:
                    st.warning("Please provide all required inputs")

                if result:
                    st.session_state.vg_result = result
                    progress_bar.progress(1.0, text="Complete!")
                    status_text.empty()
                    st.success("Video generated successfully!")
                    st.balloons()

            except Exception as e:
                st.error(f"Generation failed: {e}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())

            finally:
                generator.unload()
                st.session_state.vg_generating = False

        elif vg_generate_btn and not vg_prompt:
            st.warning("Please enter a prompt describing the video")

        # Display result
        if st.session_state.vg_result and os.path.exists(st.session_state.vg_result):
            st.subheader("Generated Video")
            st.video(st.session_state.vg_result)

            # Download button
            with open(st.session_state.vg_result, "rb") as f:
                video_bytes = f.read()
            st.download_button(
                "Download Video",
                video_bytes,
                file_name="generated_video.mp4",
                mime="video/mp4",
                use_container_width=True
            )

            # Show details
            with st.expander("Generation Details"):
                if engine_key == "wan":
                    st.json({
                        "engine": "Wan 2.1/2.2",
                        "mode": vg_mode,
                        "model": vg_wan_model,
                        "resolution": vg_wan_resolution,
                        "frames": vg_num_frames,
                        "guidance_scale": vg_guidance,
                        "seed": vg_seed,
                    })
                elif engine_key == "ltx":
                    st.json({
                        "engine": "LTX-Video",
                        "mode": vg_mode,
                        "model": vg_ltx_model,
                        "frames": vg_num_frames,
                        "guidance_scale": vg_guidance,
                        "seed": vg_seed,
                    })
                else:
                    st.json({
                        "engine": "CogVideoX",
                        "mode": vg_mode,
                        "model": vg_model_variant,
                        "frames": vg_num_frames,
                        "guidance_scale": vg_guidance,
                        "quantization": "INT8" if vg_use_quant else "None",
                        "seed": vg_seed,
                    })


# ============================================================
# STORY MODE PAGE
# ============================================================
elif tool == "Story Mode":
    st.title("Story Mode")
    st.markdown("Create multi-scene AI movies with narration, music, and transitions")

    from videogen import (
        VideoGenerator, check_system_requirements, VideoGenConfig,
        get_available_engines, concatenate_videos, add_audio_to_video,
        videogen_logger
    )
    from reelforge import (
        load_config, save_config, select_backend, check_backend_status,
        rf_generate_explainer_script, rf_clean_text_for_tts,
        rf_concatenate_audio_files, list_background_music, get_music_dir,
        ASPECT_RATIOS,
    )
    from mpv2.llm_provider import select_model, generate_text
    from mpv2.classes.TtsFactory import get_tts_instance, get_voices_for_engine
    from uuid import uuid4
    import json, time, traceback

    cfg = load_config()
    backend = cfg.get("llm_backend", "llamacpp")
    select_backend(backend)
    if backend == "ollama":
        select_model(cfg.get("ollama_model") or "llama3.2:3b")
    else:
        gguf_model = cfg.get("gguf_model", "")
        if gguf_model:
            select_model(gguf_model)

    # LLM readiness check
    _llm_ok, _llm_msg = check_backend_status()
    if not _llm_ok:
        st.warning(f"LLM not ready: {_llm_msg}. AI script generation will be unavailable. Configure in Settings.")

    # ---- Session State ----
    if "story_scenes" not in st.session_state:
        st.session_state.story_scenes = []
    if "story_result" not in st.session_state:
        st.session_state.story_result = None
    if "story_generated_videos" not in st.session_state:
        st.session_state.story_generated_videos = {}
    if "story_generation_running" not in st.session_state:
        st.session_state.story_generation_running = False
    if "story_visual_identity" not in st.session_state:
        st.session_state.story_visual_identity = ""

    # ---- System Check ----
    ok, msg, vram_info = check_system_requirements()
    if not ok:
        st.error(f"GPU not available: {msg}")
        st.info("Story Mode requires a CUDA GPU. You can still plan your story and generate narration.")

    # ---- Helper: AI Script Generator ----
    def generate_story_script(concept, num_scenes, genre, mood):
        """Use LLM to generate a structured story with scenes + visual identity anchor."""
        backend = cfg.get("llm_backend", "llamacpp")
        select_backend(backend)

        prompt = f"""Create a {num_scenes}-scene cinematic story for a short film.

Concept: {concept}
Genre: {genre}
Mood: {mood}

You MUST return a JSON object with two keys:

1. "visual_identity": A single paragraph describing the CONSISTENT visual elements across ALL scenes. Include:
   - Main subject/character appearance (specific colors, shapes, features that stay the same)
   - Art style and color palette (e.g., "warm golden tones, soft lighting")
   - Camera style (e.g., "wide cinematic shots, shallow depth of field")
   This paragraph will be appended to EVERY scene's prompt to maintain visual consistency.

2. "scenes": A JSON array where each scene has:
   - "title": Short scene title (3-5 words)
   - "visual": Detailed cinematic description for AI video generation. ALWAYS reference the same subjects/characters consistently using the SAME descriptive words.
   - "narration": Voiceover text (1-3 sentences)
   - "duration": Seconds (3-8)

Return ONLY this JSON:
{{
  "visual_identity": "Consistent visual description...",
  "scenes": [
    {{"title": "...", "visual": "...", "narration": "...", "duration": 5}},
    ...
  ]
}}

Rules:
- The visual_identity MUST describe the main subject so it looks the SAME in every scene
- Each scene's visual description should reference the subject with IDENTICAL wording
- Visuals must be CINEMATIC (camera angles, lighting, motion)
- Narration tells a cohesive story across scenes
- No meta-commentary, just the JSON"""

        response = generate_text(prompt)
        response = response.replace("```json", "").replace("```", "").strip()

        import re

        # Parse response - try as object with visual_identity first
        parsed = None
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            # Try extracting JSON object
            match = re.search(r'\{{.*\}}', response, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        # Extract visual_identity and scenes
        visual_identity = ""
        scene_list = []

        if isinstance(parsed, dict):
            visual_identity = parsed.get("visual_identity", "")
            scene_list = parsed.get("scenes", [])
        elif isinstance(parsed, list):
            # Fallback: old format without visual_identity
            scene_list = parsed

        if not scene_list:
            # Try extracting just the array
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                try:
                    scene_list = json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        if not scene_list:
            scene_list = [{"title": f"Scene {i+1}", "visual": concept, "narration": f"Scene {i+1} of the story.", "duration": 5} for i in range(num_scenes)]

        # Attach visual_identity to result
        result_scenes = scene_list[:num_scenes]
        return result_scenes, visual_identity

    # ================================================================
    # LAYOUT: Three phases - Plan | Customize | Generate
    # ================================================================

    # ---- Phase 1: Story Planning ----
    st.markdown("---")
    plan_col, preview_col = st.columns([1, 1.5])

    with plan_col:
        st.markdown("### 1. Plan Your Story")

        story_concept = st.text_area(
            "Story Concept",
            placeholder="Describe your movie idea... e.g., 'A lone astronaut discovers alien life on Mars during a sandstorm'",
            height=80,
        )

        col_a, col_b = st.columns(2)
        with col_a:
            story_num_scenes = st.slider("Number of Scenes", 2, 8, 4)
            story_genre = st.selectbox("Genre", [
                "Cinematic", "Documentary", "Fantasy", "Sci-Fi",
                "Horror", "Romance", "Action", "Nature",
                "Abstract Art", "Noir", "Comedy", "Mystery"
            ])
        with col_b:
            story_mood = st.selectbox("Mood", [
                "Epic", "Calm", "Tense", "Mysterious",
                "Joyful", "Melancholic", "Energetic", "Dreamy",
                "Dark", "Hopeful", "Nostalgic", "Surreal"
            ])
            story_language = st.text_input("Language", value="English")

        # AI Generate button
        col_gen, col_clear = st.columns([2, 1])
        with col_gen:
            ai_generate = st.button("AI Generate Script", type="primary", use_container_width=True)
        with col_clear:
            clear_scenes = st.button("Clear All", use_container_width=True)

        if clear_scenes:
            st.session_state.story_scenes = []
            st.session_state.story_result = None
            st.session_state.story_generated_videos = {}
            st.session_state.story_visual_identity = ""
            st.rerun()

        if ai_generate and story_concept:
            backend_ok, backend_msg = check_backend_status()
            if not backend_ok:
                st.error(f"LLM not ready: {backend_msg}. Configure in Settings.")
            else:
                with st.spinner("AI is writing your story..."):
                    try:
                        scenes, visual_identity = generate_story_script(story_concept, story_num_scenes, story_genre, story_mood)
                        st.session_state.story_scenes = []
                        for s in scenes:
                            st.session_state.story_scenes.append({
                                "title": s.get("title", "Untitled"),
                                "visual": s.get("visual", ""),
                                "narration": s.get("narration", ""),
                                "duration": s.get("duration", 5),
                                "image": None,
                            })
                        st.session_state.story_visual_identity = visual_identity
                        st.session_state.story_result = None
                        st.session_state.story_generated_videos = {}
                        st.rerun()
                    except Exception as e:
                        st.error(f"Script generation failed: {e}")
        elif ai_generate:
            st.warning("Enter a story concept first.")

        # Manual add scene
        st.markdown("---")
        if st.button("+ Add Blank Scene", use_container_width=True):
            st.session_state.story_scenes.append({
                "title": f"Scene {len(st.session_state.story_scenes) + 1}",
                "visual": "",
                "narration": "",
                "duration": 5,
                "image": None,
            })
            st.rerun()

    # ---- Scene Preview / Editor (right column) ----
    with preview_col:
        scenes = st.session_state.story_scenes

        if not scenes:
            st.markdown("### Your Storyboard")
            st.info("Use **AI Generate Script** to create scenes from your concept, or **+ Add Blank Scene** to build manually.")
        else:
            st.markdown(f"### Storyboard ({len(scenes)} scenes)")

            # Scene timeline overview bar
            total_dur = sum(s.get("duration", 5) for s in scenes)
            timeline_parts = []
            colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F"]
            for i, s in enumerate(scenes):
                pct = (s.get("duration", 5) / max(total_dur, 1)) * 100
                c = colors[i % len(colors)]
                timeline_parts.append(
                    f'<div style="width:{pct}%;background:{c};height:8px;display:inline-block;border-radius:2px;" '
                    f'title="Scene {i+1}: {s["title"]} ({s.get("duration",5)}s)"></div>'
                )
            st.markdown(
                f'<div style="display:flex;gap:2px;margin-bottom:4px;">{"".join(timeline_parts)}</div>'
                f'<p style="text-align:right;font-size:12px;color:#888;">Total: {total_dur:.0f}s</p>',
                unsafe_allow_html=True
            )

    # ---- Visual Identity display ----
    if scenes and st.session_state.get("story_visual_identity"):
        with st.expander("Visual Identity Anchor (auto-generated)", expanded=False):
            st.markdown(f"*{st.session_state.story_visual_identity}*")
            st.caption("This description is injected into every scene prompt when Prompt Anchoring is enabled. Edit it in the Scene Continuity settings below.")

    # ---- Phase 2: Scene Editor ----
    if scenes:
        st.markdown("---")
        st.markdown("### 2. Edit Scenes")

        scenes_to_remove = []
        for i, scene in enumerate(scenes):
            color = colors[i % len(colors)]
            with st.expander(f"Scene {i+1}: {scene['title']}", expanded=(i == 0)):
                col1, col2 = st.columns([2, 1])

                with col1:
                    new_title = st.text_input("Title", scene["title"], key=f"story_title_{i}")
                    new_visual = st.text_area(
                        "Visual Prompt (for video generation)",
                        scene["visual"],
                        height=80,
                        key=f"story_visual_{i}",
                        help="Describe exactly what the AI should generate as video"
                    )
                    new_narration = st.text_area(
                        "Narration (voiceover text)",
                        scene["narration"],
                        height=60,
                        key=f"story_narration_{i}",
                        help="This text will be spoken by the TTS voice"
                    )

                with col2:
                    new_duration = st.slider(
                        "Duration (seconds)", 2, 12, int(scene.get("duration", 5)),
                        key=f"story_dur_{i}"
                    )

                    # Reference image upload
                    uploaded_img = st.file_uploader(
                        "Reference Image (optional)",
                        type=["png", "jpg", "jpeg"],
                        key=f"story_img_{i}",
                        help="Upload an image for Image-to-Video generation"
                    )
                    if uploaded_img:
                        img_path = os.path.join(tempfile.gettempdir(), f"story_ref_{i}_{uploaded_img.name}")
                        with open(img_path, "wb") as f:
                            f.write(uploaded_img.read())
                        scene["image"] = img_path
                        st.image(img_path, width=150)
                    elif scene.get("image") and os.path.exists(scene["image"]):
                        st.image(scene["image"], width=150)

                    # Show generated video preview if exists
                    if i in st.session_state.story_generated_videos:
                        vid_path = st.session_state.story_generated_videos[i]
                        if os.path.exists(vid_path):
                            st.video(vid_path)
                            st.caption("Generated")

                # Action buttons row
                btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
                with btn_col1:
                    if i > 0 and st.button("Move Up", key=f"story_up_{i}", use_container_width=True):
                        scenes[i], scenes[i-1] = scenes[i-1], scenes[i]
                        st.rerun()
                with btn_col2:
                    if i < len(scenes) - 1 and st.button("Move Down", key=f"story_down_{i}", use_container_width=True):
                        scenes[i], scenes[i+1] = scenes[i+1], scenes[i]
                        st.rerun()
                with btn_col3:
                    if st.button("Duplicate", key=f"story_dup_{i}", use_container_width=True):
                        new_scene = dict(scene)
                        new_scene["title"] = scene["title"] + " (copy)"
                        scenes.insert(i + 1, new_scene)
                        st.rerun()
                with btn_col4:
                    if st.button("Remove", key=f"story_rm_{i}", type="secondary", use_container_width=True):
                        scenes_to_remove.append(i)

                # Update scene data from inputs
                scene["title"] = new_title
                scene["visual"] = new_visual
                scene["narration"] = new_narration
                scene["duration"] = new_duration

        # Process removals
        if scenes_to_remove:
            for idx in sorted(scenes_to_remove, reverse=True):
                scenes.pop(idx)
            st.rerun()

        # ---- Phase 3: Generation Settings & Output ----
        st.markdown("---")
        st.markdown("### 3. Generate Movie")

        settings_col, output_col = st.columns([1, 1.5])

        with settings_col:
            # Video Engine
            with st.expander("Video Engine", expanded=True):
                available_engines = []
                engine_help = {}
                if ok:
                    free_vram = vram_info.get("free_vram", 0)
                    available_engines = ["Wan 2.1 (Recommended)", "LTX-Video (Fast)", "CogVideoX"]
                    engine_help = {
                        "Wan 2.1 (Recommended)": f"Best quality | {free_vram:.0f}GB available",
                        "LTX-Video (Fast)": "Fastest generation",
                        "CogVideoX": "Good quality, versatile",
                    }
                    if free_vram >= 24:
                        available_engines.append("HunyuanVideo (HD)")

                if not available_engines:
                    st.warning("No GPU detected. Video generation unavailable.")
                    available_engines = ["Wan 2.1 (Recommended)"]

                story_engine = st.radio(
                    "Engine",
                    available_engines,
                    horizontal=True,
                    label_visibility="collapsed",
                )

                # Engine-specific settings
                if "Wan" in story_engine:
                    sc1, sc2 = st.columns(2)
                    with sc1:
                        story_wan_model = st.selectbox("Model", ["1.3b", "14b"], index=0, key="story_wan_model")
                    with sc2:
                        story_wan_res = st.selectbox("Resolution", ["480p", "720p"], index=0, key="story_wan_res")
                elif "LTX" in story_engine:
                    story_ltx_model = st.selectbox("Model", ["base", "distilled"], index=0, key="story_ltx_model")
                elif "CogVideoX" in story_engine:
                    story_cog_model = st.selectbox("Model", ["2b", "5b"], index=0, key="story_cog_model")

                story_frames = st.select_slider(
                    "Frames per Scene",
                    options=[33, 49, 81],
                    value=49,
                    format_func=lambda x: f"{x} frames (~{x/24:.1f}s at 24fps)",
                    key="story_frames",
                )
                story_guidance = st.slider("Prompt Strength", 1.0, 10.0, 5.0, 0.5, key="story_guidance")
                story_steps = st.slider("Inference Steps", 10, 50, 30, key="story_steps")

            # Audio Settings
            with st.expander("Narration & Audio", expanded=True):
                story_enable_narration = st.checkbox("Enable Voiceover Narration", value=True, key="story_narration_on")
                if story_enable_narration:
                    tts_engine = cfg.get("tts_engine", "kitten")
                    available_voices = get_voices_for_engine(tts_engine)
                    if available_voices:
                        story_voice = st.selectbox("Voice", available_voices, key="story_voice")
                    else:
                        story_voice = cfg.get("tts_voice", "Jasper")
                        st.caption(f"Using default voice: {story_voice}")

                story_enable_music = st.checkbox("Background Music", value=False, key="story_music_on")
                if story_enable_music:
                    story_music_vol = st.slider("Music Volume", 0.05, 0.5, 0.12, 0.05, key="story_music_vol")
                    available_music = list_background_music()
                    if available_music:
                        music_options = ["Random"] + available_music
                        story_music_track = st.selectbox("Track", music_options, key="story_music_track")
                    else:
                        st.caption("Add .mp3/.wav files to the `music/` folder")
                        story_music_track = None

            # Scene Continuity
            with st.expander("Scene Continuity", expanded=True):
                st.caption("Keep subjects, characters, and style consistent across scenes")

                story_continuity_mode = st.radio(
                    "Continuity Method",
                    ["None", "Prompt Anchoring", "Scene Chaining", "Both"],
                    horizontal=True,
                    key="story_continuity",
                    help=(
                        "**None**: Each scene generated independently.\n\n"
                        "**Prompt Anchoring**: A visual identity description (subject appearance, style, palette) "
                        "is injected into every scene's prompt so the AI generates consistent-looking subjects.\n\n"
                        "**Scene Chaining**: The last frame of each scene is used as the input image for the "
                        "next scene (Image-to-Video), creating visual flow between scenes.\n\n"
                        "**Both**: Combines prompt anchoring + scene chaining for maximum consistency."
                    ),
                )

                use_prompt_anchor = story_continuity_mode in ("Prompt Anchoring", "Both")
                use_scene_chain = story_continuity_mode in ("Scene Chaining", "Both")

                # Visual identity anchor (editable)
                if use_prompt_anchor:
                    default_anchor = st.session_state.get("story_visual_identity", "")
                    story_visual_anchor = st.text_area(
                        "Visual Identity Anchor",
                        value=default_anchor,
                        height=80,
                        key="story_anchor_text",
                        help=(
                            "This description is appended to EVERY scene's prompt to maintain "
                            "visual consistency. Describe your main subject's exact appearance, "
                            "the color palette, art style, and camera style. "
                            "AI Generate Script fills this automatically, but you can edit it."
                        ),
                        placeholder="e.g., A golden retriever with a red collar, warm sunset lighting, shallow depth of field, cinematic 35mm film look"
                    )
                    if not story_visual_anchor.strip():
                        st.info("Tip: Use **AI Generate Script** to auto-generate this, or write your own description of the consistent visual elements.")

                if use_scene_chain:
                    story_chain_strength = st.slider(
                        "Chaining Strength",
                        0.3, 0.9, 0.6, 0.1,
                        key="story_chain_str",
                        help=(
                            "How much the previous scene's last frame influences the next scene. "
                            "Lower = more creative freedom, Higher = stronger visual continuity. "
                            "0.5-0.7 is usually a good balance."
                        ),
                    )
                    st.caption("Scene 1 uses Text-to-Video. Scenes 2+ use last frame of previous scene as Image-to-Video input.")

                    # Warn about Wan 1.3B I2V limitation
                    if "Wan" in story_engine and story_wan_model == "1.3b":
                        st.warning(
                            "Wan 1.3B has no I2V model (all Wan I2V models are 14B). "
                            "Scene chaining will use **prompt-based continuity** instead of I2V, "
                            "keeping your 1.3B model. For true I2V chaining, use the 14B model (needs 16GB+ VRAM)."
                        )

                if story_continuity_mode == "None":
                    st.info("Each scene will be generated independently with no visual linking.")

                # Shared seed option
                story_use_shared_seed = st.checkbox(
                    "Shared Seed (same random seed for all scenes)",
                    value=False,
                    key="story_shared_seed",
                    help="Using the same seed across scenes produces similar visual patterns and textures.",
                )
                if story_use_shared_seed:
                    import random as _rng
                    story_seed_val = st.number_input("Seed", 0, 2**31, value=42, key="story_seed_val")

            # Output Settings
            # Scene Transitions
            with st.expander("Scene Transitions", expanded=False):
                from transitions import TRANSITION_TYPES, TRANSITION_DESCRIPTIONS
                story_transition = st.selectbox(
                    "Transition Type",
                    TRANSITION_TYPES,
                    format_func=lambda x: f"{x.replace('_',' ').title()} - {TRANSITION_DESCRIPTIONS.get(x, '')}",
                    key="story_transition",
                )
                if story_transition != "cut":
                    story_transition_dur = st.slider("Transition Duration", 0.25, 2.0, 0.75, 0.25, key="story_trans_dur")
                else:
                    story_transition_dur = 0

            with st.expander("Output Settings", expanded=False):
                story_aspect = st.radio(
                    "Aspect Ratio",
                    ["16:9", "9:16", "1:1", "4:5"],
                    horizontal=True,
                    key="story_aspect",
                    format_func=lambda x: {"16:9": "Landscape", "9:16": "Portrait", "1:1": "Square", "4:5": "Instagram"}.get(x, x)
                )
                story_fps = st.slider("Output FPS", 8, 30, 16, key="story_fps")
                story_neg_prompt = st.text_input(
                    "Global Negative Prompt",
                    value="low quality, blurry, distorted, watermark, text overlay",
                    key="story_neg",
                )

            # ---- GENERATE BUTTON ----
            st.markdown("---")
            generate_movie = st.button(
                "Generate Movie",
                type="primary",
                use_container_width=True,
                disabled=not ok,
            )

        # ---- Output Column ----
        with output_col:
            if generate_movie and scenes:
                st.session_state.story_result = None
                st.session_state.story_generated_videos = {}

                # Build config
                if "Wan" in story_engine:
                    engine_id = "wan"
                    gen_config = VideoGenConfig(
                        engine="wan",
                        wan_model=story_wan_model,
                        wan_resolution=story_wan_res,
                        num_frames=story_frames,
                        num_inference_steps=story_steps,
                        guidance_scale=story_guidance,
                        fps=story_fps,
                        enable_cpu_offload=True,
                    )
                elif "LTX" in story_engine:
                    engine_id = "ltx"
                    gen_config = VideoGenConfig(
                        engine="ltx",
                        ltx_model=story_ltx_model,
                        num_frames=story_frames,
                        num_inference_steps=story_steps,
                        guidance_scale=story_guidance,
                        fps=story_fps,
                        enable_cpu_offload=True,
                    )
                elif "CogVideoX" in story_engine:
                    engine_id = "cogvideox"
                    gen_config = VideoGenConfig(
                        engine="cogvideox",
                        model_variant=story_cog_model,
                        num_frames=story_frames,
                        num_inference_steps=story_steps,
                        guidance_scale=story_guidance,
                        fps=story_fps,
                        enable_cpu_offload=True,
                    )
                elif "Hunyuan" in story_engine:
                    engine_id = "hunyuan"
                    gen_config = VideoGenConfig(
                        engine="hunyuan",
                        num_frames=story_frames,
                        num_inference_steps=story_steps,
                        guidance_scale=story_guidance,
                        fps=story_fps,
                        enable_cpu_offload=True,
                    )
                else:
                    engine_id = "wan"
                    gen_config = VideoGenConfig(engine="wan", fps=story_fps, enable_cpu_offload=True)

                # ---- Background generation with thread ----
                import threading, queue

                # Resolve continuity settings before launching thread
                _use_anchor = story_continuity_mode in ("Prompt Anchoring", "Both")
                _use_chain = story_continuity_mode in ("Scene Chaining", "Both")
                _anchor_text = story_visual_anchor.strip() if _use_anchor and 'story_visual_anchor' in dir() else ""
                _chain_str = story_chain_strength if _use_chain and 'story_chain_strength' in dir() else 0.6
                _shared_seed = story_use_shared_seed if 'story_use_shared_seed' in dir() else False
                _seed = story_seed_val if _shared_seed and 'story_seed_val' in dir() else None

                # Pack all params for the background thread
                job_params = {
                    "scenes": [dict(s) for s in scenes],
                    "gen_config": gen_config,
                    "engine_id": engine_id,
                    "genre": story_genre,
                    "mood": story_mood,
                    "neg_prompt": story_neg_prompt,
                    "fps": story_fps,
                    "use_anchor": _use_anchor,
                    "anchor_text": _anchor_text,
                    "use_chain": _use_chain,
                    "shared_seed": _shared_seed,
                    "seed": _seed,
                    "enable_narration": story_enable_narration,
                    "enable_music": story_enable_music,
                    "music_track": story_music_track if story_enable_music and 'story_music_track' in dir() else None,
                    "music_vol": story_music_vol if story_enable_music and 'story_music_vol' in dir() else 0.12,
                    "transition_type": story_transition if 'story_transition' in dir() else "cut",
                    "transition_duration": story_transition_dur if 'story_transition_dur' in dir() else 0.5,
                }

                progress_q = queue.Queue()
                st.session_state.story_progress_queue = progress_q
                st.session_state.story_job_start_time = time.time()
                st.session_state.story_job_running = True
                st.session_state.story_job_log = []

                def _run_story_generation(params, pq):
                    """Background thread: generates all scenes, audio, final movie."""
                    import torch
                    gen = None
                    try:
                        scenes_data = params["scenes"]
                        total = len(scenes_data)

                        def _log(msg, pct=None, phase="", scene_idx=None, scene_total=None,
                                 step=None, step_total=None, eta=None, vram=None):
                            entry = {
                                "msg": msg, "pct": pct, "phase": phase, "time": time.time(),
                                "scene_idx": scene_idx, "scene_total": scene_total,
                                "step": step, "step_total": step_total, "eta": eta, "vram": vram,
                            }
                            pq.put(entry)

                        # Phase: Load model
                        _log(f"Loading {params['engine_id'].upper()} model...", 0.02, phase="model_load")
                        if torch.cuda.is_available():
                            free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)
                            _log(f"GPU VRAM: {free / 1e9:.1f}GB free", 0.02, phase="model_load",
                                 vram=round(free / 1e9, 1))

                        cfg = params["gen_config"]
                        if params["shared_seed"] and params["seed"] is not None:
                            cfg.seed = params["seed"]

                        _log(f"Engine: {cfg.engine} | Model: {getattr(cfg, 'wan_model', getattr(cfg, 'ltx_model', getattr(cfg, 'model_variant', '')))} | "
                             f"Frames: {cfg.num_frames} | Steps: {cfg.num_inference_steps} | CPU offload: {cfg.enable_cpu_offload}",
                             0.03, phase="model_load")
                        _log("Downloading/loading model weights (this can take minutes on first run)...", 0.03, phase="model_load")

                        gen = VideoGenerator(cfg)

                        _log("Model loaded successfully", 0.05, phase="model_load")
                        if torch.cuda.is_available():
                            free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)
                            _log(f"Post-load VRAM: {free / 1e9:.1f}GB free", 0.05, phase="model_load",
                                 vram=round(free / 1e9, 1))

                        video_paths = []
                        prev_video_path = None

                        # Phase: Generate scenes
                        for i, scene in enumerate(scenes_data):
                            scene_start = time.time()
                            title = scene.get("title", f"Scene {i+1}")

                            prompt = scene.get("visual", "") or scene.get("title", "A cinematic scene")
                            enhanced = f"{prompt}, cinematic, {params['genre'].lower()} style, {params['mood'].lower()} atmosphere, high quality"
                            if params["use_anchor"] and params["anchor_text"]:
                                enhanced = f"{enhanced}, {params['anchor_text']}"

                            # Progress callback from diffusion pipeline
                            def make_scene_cb(si, st_total):
                                def cb(step, total, msg):
                                    elapsed = time.time() - scene_start
                                    eta_s = (elapsed / max(step, 1)) * (total - step) if step > 0 else None
                                    vram_now = None
                                    if torch.cuda.is_available():
                                        vram_now = round((torch.cuda.get_device_properties(0).total_memory
                                                          - torch.cuda.memory_reserved(0)) / 1e9, 1)
                                    overall_pct = 0.05 + ((si + step / max(total, 1)) / (st_total + 2)) * 0.65
                                    _log(msg, min(overall_pct, 0.95), phase="scene_gen",
                                         scene_idx=si, scene_total=st_total,
                                         step=step, step_total=total,
                                         eta=round(eta_s) if eta_s else None, vram=vram_now)
                                return cb

                            scene_cb = make_scene_cb(i, total)

                            # Determine I2V chain vs T2V
                            chain_image = None
                            gen_mode = "Text-to-Video"
                            use_i2v = False

                            # Check if I2V is viable (Wan I2V requires 14B model - needs 16GB+ VRAM)
                            engine = params["gen_config"].engine
                            wan_model = getattr(params["gen_config"], "wan_model", "1.3b")
                            i2v_viable = True
                            if engine == "wan" and "1.3b" in wan_model.lower():
                                # Wan has NO 1.3B I2V model - all I2V models are 14B (~28GB)
                                # Only use I2V if user explicitly uploaded a reference image
                                i2v_viable = False

                            if scene.get("image") and os.path.exists(scene["image"]):
                                chain_image = scene["image"]
                                use_i2v = True
                                gen_mode = "Image-to-Video (reference image)"
                                if not i2v_viable:
                                    _log(f"Warning: Wan 1.3B has no I2V model. Using 14B I2V (large download, needs 16GB+ VRAM).",
                                         phase="scene_gen", scene_idx=i, scene_total=total)
                                    use_i2v = True  # User explicitly wants I2V
                            elif params["use_chain"] and prev_video_path and i > 0:
                                if i2v_viable:
                                    try:
                                        _log(f"Extracting last frame from scene {i} for chaining...",
                                             phase="scene_gen", scene_idx=i, scene_total=total)
                                        chain_image = gen._extract_last_frame(prev_video_path)
                                        use_i2v = True
                                        gen_mode = "Image-to-Video (chained from previous scene)"
                                    except Exception as ce:
                                        _log(f"Chaining failed, using T2V: {ce}", phase="scene_gen",
                                             scene_idx=i, scene_total=total)
                                else:
                                    # Fallback: extract last frame description into the prompt for T2V continuity
                                    _log(f"Scene chaining via prompt (Wan 1.3B has no I2V model, avoiding 14B download)",
                                         phase="scene_gen", scene_idx=i, scene_total=total)
                                    # Add continuity hint: reference previous scene visually in the prompt
                                    enhanced = f"continuation of previous scene, {enhanced}, smooth transition, same visual style and subjects"
                                    gen_mode = "Text-to-Video (prompt-chained)"

                            _log(f"Scene {i+1}/{total}: \"{title}\" | Mode: {gen_mode}",
                                 phase="scene_gen", scene_idx=i, scene_total=total)

                            try:
                                if use_i2v and chain_image:
                                    vid = gen.generate_image2video(chain_image, enhanced, progress_callback=scene_cb)
                                else:
                                    vid = gen.generate_text2video(enhanced, negative_prompt=params["neg_prompt"],
                                                                  progress_callback=scene_cb)

                                video_paths.append(vid)
                                prev_video_path = vid
                                elapsed = time.time() - scene_start
                                _log(f"Scene {i+1} complete ({elapsed:.0f}s)",
                                     phase="scene_done", scene_idx=i, scene_total=total)

                            except Exception as e:
                                _log(f"Scene {i+1} FAILED: {e}", phase="scene_error",
                                     scene_idx=i, scene_total=total)
                                continue

                        if not video_paths:
                            _log("All scenes failed. No video generated.", 1.0, phase="error")
                            return

                        # Phase: TTS Narration
                        audio_path = None
                        if params["enable_narration"]:
                            _log("Generating voiceover narration...", 0.75, phase="tts")
                            try:
                                tts = get_tts_instance()
                                output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mp")
                                os.makedirs(output_dir, exist_ok=True)
                                aud_paths = []
                                for j, sc in enumerate(scenes_data):
                                    if j >= len(video_paths):
                                        break
                                    narr = sc.get("narration", "")
                                    if narr.strip():
                                        _log(f"TTS scene {j+1}/{total}: \"{narr[:50]}...\"", phase="tts")
                                        clean = rf_clean_text_for_tts(narr)
                                        af = os.path.join(output_dir, f"story_tts_{uuid4()}.wav")
                                        tts.synthesize(clean, af)
                                        if os.path.exists(af) and os.path.getsize(af) > 100:
                                            aud_paths.append(af)
                                        else:
                                            aud_paths.append(None)
                                    else:
                                        aud_paths.append(None)

                                valid = [p for p in aud_paths if p]
                                if valid:
                                    combined = os.path.join(output_dir, f"story_narration_{uuid4()}.wav")
                                    rf_concatenate_audio_files(valid, combined)
                                    audio_path = combined
                                    _log(f"Narration complete ({len(valid)} clips)", 0.80, phase="tts")
                            except Exception as e:
                                _log(f"TTS failed: {e}", phase="tts")

                        # Phase: Assembly
                        _log("Assembling final movie...", 0.85, phase="assembly")
                        if len(video_paths) > 1:
                            transition_type = params.get("transition_type", "cut")
                            transition_dur = params.get("transition_duration", 0.5)
                            if transition_type and transition_type != "cut":
                                from transitions import concatenate_with_transitions
                                _log(f"Joining {len(video_paths)} scenes with '{transition_type}' transition...", 0.86, phase="assembly")
                                final_video = concatenate_with_transitions(
                                    video_paths, transition_type=transition_type,
                                    duration=transition_dur, fps=params["fps"],
                                )
                            else:
                                _log(f"Concatenating {len(video_paths)} scene videos...", 0.86, phase="assembly")
                                final_video = concatenate_videos(video_paths, fps=params["fps"])
                        else:
                            final_video = video_paths[0]

                        if audio_path and os.path.exists(audio_path):
                            _log("Adding narration audio track...", 0.90, phase="assembly")
                            final_video = add_audio_to_video(final_video, audio_path, volume=1.0)

                        if params["enable_music"]:
                            _log("Mixing background music...", 0.93, phase="assembly")
                            try:
                                mpath = None
                                if params["music_track"] and params["music_track"] != "Random":
                                    mpath = str(get_music_dir() / params["music_track"])
                                else:
                                    ml = list_background_music()
                                    if ml:
                                        import random as _r
                                        mpath = str(get_music_dir() / _r.choice(ml))
                                if mpath and os.path.exists(mpath):
                                    final_video = add_audio_to_video(final_video, mpath, volume=params["music_vol"])
                                    _log("Background music added", 0.95, phase="assembly")
                            except Exception as e:
                                _log(f"Music mixing failed: {e}", phase="assembly")

                        # Done!
                        _log("DONE", 1.0, phase="complete")
                        pq.put({
                            "phase": "result",
                            "video_path": final_video,
                            "scene_videos": video_paths,
                            "engine": params["engine_id"],
                            "total_scenes": len(video_paths),
                        })

                    except Exception as e:
                        pq.put({"phase": "fatal_error", "msg": str(e), "traceback": traceback.format_exc()})
                    finally:
                        if gen:
                            gen.unload()

                # Launch thread
                thread = threading.Thread(target=_run_story_generation, args=(job_params, progress_q), daemon=True)
                thread.start()
                st.session_state.story_bg_thread = thread
                st.rerun()

            elif generate_movie and not scenes:
                st.warning("Add scenes to your storyboard first.")

        # ============================================================
        # LIVE PROGRESS MONITOR (survives tab switches)
        # ============================================================
        _bg_thread = st.session_state.get("story_bg_thread")
        _job_running = st.session_state.get("story_job_running", False)
        _progress_q = st.session_state.get("story_progress_queue")

        if _job_running and _progress_q is not None:
            import queue as _q_mod

            st.markdown("---")
            st.markdown("### Generating Movie...")

            # Drain queue into log
            if "story_job_log" not in st.session_state:
                st.session_state.story_job_log = []

            result_data = None
            fatal_error = None
            while True:
                try:
                    entry = _progress_q.get_nowait()
                    if entry.get("phase") == "result":
                        result_data = entry
                    elif entry.get("phase") == "fatal_error":
                        fatal_error = entry
                    else:
                        st.session_state.story_job_log.append(entry)
                except _q_mod.Empty:
                    break

            log = st.session_state.story_job_log
            job_start = st.session_state.get("story_job_start_time", time.time())
            elapsed_total = time.time() - job_start

            # Handle completion
            if result_data:
                st.session_state.story_job_running = False
                st.session_state.story_result = {
                    "video_path": result_data["video_path"],
                    "scenes": [dict(s) for s in scenes],
                    "scene_videos": result_data.get("scene_videos", []),
                    "engine": result_data.get("engine", ""),
                    "total_scenes": result_data.get("total_scenes", 0),
                    "frames_per_scene": story_frames if 'story_frames' in dir() else 49,
                    "fps": story_fps if 'story_fps' in dir() else 16,
                }
                st.session_state.story_job_log = []
                st.balloons()
                st.rerun()
            elif fatal_error:
                st.session_state.story_job_running = False
                st.error(f"Generation failed: {fatal_error.get('msg', 'Unknown error')}")
                with st.expander("Error Details"):
                    st.code(fatal_error.get("traceback", ""))
                st.session_state.story_job_log = []
            else:
                # Still running - show rich progress dashboard
                latest = log[-1] if log else {}
                pct = latest.get("pct", 0) or 0

                # Progress bar
                st.progress(min(pct, 0.99), text=latest.get("msg", "Working..."))

                # Elapsed time + ETA
                time_col, vram_col, phase_col = st.columns(3)
                with time_col:
                    mins, secs = divmod(int(elapsed_total), 60)
                    st.metric("Elapsed", f"{mins}m {secs}s")
                with vram_col:
                    vram_entries = [e for e in log if e.get("vram") is not None]
                    if vram_entries:
                        st.metric("GPU VRAM Free", f"{vram_entries[-1]['vram']}GB")
                    else:
                        st.metric("GPU VRAM Free", "...")
                with phase_col:
                    phase = latest.get("phase", "")
                    phase_labels = {
                        "model_load": "Loading Model",
                        "scene_gen": "Generating Scenes",
                        "scene_done": "Scene Complete",
                        "scene_error": "Scene Error",
                        "tts": "Generating Narration",
                        "assembly": "Assembling Movie",
                    }
                    st.metric("Phase", phase_labels.get(phase, phase.replace("_", " ").title() or "Starting"))

                # Scene progress tracker
                total_scenes = latest.get("scene_total") or len(scenes)
                current_scene = latest.get("scene_idx")
                if current_scene is not None:
                    st.markdown("**Scene Progress:**")
                    scene_cols = st.columns(min(total_scenes, 8))
                    completed_scenes = [e.get("scene_idx") for e in log if e.get("phase") == "scene_done"]
                    error_scenes = [e.get("scene_idx") for e in log if e.get("phase") == "scene_error"]
                    for si in range(total_scenes):
                        with scene_cols[si % len(scene_cols)]:
                            if si in completed_scenes:
                                st.success(f"Scene {si+1}")
                            elif si in error_scenes:
                                st.error(f"Scene {si+1}")
                            elif si == current_scene and phase == "scene_gen":
                                st.warning(f"Scene {si+1}...")
                            else:
                                st.caption(f"Scene {si+1}")

                # Inference step detail
                step = latest.get("step")
                step_total = latest.get("step_total")
                eta = latest.get("eta")
                if step is not None and step_total is not None and phase == "scene_gen":
                    step_pct = step / max(step_total, 1)
                    eta_str = f" | ETA: {eta}s" if eta else ""
                    st.caption(f"Inference: step {step}/{step_total} ({step_pct*100:.0f}%){eta_str}")

                # Live log feed (last 8 messages)
                with st.expander("Live Log", expanded=True):
                    recent = log[-12:] if len(log) > 12 else log
                    log_lines = []
                    for e in recent:
                        t = e.get("time", 0) - job_start
                        msg = e.get("msg", "")
                        log_lines.append(f"[{t:6.1f}s] {msg}")
                    st.code("\n".join(log_lines) if log_lines else "Starting...", language="log")

                # Auto-refresh every 2 seconds
                time.sleep(2)
                st.rerun()

        # ============================================================
        # RESULTS SECTION (shown when generation is complete)
        # ============================================================
        result = st.session_state.story_result
        if result and not _job_running:
            st.markdown("---")
            st.markdown("### Your Movie")

            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mp")
            vid_path = result["video_path"]

            if os.path.exists(vid_path):
                st.video(vid_path)

                dl_col, info_col = st.columns([2, 1])
                with dl_col:
                    with open(vid_path, "rb") as vf:
                        video_bytes = vf.read()
                        fname = f"story_{story_concept[:20].replace(' ', '_') if story_concept else 'movie'}.mp4"
                        st.download_button(
                            "Download Movie",
                            video_bytes,
                            file_name=fname,
                            mime="video/mp4",
                            use_container_width=True,
                            type="primary",
                        )
                with info_col:
                    file_size_mb = os.path.getsize(vid_path) / (1024 * 1024)
                    st.metric("File Size", f"{file_size_mb:.1f} MB")

                st.caption(f"Saved to: `{vid_path}`")
            else:
                st.error(f"Movie file not found at: {vid_path}")

            with st.expander("Scene Breakdown", expanded=True):
                scene_vids = result.get("scene_videos", [])
                result_scenes = result.get("scenes", [])
                if scene_vids:
                    cols_per_row = min(3, len(scene_vids))
                    for row_start in range(0, len(scene_vids), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for col_idx in range(cols_per_row):
                            scene_idx = row_start + col_idx
                            if scene_idx < len(scene_vids):
                                with cols[col_idx]:
                                    vp = scene_vids[scene_idx]
                                    if os.path.exists(vp):
                                        st.video(vp)
                                        if scene_idx < len(result_scenes):
                                            s = result_scenes[scene_idx]
                                            st.markdown(f"**Scene {scene_idx+1}: {s.get('title', '')}**")
                                            narr = s.get('narration', '')
                                            if narr:
                                                st.caption(f"_{narr[:100]}_")
                                        with open(vp, "rb") as sf:
                                            st.download_button(
                                                f"Download Scene {scene_idx+1}",
                                                sf.read(),
                                                file_name=f"scene_{scene_idx+1}.mp4",
                                                mime="video/mp4",
                                                key=f"dl_scene_{scene_idx}",
                                                use_container_width=True,
                                            )
                else:
                    st.info("No individual scene videos available.")

            with st.expander("Generation Details & File Locations"):
                st.markdown("**Output Files:**")
                st.code(f"Final movie: {vid_path}", language=None)
                for i, vp in enumerate(result.get("scene_videos", [])):
                    st.code(f"Scene {i+1}:    {vp}", language=None)
                st.markdown(f"**Output directory:** `{output_dir}`")
                st.markdown("---")
                st.json({
                    "engine": result.get("engine", ""),
                    "total_scenes": result.get("total_scenes", 0),
                    "frames_per_scene": result.get("frames_per_scene", ""),
                    "fps": result.get("fps", ""),
                })


# ============================================================
# LOGS PAGE
# ============================================================
elif tool == "Logs":
    st.title("Generation Logs")
    st.caption("View logs from Video Generator and other background processes")

    from videogen import get_generation_logs, get_recent_logs, clear_logs, videogen_logger

    # Controls row
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        auto_refresh = st.checkbox("Auto-refresh", value=True, help="Automatically refresh logs every 2 seconds")

    with col2:
        if st.button("Clear Logs", type="secondary"):
            clear_logs()
            st.rerun()

    with col3:
        log_lines = st.select_slider(
            "Lines to show",
            options=[50, 100, 200, 500],
            value=200
        )

    # Auto-refresh mechanism
    if auto_refresh:
        import time
        st.empty()
        time.sleep(0.1)  # Small delay to ensure state is fresh

    # Get logs (already formatted as strings)
    log_entries = get_recent_logs(limit=log_lines)

    if log_entries:
        # Display in a code block with scrolling
        st.code("\n".join(log_entries), language="log")

        # Stats
        st.caption(f"Showing {len(log_entries)} log entries")
    else:
        st.info("No logs yet. Generate a video to see logs here.")

    # Log file info
    with st.expander("Log File Location"):
        log_file = videogen_logger.log_file
        st.text(f"Log file: {log_file}")
        if os.path.exists(log_file):
            file_size = os.path.getsize(log_file)
            st.text(f"File size: {file_size / 1024:.1f} KB")

    # Auto-refresh trigger
    if auto_refresh:
        import streamlit.components.v1 as components
        components.html(
            """
            <script>
                setTimeout(function() {
                    window.parent.document.querySelector('[data-testid="stApp"]').click();
                }, 2000);
            </script>
            """,
            height=0
        )


# ============================================================
# SETTINGS PAGE
# ============================================================
elif tool == "Settings":
    st.title("Settings")
    st.caption("Configure StudioLite and ReelForge settings")

    cfg = load_config()

    # Status indicators
    st.subheader("System Status")
    col_status1, col_status2, col_status3 = st.columns(3)

    with col_status1:
        llamacpp_ok, llamacpp_msg = check_llamacpp_available()
        gguf_models = get_gguf_models()
        if llamacpp_ok and gguf_models:
            st.success(f"llama.cpp: {len(gguf_models)} model(s)")
        elif llamacpp_ok:
            st.warning("llama.cpp: No models")
        else:
            st.error("llama.cpp: Not installed")

    with col_status2:
        from transcriber import check_whisperx_installed, get_device
        whisper_ok, whisper_msg = check_whisperx_installed()
        if whisper_ok:
            st.success(f"Whisper: {get_device().upper()}")
        else:
            st.warning("Whisper: Not Available")

    with col_status3:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            st.success(f"GPU: {gpu_name[:20]}...")
        else:
            st.warning("GPU: CPU Only")

    st.markdown("---")

    # LLM Settings
    with st.expander("LLM (Text Generation)", expanded=True):
        st.markdown("Configure the AI backend for script generation in ReelForge.")

        # Backend selection
        backends = ["llamacpp", "ollama"]
        current_backend = cfg.get("llm_backend", "llamacpp")
        c_llm_backend = st.selectbox(
            "LLM Backend",
            backends,
            index=backends.index(current_backend) if current_backend in backends else 0,
            help="llamacpp (recommended): Works offline with GGUF models. Ollama: Requires Ollama server."
        )

        if c_llm_backend == "llamacpp":
            st.markdown("**llama.cpp (GGUF Models)**")
            gguf_models = get_gguf_models()
            if gguf_models:
                c_gguf_model = st.selectbox("Select Model", gguf_models, index=0)
            else:
                st.warning("No GGUF models found. Download one below.")
                c_gguf_model = ""

            # Model download section
            st.markdown("**Download Models**")
            for model_info in RECOMMENDED_MODELS:
                col_m1, col_m2 = st.columns([3, 1])
                with col_m1:
                    st.markdown(f"**{model_info['name']}** ({model_info['size']})")
                    st.caption(model_info['description'])
                with col_m2:
                    if st.button(f"Download", key=f"dl_{model_info['id']}", use_container_width=True):
                        with st.spinner(f"Downloading {model_info['name']}..."):
                            try:
                                path = download_model(model_info['id'], model_info['file'])
                                st.success(f"Downloaded to {path}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Download failed: {e}")

        else:  # Ollama
            st.markdown("**Ollama**")
            ollama_ok, ollama_msg = check_ollama_connection()
            if ollama_ok:
                st.success("Ollama is running")
            else:
                st.warning("Ollama not running. [Install Ollama](https://ollama.com/download)")

            c_ollama_url = st.text_input("Ollama Base URL", value=cfg.get("ollama_base_url", "http://127.0.0.1:11434"))
            c_ollama_model = st.text_input("Ollama Model", value=cfg.get("ollama_model", ""), placeholder="e.g., mistral:7b")
            st.caption("Run `ollama pull mistral:7b` to download a model")

    # Image Generation Settings
    with st.expander("Image Generation", expanded=True):
        st.markdown("Configure image generation providers for ReelForge.")
        img_providers = ["nanobanana2", "sdxl_turbo", "fooocus"]
        current_provider = cfg.get("image_provider", "nanobanana2")
        c_img_provider = st.selectbox(
            "Default Provider",
            img_providers,
            index=img_providers.index(current_provider) if current_provider in img_providers else 0
        )

        st.markdown("**SDXL Models (Local)**")
        st.caption("Download realistic face models for better image quality")
        local_sdxl = list_local_models()
        if local_sdxl:
            st.success(f"Installed: {', '.join(local_sdxl[:3])}{'...' if len(local_sdxl) > 3 else ''}")

        for model_info in RECOMMENDED_IMAGE_MODELS:
            col_m1, col_m2 = st.columns([3, 1])
            with col_m1:
                st.markdown(f"**{model_info['name']}** ({model_info['size']}) - {model_info['style']}")
                st.caption(model_info['description'])
            with col_m2:
                # Check if already downloaded
                is_downloaded = model_info['file'] in local_sdxl
                if is_downloaded:
                    st.success("Installed")
                elif st.button(f"Download", key=f"dl_img_{model_info['id']}", use_container_width=True):
                    with st.spinner(f"Downloading {model_info['name']}..."):
                        try:
                            path = download_image_model(model_info['id'], model_info['file'])
                            st.success(f"Downloaded!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Download failed: {e}")

        st.markdown("---")
        st.markdown("**NanoBanana2 (Gemini)**")
        cc1, cc2 = st.columns(2)
        with cc1:
            c_nb2_key = st.text_input("API Key", value=cfg.get("nanobanana2_api_key", ""), type="password")
            c_nb2_model = st.text_input("Model", value=cfg.get("nanobanana2_model", "gemini-3.1-flash-image-preview"))
        with cc2:
            c_nb2_url = st.text_input("API URL", value=cfg.get("nanobanana2_api_base_url", "https://generativelanguage.googleapis.com/v1beta"))
            c_nb2_ratio = st.selectbox("Aspect Ratio", ["9:16", "16:9", "1:1", "4:3"],
                                       index=["9:16", "16:9", "1:1", "4:3"].index(cfg.get("nanobanana2_aspect_ratio", "9:16")))

        st.markdown("**Fooocus**")
        cc3, cc4 = st.columns(2)
        with cc3:
            c_fooocus_url = st.text_input("Fooocus API URL", value=cfg.get("fooocus_api_url", "http://127.0.0.1:8888"))
        with cc4:
            c_fooocus_style = st.text_input("Fooocus Style", value=cfg.get("fooocus_style", "Fooocus V2"))

    # TTS & STT Settings
    with st.expander("TTS & Speech-to-Text"):
        st.markdown("Configure text-to-speech and speech-to-text settings.")

        st.markdown("**Text-to-Speech Engine**")
        tts_engines = ["piper", "kitten"]
        current_engine = cfg.get("tts_engine", "piper")
        c_tts_engine = st.selectbox(
            "TTS Engine",
            tts_engines,
            index=tts_engines.index(current_engine) if current_engine in tts_engines else 0,
            help="Piper: High-quality neural TTS (recommended). KittenTTS: Lightweight, faster but lower quality."
        )

        # Show voices based on engine
        if c_tts_engine == "piper":
            piper_voices = ["Amy", "Ryan", "Lessac", "Kristin", "Bryce", "Danny", "Joe", "Kathleen"]
            current_voice = cfg.get("tts_voice", "Amy")
            c_tts_voice = st.selectbox(
                "Voice",
                piper_voices,
                index=piper_voices.index(current_voice) if current_voice in piper_voices else 0,
                help="High-quality neural voices. Models download automatically."
            )
        else:
            kitten_voices = ["Jasper", "Luna", "Marcus", "Elena", "Thomas", "Sofia", "Alex", "Emma"]
            current_voice = cfg.get("tts_voice", "Jasper")
            c_tts_voice = st.selectbox(
                "Voice",
                kitten_voices,
                index=kitten_voices.index(current_voice) if current_voice in kitten_voices else 0,
                help="Male: Jasper, Marcus, Thomas, Alex | Female: Luna, Elena, Sofia, Emma"
            )

        st.markdown("**Background Music Defaults**")
        cc_m1, cc_m2 = st.columns(2)
        with cc_m1:
            c_music_enabled = st.checkbox("Enable by Default", value=cfg.get("background_music_enabled", False))
        with cc_m2:
            c_music_volume = st.slider("Default Volume", 0.05, 0.5, cfg.get("background_music_volume", 0.15), 0.05)

        st.markdown("**Speech-to-Text**")
        c_stt = st.selectbox("STT Provider", ["local_whisper", "third_party_assemblyai"], index=0)
        cc5, cc6, cc7 = st.columns(3)
        with cc5:
            c_whisper_m = st.selectbox("Whisper Model", ["tiny", "base", "small", "medium", "large"],
                                       index=["tiny", "base", "small", "medium", "large"].index(cfg.get("whisper_model", "base")))
        with cc6:
            c_whisper_d = st.text_input("Whisper Device", value=cfg.get("whisper_device", "auto"))
        with cc7:
            c_whisper_c = st.selectbox("Compute Type", ["int8", "float16", "float32"],
                                       index=["int8", "float16", "float32"].index(cfg.get("whisper_compute_type", "int8")))

    # Video Settings
    with st.expander("Video Settings"):
        st.markdown("Configure video generation and export settings.")
        cc8, cc9 = st.columns(2)
        with cc8:
            c_sentences = st.slider("Default Script Sentences", 2, 12, cfg.get("script_sentence_length", 4))
            c_threads = st.slider("MoviePy Threads", 1, 16, cfg.get("threads", 2))
        with cc9:
            c_font = st.text_input("Font", value=cfg.get("font", "bold_font.ttf"))
            c_magick = st.text_input("ImageMagick Path", value=cfg.get("imagemagick_path", "/usr/bin/convert"))

    # General Settings
    with st.expander("General"):
        c_verbose = st.checkbox("Verbose Logging", value=cfg.get("verbose", True))
        c_headless = st.checkbox("Headless Mode", value=cfg.get("headless", False))

    st.markdown("---")

    # Save button
    if st.button("Save All Settings", type="primary", use_container_width=True):
        cfg["llm_backend"] = c_llm_backend
        if c_llm_backend == "llamacpp":
            if 'c_gguf_model' in dir() and c_gguf_model:
                cfg["gguf_model"] = c_gguf_model
        else:
            cfg["ollama_base_url"] = c_ollama_url
            cfg["ollama_model"] = c_ollama_model
        cfg["image_provider"] = c_img_provider
        cfg["nanobanana2_api_key"] = c_nb2_key
        cfg["nanobanana2_model"] = c_nb2_model
        cfg["nanobanana2_api_base_url"] = c_nb2_url
        cfg["nanobanana2_aspect_ratio"] = c_nb2_ratio
        cfg["fooocus_api_url"] = c_fooocus_url
        cfg["fooocus_style"] = c_fooocus_style
        cfg["tts_engine"] = c_tts_engine
        cfg["tts_voice"] = c_tts_voice
        cfg["background_music_enabled"] = c_music_enabled
        cfg["background_music_volume"] = c_music_volume
        cfg["stt_provider"] = c_stt
        cfg["whisper_model"] = c_whisper_m
        cfg["whisper_device"] = c_whisper_d
        cfg["whisper_compute_type"] = c_whisper_c
        cfg["script_sentence_length"] = c_sentences
        cfg["threads"] = c_threads
        cfg["font"] = c_font
        cfg["imagemagick_path"] = c_magick
        cfg["verbose"] = c_verbose
        cfg["headless"] = c_headless
        save_config(cfg)
        st.success("Settings saved successfully!")
        st.rerun()

    # ---- Model Hub ----
    st.markdown("---")
    st.subheader("Model Hub")
    st.caption("Video generation models: installed status, VRAM requirements, and recommendations")

    from model_hub import get_model_status, get_recommended_model

    model_status = get_model_status()
    built_in = [m for m in model_status if m["built_in"]]
    third_party = [m for m in model_status if not m["built_in"]]

    st.markdown("**Built-in Engines**")
    for m in built_in:
        status_icon = "installed" if m["installed"] else "not downloaded"
        modes = ", ".join(m["modes"])
        st.markdown(
            f"- **{m['name']}** | VRAM: {m['vram_min']}GB+ | "
            f"Quality: {m['quality']} | Speed: {m['speed']} | "
            f"Modes: {modes} | Status: {status_icon}"
        )

    if third_party:
        st.markdown("**Community Models (installable)**")
        for m in third_party:
            st.markdown(
                f"- **{m['name']}** | VRAM: {m['vram_min']}GB+ | "
                f"Quality: {m['quality']} | Speed: {m['speed']}"
            )

    # Recommendation
    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            rec = get_recommended_model(vram)
            if rec:
                st.info(f"Recommended for your GPU ({vram:.0f}GB): **{rec}**")
    except Exception:
        pass
