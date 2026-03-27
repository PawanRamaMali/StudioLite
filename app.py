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
     "Merge Videos", "Extract Frame", "Export Video", "Transcribe", "View & Publish",
     "ReelForge", "Video Generator", "Settings"]
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
    st.markdown("Generate real videos using CogVideoX or LTX-Video diffusion models")

    from videogen import VideoGenerator, check_system_requirements, VideoGenConfig, get_available_engines
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

        # Recommend LTX for high VRAM, CogVideoX for low VRAM
        default_engine_idx = 1 if vram >= 12 else 0
        vg_engine = st.radio(
            "Video Engine",
            ["CogVideoX", "LTX-Video (Best Quality)"],
            index=default_engine_idx,
            horizontal=True,
            help="LTX-Video: Best quality, 30 FPS, requires 10GB+ VRAM. CogVideoX: Good quality, works on 8GB+"
        )
        engine_key = "ltx" if "LTX" in vg_engine else "cogvideox"

        # Mode selection
        vg_mode = st.radio(
            "Generation Mode",
            ["Text to Video", "Image to Video", "Extend Video"],
            horizontal=True,
            help="Choose how to generate your video"
        )

        # Prompt input
        vg_prompt = st.text_area(
            "Describe the video",
            placeholder="A panda playing guitar in a bamboo forest, cinematic lighting, smooth camera movement...",
            height=120,
            help="Describe what should happen in the video. Be specific about motion and style."
        )

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
            if engine_key == "ltx":
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

            progress_bar = st.progress(0, text="Initializing...")
            status_text = st.empty()

            def on_progress(step, total, msg):
                progress_bar.progress(step / total, text=msg)
                status_text.text(msg)

            # Configure generator based on selected engine
            if engine_key == "ltx":
                config = VideoGenConfig(
                    engine="ltx",
                    ltx_model=vg_ltx_model,
                    num_frames=vg_num_frames,
                    num_inference_steps=vg_steps,
                    guidance_scale=vg_guidance,
                    width=768 if vram >= 16 else 704,
                    height=512,
                    fps=24,
                    seed=vg_seed,
                )
            else:
                config = VideoGenConfig(
                    engine="cogvideox",
                    model_variant=vg_model_variant,
                    num_frames=vg_num_frames,
                    num_inference_steps=vg_steps,
                    guidance_scale=vg_guidance,
                    quantization="int8" if vg_use_quant else "none",
                    fps=8,
                    seed=vg_seed,
                )

            generator = VideoGenerator(config)

            try:
                if vg_mode == "Text to Video":
                    on_progress(0, 5, "Starting text-to-video generation...")
                    result = generator.generate_text2video(
                        vg_prompt,
                        progress_callback=on_progress
                    )

                elif vg_mode == "Image to Video" and vg_image_file:
                    on_progress(0, 5, "Processing image...")
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
                    on_progress(0, 5, "Processing video...")
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
                    result = None

                if result:
                    st.session_state.vg_result = result
                    progress_bar.progress(1.0, text="Complete!")
                    st.success("Video generated successfully!")

            except Exception as e:
                st.error(f"Generation failed: {e}")
                import traceback
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
                st.json({
                    "mode": vg_mode,
                    "model": vg_model_choice,
                    "frames": vg_num_frames,
                    "guidance_scale": vg_guidance,
                    "quantization": "INT8" if vg_use_quant else "None",
                    "seed": vg_seed,
                })


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
