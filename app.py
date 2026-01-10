#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import tempfile
import os
from remover import WatermarkRemover
from youtube_uploader import (
    YouTubeUploader, check_dependencies, check_client_secrets, get_category_list
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
     "Merge Videos", "Extract Frame", "Export Video", "View & Publish"]
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
        # Convert to H.264 MP4 for browser compatibility
        with st.spinner("Preparing video for preview..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                preview_path = tmp.name
            if remover.export_video(input_path, preview_path, "mp4", "medium", None):
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
