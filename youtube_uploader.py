#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YouTube Video Uploader using YouTube Data API v3.

Requires:
1. A Google Cloud project with YouTube Data API v3 enabled
2. OAuth 2.0 credentials (client_secrets.json)
3. User authorization to upload videos

Setup instructions:
1. Go to https://console.cloud.google.com/
2. Create a new project or select existing one
3. Enable YouTube Data API v3
4. Create OAuth 2.0 credentials (Desktop app type)
5. Download credentials as 'client_secrets.json' and place in project root
"""

import os
import logging
import pickle
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# YouTube API constants
YOUTUBE_UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# Valid privacy statuses
VALID_PRIVACY_STATUSES = ("public", "private", "unlisted")

# Video categories (common ones)
VIDEO_CATEGORIES = {
    "Film & Animation": "1",
    "Autos & Vehicles": "2",
    "Music": "10",
    "Pets & Animals": "15",
    "Sports": "17",
    "Travel & Events": "19",
    "Gaming": "20",
    "People & Blogs": "22",
    "Comedy": "23",
    "Entertainment": "24",
    "News & Politics": "25",
    "Howto & Style": "26",
    "Education": "27",
    "Science & Technology": "28",
    "Nonprofits & Activism": "29",
}


def check_dependencies() -> tuple[bool, str]:
    """Check if YouTube API dependencies are installed."""
    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
        return True, "Dependencies installed"
    except ImportError as e:
        return False, f"Missing dependency: {e}. Run: pip install google-api-python-client google-auth-oauthlib"


def check_client_secrets(secrets_path: str = "client_secrets.json") -> tuple[bool, str]:
    """Check if client secrets file exists."""
    if os.path.exists(secrets_path):
        return True, secrets_path
    return False, f"Missing {secrets_path}. Download from Google Cloud Console."


class YouTubeUploader:
    """Handles YouTube video uploads using OAuth 2.0."""

    def __init__(self, client_secrets_path: str = "client_secrets.json",
                 token_path: str = "youtube_token.pickle"):
        """
        Initialize the uploader.

        Args:
            client_secrets_path: Path to OAuth client secrets JSON file
            token_path: Path to store/load authentication token
        """
        self.client_secrets_path = client_secrets_path
        self.token_path = token_path
        self.credentials = None
        self.youtube = None

    def is_authenticated(self) -> bool:
        """Check if user is already authenticated."""
        if os.path.exists(self.token_path):
            try:
                with open(self.token_path, "rb") as token:
                    self.credentials = pickle.load(token)
                if self.credentials and self.credentials.valid:
                    return True
                if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                    from google.auth.transport.requests import Request
                    self.credentials.refresh(Request())
                    self._save_credentials()
                    return True
            except Exception as e:
                logger.warning(f"Failed to load credentials: {e}")
        return False

    def _save_credentials(self):
        """Save credentials to token file."""
        with open(self.token_path, "wb") as token:
            pickle.dump(self.credentials, token)

    def authenticate_with_local_server(self, port: int = 8080) -> bool:
        """
        Authenticate using local server redirect (recommended method).

        This will open a browser window for the user to authorize the app,
        then automatically capture the authorization code via local redirect.

        Args:
            port: Local port for the OAuth callback server

        Returns:
            True if successful, False otherwise
        """
        try:
            from google_auth_oauthlib.flow import InstalledAppFlow

            flow = InstalledAppFlow.from_client_secrets_file(
                self.client_secrets_path,
                scopes=[YOUTUBE_UPLOAD_SCOPE]
            )

            # Run local server to handle OAuth callback
            self.credentials = flow.run_local_server(
                port=port,
                prompt='consent',
                success_message='Authentication successful! You can close this window and return to StudioLite.',
                open_browser=True
            )

            self._save_credentials()
            return True
        except Exception as e:
            logger.error(f"Failed to authenticate: {e}")
            return False

    def build_service(self) -> bool:
        """Build the YouTube API service."""
        if not self.credentials:
            return False
        try:
            from googleapiclient.discovery import build
            self.youtube = build(
                YOUTUBE_API_SERVICE_NAME,
                YOUTUBE_API_VERSION,
                credentials=self.credentials
            )
            return True
        except Exception as e:
            logger.error(f"Failed to build YouTube service: {e}")
            return False

    def upload_video(
        self,
        video_path: str,
        title: str,
        description: str = "",
        category: str = "Entertainment",
        tags: Optional[list] = None,
        privacy_status: str = "private",
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Upload a video to YouTube.

        Args:
            video_path: Path to the video file
            title: Video title (max 100 characters)
            description: Video description (max 5000 characters)
            category: Video category name
            tags: List of tags (max 500 characters total)
            privacy_status: One of 'public', 'private', 'unlisted'
            progress_callback: Optional callback function(progress_percent)

        Returns:
            Dict with 'success', 'video_id', 'url', 'error' keys
        """
        result = {"success": False, "video_id": None, "url": None, "error": None}

        if not os.path.exists(video_path):
            result["error"] = f"Video file not found: {video_path}"
            return result

        if not self.youtube:
            if not self.build_service():
                result["error"] = "Failed to build YouTube service. Please re-authenticate."
                return result

        # Validate inputs
        title = title[:100] if title else "Untitled Video"
        description = description[:5000] if description else ""

        if privacy_status not in VALID_PRIVACY_STATUSES:
            privacy_status = "private"

        category_id = VIDEO_CATEGORIES.get(category, "24")  # Default to Entertainment

        if tags:
            tags = [t.strip() for t in tags if t.strip()][:30]  # Max 30 tags
        else:
            tags = []

        try:
            from googleapiclient.http import MediaFileUpload
            from googleapiclient.errors import HttpError

            body = {
                "snippet": {
                    "title": title,
                    "description": description,
                    "tags": tags,
                    "categoryId": category_id
                },
                "status": {
                    "privacyStatus": privacy_status,
                    "selfDeclaredMadeForKids": False
                }
            }

            # Create media upload object with resumable upload
            media = MediaFileUpload(
                video_path,
                mimetype="video/*",
                resumable=True,
                chunksize=1024 * 1024  # 1MB chunks
            )

            # Create the upload request
            request = self.youtube.videos().insert(
                part=",".join(body.keys()),
                body=body,
                media_body=media
            )

            # Execute upload with progress tracking
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status and progress_callback:
                    progress_callback(int(status.progress() * 100))

            if response:
                video_id = response.get("id")
                result["success"] = True
                result["video_id"] = video_id
                result["url"] = f"https://www.youtube.com/watch?v={video_id}"
                logger.info(f"Video uploaded successfully: {result['url']}")

        except HttpError as e:
            error_msg = str(e)
            if "quotaExceeded" in error_msg:
                result["error"] = "YouTube API quota exceeded. Try again tomorrow."
            elif "forbidden" in error_msg.lower():
                result["error"] = "Permission denied. Check your YouTube channel settings."
            else:
                result["error"] = f"Upload failed: {error_msg}"
            logger.error(result["error"])
        except Exception as e:
            result["error"] = f"Upload failed: {str(e)}"
            logger.error(result["error"])

        return result

    def logout(self):
        """Remove stored credentials."""
        if os.path.exists(self.token_path):
            os.remove(self.token_path)
        self.credentials = None
        self.youtube = None


def get_category_list() -> list:
    """Return list of available video categories."""
    return list(VIDEO_CATEGORIES.keys())
