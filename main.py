"""YouTube Livestream Recorder.

This module records a YouTube livestream into a single media file using yt-dlp.
Authentication is performed via cookies loaded directly from a local browser
profile (for example, Chrome), which allows access to login-restricted content.

The recording continues as long as:
  * the livestream is active,
  * the network connection is stable, and
  * the Python process is running.

Stop the recording by terminating the process (Ctrl+C).

Notes:
    - This implementation intentionally avoids setting an explicit `format`
      or `live_from_start` in the yt-dlp options. In practice, those
      parameters can cause "Requested format is not available" errors for
      some YouTube livestreams when different manifests (e.g., MPD vs HLS)
      are served between probe and download.
    - Instead, yt-dlp's default format selection is used, which is the same
      behavior as running `yt-dlp --cookies-from-browser <browser> <url>`
      from the command line.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import yt_dlp

import common
from custom_logger import CustomLogger
from logmod import logs

# Initialize logging using the application's configuration.
logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)


def download_youtube_live(url: str, browser: str = "chrome") -> None:
    """Record a YouTube livestream into a single output file.

    This function uses yt-dlp to download a YouTube livestream and write it
    into one file in the configured data directory. Browser cookies are loaded
    directly from the specified browser profile to ensure that the request is
    authenticated.

    The behavior is intentionally close to the CLI invocation:

        yt-dlp --cookies-from-browser <browser> "<url>" \
               -o "<data_dir>/%(title)s.%(ext)s"

    Args:
        url:
            The URL of the YouTube livestream to record.
        browser:
            The browser name from which to load cookies. Typical values
            include "chrome", "firefox", "edge", or "brave". The browser must
            be installed locally, and the user must be logged into YouTube in
            that browser profile.

    Raises:
        RuntimeError:
            If yt-dlp encounters a fatal error while starting or running
            the download.
    """
    # Directory where the recorded file will be stored.
    output_folder = common.get_configs("data")
    os.makedirs(output_folder, exist_ok=True)

    # Output template: one file per livestream, named by YouTube title.
    outtmpl = os.path.join(output_folder, "%(title)s.%(ext)s")

    # Minimal, robust yt-dlp configuration.
    # NOTE:
    #   - We do NOT set "format" here; we let yt-dlp use its default
    #     best/bestvideo+bestaudio logic.
    #   - We do NOT set "live_from_start" here; for some lives this can
    #     change which manifest is used and trigger "Requested format is not
    #     available".
    ydl_opts: Dict[str, Any] = {
        # Where to save the file.
        "outtmpl": outtmpl,

        # Load authenticated cookies directly from the browser profile.
        # This is equivalent to --cookies-from-browser=<browser>.
        "cookiesfrombrowser": (browser,),

        # Resume an interrupted download into the same file when possible.
        "continuedl": True,
        "overwrites": False,
    }

    logger.info(
        "Starting livestream recording.\n"
        f"  URL: {url}\n"
        f"  Output directory: {output_folder}\n"
        f"  Cookies from browser: {browser}"
    )

    try:
        # yt-dlp will keep writing to a single file until the stream ends
        # or the process is stopped.
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore[arg-type]
            ydl.download([url])
    except yt_dlp.utils.DownloadError as exc:  # type: ignore[attr-defined]
        logger.error(f"yt-dlp reported a fatal download error: {exc}")
        raise RuntimeError("Failed to record the YouTube livestream.") from exc

    logger.info("yt-dlp finished (stream ended or process was stopped).")


if __name__ == "__main__":
    # Retrieve the target livestream URL from the application configuration
    # and start recording using the default browser ("chrome") for cookies.
    live_url = common.get_configs("url")
    download_youtube_live(live_url, browser="chrome")
