import os
from pytubefix import YouTube  # pyright: ignore[reportMissingImports]
from pytubefix.cli import on_progress  # pyright: ignore[reportMissingImports]
import common
from custom_logger import CustomLogger
from logmod import logs

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)


def download_youtube_live(url):
    output_folder = common.get_configs("data")
    os.makedirs(output_folder, exist_ok=True)

    # Initialize YouTube object with OAuth enabled (no Client needed)
    yt = YouTube(
        url,
        use_oauth=True,           # triggers Google login when needed
        allow_oauth_cache=True,   # reuse stored OAuth credentials
        on_progress_callback=on_progress,
    )

    # Choose the best progressive MP4 stream (up to 1080p if available)
    stream = (
        yt.streams
        .filter(progressive=True, file_extension="mp4")
        .order_by("resolution")
        .desc()
        .first()
    )

    if not stream:
        logger.error("No suitable progressive MP4 stream found.")
        return

    safe_title = yt.title.replace("/", "_").replace("\\", "_")
    filename = f"{safe_title}.mp4"

    logger.info(f"Downloading: {yt.title} ({stream.resolution})")

    stream.download(
        output_path=output_folder,
        filename=filename,
    )

    logger.info(f"Download complete: {os.path.join(output_folder, filename)}")


if __name__ == "__main__":
    live_url = common.get_configs("url")
    download_youtube_live(live_url)
