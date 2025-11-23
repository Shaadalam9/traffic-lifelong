import yt_dlp
import os
import common
from custom_logger import CustomLogger
from logmod import logs

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger


def download_youtube_live(url):
    output_folder = common.get_configs("data")
    cookies_file = "data/cookies.txt"

    os.makedirs(output_folder, exist_ok=True)

    ydl_opts = {
        "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "live_from_start": True,   # start from beginning of DVR window (if any)

        # one single mp4 file in the folder
        "outtmpl": os.path.join(output_folder, "%(title)s.%(ext)s"),
        "merge_output_format": "mp4",

        # authentication
        "cookiefile": cookies_file,

        # resume if partially downloaded, but never make new files
        "continuedl": True,
        "overwrites": False,

        "postprocessors": [{
            "key": "FFmpegVideoRemuxer",
            "preferedformat": "mp4",
        }],
    }

    logger.info(f"Starting continuous recording with cookies: {cookies_file}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # this call will keep running and writing to ONE file
        ydl.download([url])
    logger.info("yt-dlp finished (stream ended or process stopped).")


if __name__ == "__main__":
    live_url = common.get_configs("url")
    download_youtube_live(live_url)
