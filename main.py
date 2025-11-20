import yt_dlp
import os
import common
from custom_logger import CustomLogger
from logmod import logs

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger


def download_youtube_live(url):
    # Set your folder here
    output_folder = common.get_configs("data")

    # Create folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    ydl_opts = {
        'format': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]',
        'live_from_start': True,

        # Save into the folder: downloads/<title>.mp4
        'outtmpl': os.path.join(output_folder, '%(title)s.%(ext)s'),

        'merge_output_format': 'mp4',

        'postprocessors': [{
            'key': 'FFmpegVideoRemuxer',
            'preferedformat': 'mp4'
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


if __name__ == "__main__":
    live_url = common.get_configs("url")
    download_youtube_live(live_url)
