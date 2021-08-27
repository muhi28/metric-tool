import os


def get_video_files(folder_path):
    """
        parse video file paths from selected directory
    :param folder_path: selected folder path
    :return: list of video file paths
    """
    video_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".mp4")]

    return video_files
