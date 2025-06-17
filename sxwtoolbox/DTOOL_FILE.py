# file operation
import os

def list_files(directory):
    """
    return all filenames in a directory

    @param:
    directory (str): target directory

    @return:
    list: all filenames in a directory
    """
    try:
        files = os.listdir(directory)
        file_list = [f for f in files if os.path.isfile(os.path.join(directory, f))]
        return file_list
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

