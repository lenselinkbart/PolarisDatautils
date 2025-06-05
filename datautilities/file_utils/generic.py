
import os

def delete_files(file_list):
    """
    This function deletes a list of files.

    Args:
    - file_list: List of paths to the files to delete.

    Returns:
    - None
    """

    for file in file_list:
        try:
            os.remove(file)
            #print(f"File {file} has been deleted successfully")
        except FileNotFoundError:
            print(f"File {file} not found")
        except OSError as e:
            print(f"Error: {e.filename} - {e.strerror}.")
