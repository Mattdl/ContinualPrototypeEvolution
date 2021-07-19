import os


def createdirs(dirpath):
    try:
        os.makedirs(os.path.dirname(dirpath), mode=0o750, exist_ok=True)
    except Exception as e:
        print(e)
        print("ERROR IN CREATING DIRS:", os.path.dirname(dirpath))
