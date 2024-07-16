import os
import argparse
from minio import Minio
from tqdm import tqdm

# AirLab server config
BUCKET = "vla"
ENDPOINT = "airlab-share-01.andrew.cmu.edu:9000"

# Public keys (for downloading)
ACCESS_KEY = "pLyVvRrA7KZiGdseXbUM"
SECRET_KEY = "caXs1AynlxAUA47Sc4GdC5i7gcLjp5jElLE7v8Hu"


def get_from_server(client, bucket_name, source_name, target_name):
    """
    Downloads a specific file from server using Minio

    Args: 
    client: Minio client object with set up with keys
    bucket_name: str name of data bucket
    source_name: name of file on server
    target_name: name of file locally

    Returns: True
    """
    print(f"Downloading {source_name} from {bucket_name}...")
    client.fget_object(bucket_name, source_name, target_name)
    print(f"Successfully downloaded {source_name} to {target_name}!")

    return True


def download(args):
    """
    Configures download client and loops through files
    """
    source_files = {"matterport":"Matterport.zip", "scannet":"Scannet.zip", "hm3d":"HM3D.zip", "unity":"Unity.zip", "arkitscenes":"ARKitScenes.zip", "3rscan":"3RScan.zip"}
    client = Minio(ENDPOINT,
                access_key=ACCESS_KEY,
                secret_key=SECRET_KEY,
                secure=True)
    
    if not os.path.exists(args.download_path):
        os.mkdir(args.download_path)

    file_list = [file for name, file in source_files.items()]
    
    if args.subset:
        subset_name = args.subset.lower()
        try:
            file = source_files[subset_name]
            file_list = [file]
        except KeyError:
            print(f"{args.subset} is not a valid source name. See the valid subset names in the documentation or help command.")
            return False

    for file in tqdm(file_list):
        target_name = os.path.join(args.download_path, file)
        res = get_from_server(client,
                            BUCKET,
                            source_name=file,
                            target_name=target_name)
    
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--download_path', default='VLA-3D_dataset', help="Directory to store downloaded data.")
    parser.add_argument('--subset', required=False, type=str, help="One of Matterport/Scannet/HM3D/Unity/ARKitScenes/3RScan.")

    args = parser.parse_args()

    download(args)

    