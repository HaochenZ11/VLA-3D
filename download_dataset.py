import os
import argparse
from tqdm import tqdm
import boto3
from botocore import UNSIGNED
from botocore.client import Config

# AirLab server config
BUCKET = "vla"
ENDPOINT = "https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330"

def get_from_server(client, bucket_name, source_name, target_name):
    """
    Downloads a specific file from server using boto3

    Args: 
    client: boto3 S3 client object
    bucket_name: str name of data bucket
    source_name: name of file on server
    target_name: name of file locally

    Returns: True
    """
    print(f"Downloading {source_name} from {bucket_name}...")
    try:
        resp = client.get_object(Bucket=bucket_name, Key=source_name)
        
        # Create target directory if it does not exist
        target_dir = os.path.dirname(target_name)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        # Get file size for progress bar
        file_size = resp['ContentLength']
        
        with open(target_name, 'wb') as file_data:
            with tqdm(total=file_size, unit='B', unit_scale=True, desc=source_name) as pbar:
                for chunk in resp["Body"].iter_chunks(chunk_size=1024 * 1024):
                    if chunk:
                        pbar.update(len(chunk))
                        file_data.write(chunk)
    except Exception as err:
        print(f"Error while downloading {source_name} to {target_name}: {err}")
    print(f"Successfully downloaded {source_name} to {target_name}!")

    return True


def download(args):
    """
    Configures download client and loops through files
    """
    source_files = {"matterport":"Matterport.zip", "scannet":"Scannet.zip", "hm3d":"HM3D.zip", "unity":"Unity.zip", "arkitscenes":"ARKitScenes.zip", "3rscan":"3RScan.zip"}
    
    client = boto3.client("s3", 
                         endpoint_url=ENDPOINT, 
                         config=Config(signature_version=UNSIGNED))
    
    if not os.path.exists(args.download_path):
        os.makedirs(args.download_path)

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


