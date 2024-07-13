from minio import Minio
import os
import argparse

def download_from_airlab_server(client: Minio, bucket_name, source_name, target_name):
    """
    Downloads a file using Minio.

    Args:

    Returns:
    """
    print(f"Downloading {source_name} from {bucket_name}...")
    client.fget_object(bucket_name, source_name, target_name)
    print(f"Successfully downloaded {source_name} to {target_name}!")

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--output', required=True, help='Path to output directory.')

    args = parser.parse_args()

    output_directory = os.path.join(args.output, 'VLA_Dataset')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    bucket_name = "vla"
    endpoint_url = "airlab-share-01.andrew.cmu.edu:9000"

    access_key = "pLyVvRrA7KZiGdseXbUM"
    secret_key = "caXs1AynlxAUA47Sc4GdC5i7gcLjp5jElLE7v8Hu"

    client = Minio(endpoint_url,
                access_key=access_key,
                secret_key=secret_key,
                secure=True,
                cert_check=False)
    
    datasets = ['Unity.zip', '3RScan.zip', 'ARKitScenes.zip', 'HM3D.zip', 'Scannet.zip', 'Matterport.zip']

    for dataset in datasets:
        res = download_from_airlab_server(client,
                                        bucket_name,
                                        source_name=dataset,
                                        target_name=os.path.join(output_directory, dataset))