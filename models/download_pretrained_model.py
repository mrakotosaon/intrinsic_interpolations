import os
import urllib
import tarfile

def download_dataset(source_url, target_dir, target_file):
    global downloaded
    downloaded = 0

    print('downloading ... ')
    urllib.urlretrieve(source_url, filename=target_file)
    print('downloading ... done')

    print('extracting ...')
    tar = tarfile.open(target_file, "r:gz")
    tar.extractall()
    tar.close()
    os.remove(target_file)
    print('extracting ... done')


if __name__ == '__main__':
    source_url = 'https://nuage.lix.polytechnique.fr/index.php/s/r6nmfsCQayrozD9/download'
    target_dir = os.path.dirname(os.path.abspath(__file__))
    target_file = os.path.join(target_dir, 'intrinsic_interpolation_pretrained_DFAUST.tar.gz')
    download_dataset(source_url,  target_dir, target_file)
