import requests


def read_output(path):
    """Opens the output of the MAUS aligner."""
    with open(path, 'r') as f:
        r = f.read().replace('<', ' ').replace('>', ' ').split()
    return r


def get_url(output_lst):
    """Extract the download url for the TextGrid file."""
    url = [url for url in output_lst if 'https' in url]
    return url


def download_textgrid(url, filename):
    response = requests.get(url)
    file = f'{filename}.TextGrid'
    if response.status_code == 200:
        print('--MAUS response: True')
        with open(file, 'wb') as f:
            f.write(response.content)
        print(f'--TextGrid file "{file.split('/')[-1]}" downloaded successfully.')
        print('-'*40)
    else:
        print(f"--Failed to download file. Status code: {response.status_code}")

def download_par_file(url, filename):
    response = requests.get(url)
    file = f'{filename}.par'
    if response.status_code == 200:
        print('--G2P Response: True')
        with open(file, 'wb') as f:
            f.write(response.content)
        print(f'--File "{file.split('/')[-1]}" downloaded successfully.')
        print('-'*40)
    else:
        print(f"--Failed to download file. Status code: {response.status_code}")
    

def main():
    # Change to current folder if needed
    path = "/Users/matteo/Desktop/HT2024/5LN709_master_thesis/response_output.txt"
    output = read_output(path)
    url = get_url(output)
    download_textgrid(url[0])

if __name__ == '__main__':
    main()