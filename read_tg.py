
import sys

def read_textgrid(file):
    with open(file, "r", encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        # remove whitespaces at the beginning and at the end of each string
        lin = [lin.strip() for lin in lines]
    return lin


if __name__ == '__main__':
    path = sys.argv[1]
    tg = read_textgrid(path)
    print(tg)
