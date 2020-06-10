import os

def label(input_dir, output_file):
    files = [f for f in os.listdir(input_dir) if f.endswith(".png")]

if __name__ == '__main__':
    print('a')
