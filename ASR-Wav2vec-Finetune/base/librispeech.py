'''
Generate the corresponding txt for LibriSpeech
'''
import os
def get_filename(directory, l):
    fname = l[0]
    fname = directory + '/' + '/'.join(fname.split('-')[:2]) + '/' + fname + '.flac'
    return [fname, l[1]]
if __name__ == '__main__':
    directory = '../dataset/librispeech-100'
    save_name = 'librispeech-100.txt'
    txt = ['path|transcript\n']
    for i in os.listdir(directory):
        for j in os.listdir(os.path.join(directory, i)):
            path = os.path.join(directory, i, j, i + '-' + j + '.trans.txt')
            with open(path) as p:
                lines = p.readlines()
                lines = ['|'.join(get_filename(directory, l.split(' ', 1))) for l in lines]
                txt += lines
    print(len(txt))
    with open(save_name, 'w') as f:
        f.writelines(txt)

