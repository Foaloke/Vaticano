'''
Created on 7 Jul 2017

@author: mtonnicchi
'''

import codecs
import os
import requests

def load_source(source_dir, source_file):
    full_path = os.path.join(source_dir, source_file)
    if not os.path.isfile(full_path):
        print('File not found! ('+full_path+')')
    else:
        with codecs.open(os.path.join(source_dir, source_file), 'rb', 'utf-8') as file_conn:
            return file_conn.read()

def save_source(source_dir, source_file, content):
    prepare_directory(source_dir)
    full_path = os.path.join(source_dir, source_file)
    if os.path.isfile(full_path):
        print('File already exists! ('+full_path+')')
    else:
        with codecs.open(os.path.join(source_dir, source_file), 'w', 'utf-8') as file_conn:
            file_conn.write(content)
            file_conn.close()

def file_exists(source_dir, source_file):
    full_path = os.path.join(source_dir, source_file)
    return os.path.isfile(full_path)

def path_exists(full_path):
    return os.path.isfile(full_path)

def download_from_url(source_url, source_dir, source_file):
    response = requests.get(source_url)
    raw_source = response.content
    text = raw_source.decode('utf-8')

    prepare_directory(source_dir)

    with codecs.open(os.path.join(source_dir, source_file), 'w', 'utf-8') as out_conn:
        out_conn.write(text)
        out_conn.close()
    
    return text

def prepare_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    
    
    