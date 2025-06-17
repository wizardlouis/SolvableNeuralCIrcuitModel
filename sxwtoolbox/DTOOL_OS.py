import os

opj=os.path.join

def create(path):
    if not os.path.exists(path):
        os.makedirs(path)

