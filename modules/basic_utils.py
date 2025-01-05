import os
import ujson as json
import shutil


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def read_lines(filepath):
    with open(filepath, "r") as f:
        return [e.strip("\n") for e in f.readlines()]


def mkdirp(p):
    if not os.path.exists(p):
        os.makedirs(p)


def deletedir(p):
    if os.path.exists(p):
        shutil.rmtree(p)
