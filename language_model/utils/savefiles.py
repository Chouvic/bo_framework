# -*- coding: utf-8 -*-
import os
import os.path

def save_str_app(targetstr, filename):
    filename = "results/" + filename
    basedir = os.path.dirname(filename)
    try:
        if not os.path.exists(basedir):
            os.makedirs(basedir)
    except OSError as exc:
        pass

    with open(filename, 'a') as f:
        f.write('\n')
        f.write(targetstr)

def saveResultAppend(filename, info, results):
    filename = "results/" + filename
    basedir = os.path.dirname(filename)
    try:
        if not os.path.exists(basedir):
            os.makedirs(basedir)
    except OSError as exc:
        pass
    with open(filename, 'a') as f:
        f.write('\n')
        f.write(info +'\n')
        f.write(str(results))
        f.write('\n')

def saveResult(filename, results):
    with open(filename, 'w') as f:
        f.write(str(results))
