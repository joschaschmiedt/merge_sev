#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Joscha Schmiedt
# @Date:   2019-02-04 14:45:08
# @Last Modified by:   Joscha Schmiedt
# @Last Modified time: 2019-02-11 11:53:22
#
# merge_sev.py - Merge separate SEV files into one headerless DAT file
#
#
# TODO  
# * Currently all channels are loaded into memory. This doesn't work fo
#   arbitrary data lengths and will be enhanced in a later version


from __future__ import division, print_function

import numpy as np
import re
import argparse
import os
from sys import exit
import json
from getpass import getuser
import datetime
from tqdm import tqdm

HEADERSIZE = 40; # bytes
ALLOWED_FORMATS = ('single','int32','int16','int8','double','int64')

def read_header(filename):    
    header = {}
    
    with open(filename, 'rb') as f:
        header['fileSizeBytes'] = int.from_bytes(f.read(8), byteorder='little', signed=False)
        header['fileType'] = f.read(3).decode('utf-8')
        header['fileVersion'] = int.from_bytes(f.read(1), byteorder='little', signed=False)
        if header['fileVersion'] < 3:            
            header['eventName']  = f.read(4).decode('utf-8')
            if header['fileVersion'] == 2:
                header['eventName'] = header['eventName'][::-1]
                
                
            header['channelNum'] = int.from_bytes(f.read(2), byteorder='little', signed=False)
            header['totalNumChannels'] = int.from_bytes(f.read(2), byteorder='little', signed=False)
            header['sampleWidthBytes'] = int.from_bytes(f.read(2), byteorder='little', signed=False)
            f.seek(2, 1)
            header['dForm'] = ALLOWED_FORMATS[np.bitwise_and(int.from_bytes(f.read(1), 
                                                                  byteorder='little', signed=False), 7)]
            decimate = int.from_bytes(f.read(1), byteorder='little', signed=False)
            rate = int.from_bytes(f.read(2), byteorder='little', signed=False)
            
        if header['fileVersion'] > 0:
            header['Fs'] = 2**(rate)*25000000/(2**12)/decimate
        return header
        
def read_data(filename):
    h = read_header(filename)
    with open(filename, 'rb') as f:
        f.seek(HEADERSIZE)
        data = np.fromfile(f, dtype=h['dForm'])
    return data  

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key=alphanum_key)

def get_filenames():
    from tkinter import Tk
    from tkinter.filedialog import askopenfilenames
    Tk().withdraw() 
    return askopenfilenames(filetypes=(("sev files","*.sev"),("all files","*.*")))


def md5sum(filename):
    from hashlib import md5
    hash = md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(128 * hash.block_size), b""):
            hash.update(chunk)
    return hash.hexdigest()

def all_elements_equal(elements):
    return all(elem == elements[0] for elem in elements)


if __name__ == "__main__":

    # Short description of the program
    desc = "Merge multiple SEV files into one headerless DAT file"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('files', type=str, nargs='*')
    parser.add_argument("-n", "--no-natural-sort",
                        action="store_true", default=False,
                        help="Do not sort channels using natural sorting (default: False)")
    parser.add_argument("-m", "--remove-median",
                        action="store_true", default=True,
                        help="Subtract median offset of each channel (default: False)")



    # Parse CL-arguments - abort if things get wonky
    args = parser.parse_args()

    if not args.files:
        files = get_filenames()
        if not files:
            exit()
    else:
        files = args.files

    files = [os.path.abspath(x) for x in files]    
    if not args.no_natural_sort:
        files = natural_sort(files)
    
    datadir = os.path.dirname(files[0])
    basenames = [os.path.basename(x) for x in files]
    sharedBasename = [x[:re.search("_[Cc]h.*\.sev", x).start()] for x in basenames]
    if not all_elements_equal(sharedBasename):
        raise ValueError("Not all basenames are the same")

    if not all_elements_equal([os.path.getsize(f) for f in files]):
        raise ValueError('Not all files have same size')
   
    headers = [read_header(f) for f in files]

    targetFilename = os.path.join(datadir, sharedBasename[0] + '.dat')
    print("Merging {0} files...".format(len(files))) 
    for idx, filename in enumerate(tqdm(files)):

        data = read_data(filename)

        if idx == 0:
            target = np.memmap(targetFilename, mode='w+', shape=(data.size,len(files)),
                               dtype=headers[0]["dForm"])
    
        if args.remove_median:
            data -= np.median(data, keepdims=True).astype(data.dtype)

        target[:,idx] = data
    

    # write info file
    info = {
        "originalFiles": basenames,
        "samplingRate": headers[0]["Fs"],
        "dtype": headers[0]["dForm"], 
        "numberOfChannels": len(basenames),
        "mergedBy": getuser(),
        "mergeTime": str(datetime.datetime.now()),
        "md5sum": md5sum(targetFilename),
        "channelMedianSubtracted": args.remove_median
    }
    
    jsonFile = os.path.join(datadir, sharedBasename[0] + '.json')
    print('Writing info file to {0}...'.format(jsonFile))
    with open(jsonFile, 'w') as fid:
        json.dump(info, fid, indent=4)
    print("")


    




    

