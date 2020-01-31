#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Joscha Schmiedt
# @Date:   2019-02-04 14:45:08
# @Last Modified by:   Joscha Schmiedt
# @Last Modified time: 2019-08-26 10:00:03
#
# merge_sev.py - Merge separate TDT SEV files into one HDF5 file
#
#


from __future__ import division, print_function

import numpy as np
import re
import argparse
import os
from sys import exit
import json
from getpass import getuser
import datetime
from tqdm.auto import tqdm
import h5py

HEADERSIZE = 40 # bytes
ALLOWED_FORMATS = ('single','int32','int16','int8','double','int64')

def read_header(filename):
    """Read the header of a TDT SEV file created by the RS4 streamer"""
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
        else:
            header['Fs'] = 0
            
        return header
        
def read_data(filename):
    """Read data from a TDT SEV file created by the RS4 streamer"""

    h = read_header(filename)
    with open(filename, 'rb') as f:
        f.seek(HEADERSIZE)
        data = np.fromfile(f, dtype=h['dForm'])
    return data  

def natural_sort(l): 
    """Sort a list of strings using numbers
    
    Ch1 will be followed by Ch2 and not Ch11.
    
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key=alphanum_key)

def get_filenames():
    """Simple file dialog to select SEV files"""
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
    
    desc = "Merge multiple SEV files into one HDF5 file (2048 byte header)"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('files', type=str, nargs='*')
    parser.add_argument("-n", "--no-natural-sort",
                        action="store_true", default=False,
                        help="Do not sort channels using natural sorting (default: False)")
    parser.add_argument("-m", "--remove-median",
                        action="store_true", default=False,
                        help="Subtract median offset of each channel (default: False)")
    parser.add_argument("--channels-at-once", type=int, default=32, 
                        help="Channels/files to read/write at once")
    parser.add_argument("--output-dir", type=str, 
                        help="Output directory for merged file")
     
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
    
    if not args.output_dir:
        outputdir = datadir
    else:
        outputdir = args.output_dir
    
    if not os.path.isdir(outputdir):
        raise IOError("{0} is not a valid output directory".format(outputdir))
    
    basenames = [os.path.basename(x) for x in files]
    sharedBasename = [x[:re.search("_[Cc]h.*\.sev", x).start()] for x in basenames]
    if not all_elements_equal(sharedBasename):
        raise ValueError("Not all basenames are the same")

    if not all_elements_equal([os.path.getsize(f) for f in files]):
        raise ValueError('Not all files have same size')
   
    headers = [read_header(f) for f in files]

    targetFilename = os.path.join(outputdir, sharedBasename[0] + '.hdf5')
    with h5py.File(targetFilename, 'w') as targetFile:
                
        idxStartStop = [np.clip(np.array((jj, jj+args.channels_at_once)), 
                                a_min=None, a_max=len(files)) 
                        for jj in range(0,len(files),args.channels_at_once)]                 
        print("Merging {0} files in {1} chunks a {2} channels into \n   {3}".format(
              len(files), len(idxStartStop), args.channels_at_once, 
              targetFilename))
        for (start, stop) in tqdm(iterable=idxStartStop, desc="chunk", unit="chunk"): 
            data = [read_data(files[jj]) for jj in range(start, stop)]
            data = np.vstack(data).T           
            if start == 0:
                target = targetFile.create_dataset("data",
                                                   shape=(data.shape[0], len(files)),
                                                   dtype=headers[0]["dForm"])
        
            if args.remove_median:
                data -= np.median(data, keepdims=True).astype(data.dtype)                        

            target[:, start:stop] = data
        
        # trialdefinition is a dataset necessary for Syncopy    
        # trialdefinition = targetFile.create_dataset("trialdefinition",
        #                                         shape=(1, 3),
        #                                         dtype=np.uint64)
        # trialdefinition[:] = np.array([0, target.shape[0], 0])
           
        info = {
            "filename" : targetFilename,
            "dataclass" : "AnalogData",
            "data_dtype" : headers[0]["dForm"],
            "data_shape" : target.shape,
            "data_offset" : target.id.get_offset(),
            # "trl_dtype" : trialdefinition.dtype.name,
            # "trl_shape" : trialdefinition.shape,
            # "trl_offset" : trialdefinition.id.get_offset(),
            "order" : "C",
            "dimord" : ["time", "channel"],                    
            "samplerate" : headers[0]["Fs"],
            "channel" : ["channel_{:03d}".format(iChannel) 
                         for iChannel in range(1, len(files))],
            "_version" : "0.1a",
            "_log" : ""
        }
                        
        targetFile.attrs["_log"] = info["_log"]
        targetFile.attrs["_version"] = info["_version"]
        targetFile.attrs["channel"] = info["channel"]
        targetFile.attrs["samplerate"] = info["samplerate"]

        
    # write info file
    info["cfg"] = {
        "originalFiles": basenames,
        "samplingRate": headers[0]["Fs"],
        "dtype": headers[0]["dForm"], 
        "numberOfChannels": len(basenames),
        "mergedBy": getuser(),
        "mergeTime": str(datetime.datetime.now()),
        "md5sum": md5sum(targetFilename),
        "channelMedianSubtracted": args.remove_median
    }

    
    jsonFile = os.path.join(outputdir, sharedBasename[0] + '.info')
    print('Writing info file to {0}...'.format(jsonFile))
    with open(jsonFile, 'w') as fid:
        json.dump(info, fid, indent=4)
