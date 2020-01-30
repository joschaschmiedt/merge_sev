# Merge TDT SEV files into HDF5 file

This small command line script can be used to merge the raw data files created
by the TDT RS4 streamer into one HDF5 file that can be used with most
spikesorters such as SpyKING CIRCUS, the Klusters suite or Kilosort.

The input files can either be passed as an argument to the merge_sev.py script,
or if ommited selected in a popup GUI.

The output is a HDF5 and an INFO JSON file with metadata. The HDF5 file can 
either be read using an HDF5 library or as a raw binary data with a 2048 byte
header.