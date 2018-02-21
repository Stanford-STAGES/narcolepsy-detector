# @requirements:
# pip3 install --upgrade pip
# pip3 install pyedflib

# @description
# Tool for (1) identifying all unique .edf channel names in a specified directory and
# (2) creating categories to group like channels for processing in a pipeline.
# Creates json file for with fields named by the categories provided as well as
# - pathname: Path of directory initially parsed
# - edfFiles: List of all edf files parsed for channel names
# - categories: List of categories (string labels) provided, which match the remaining field names
# @usage From the command line:
#  python channel_label_identifier.py <path name with .edf files> [channel_categories, separated_by_spaces]
#
# @example Create a json file with categories for C3 and C4 through user selection:
#   python channel_label_identifier . C3 C4
# @example List all unique signal labels found in the current (.) directory
# python channel_label_identifier .
#
# @author Hyatt Moore
# @date 2/20/2018

import sys
import json
from pathlib import Path
from pyedflib import EdfReader
import numpy as np

JSON_FILENAME = 'signal_labels.json'

def getEDFFilenames(path2check):
    edfFiles = getEDFFiles(path2check)
    return [str(i) for i in edfFiles];

def getEDFFiles(path2check):
    p = Path(path2check)
    # verify that we have an accurate directory
    # if so then list all .edf/.EDF files
    if p.is_dir():
        print('Checking',path2check,"for edf files.");
        edfFiles = p.glob('*.[Ee][Dd][Ff]'); # make search case-insensitive
    else:
        print(path2check," is not a valid directory.")
        edfFiles = [];
    return list(edfFiles);

def getSignalHeaders(edfFilename):
    print("Reading headers from ",edfFilename)
    edfR = EdfReader(str(edfFilename));
    return edfR.getSignalHeaders();

def getChannelLabels(edfFilename):
    channelHeaders = getSignalHeaders(edfFilename)
    return [fields["label"] for fields in channelHeaders]

def displaySetSelection(label_set):
    numCols = 4
    curItem = 0
    width = 24;
    rowStr = '';
    for label in label_set:
        rowStr = rowStr+str(str(str(curItem)+".").ljust(4)+label).ljust(width);
        curItem = curItem+1;
        if curItem%numCols==0:
            print(rowStr);
            rowStr = '';
    if(len(rowStr)>0):
        print(rowStr);

def printUsage(toolName):
    print("Usage:\n\t",toolName," <pathname to search> <channel category> {<channel category>}");
    print("Example:\n\t",toolName," . C3 C4");

if __name__ == '__main__':
    # if number of input arguments is none
    if len(sys.argv) < 2:
        printUsage(sys.argv[0]);
    else:
        path2check = sys.argv[1];
        jsonFileOut = Path(path2check).joinpath(JSON_FILENAME);
        channelsToID  = sys.argv[2:];

        edfFiles = getEDFFilenames(path2check)
        num_edfs = len(edfFiles)
        if num_edfs == 0:
            print("No files found!")
        else:
            label_set = set()
            for edfFile in edfFiles:
                # only add unique channel labels to our set`
                label_set = label_set.union(set(getChannelLabels(edfFile)))

            label_list = sorted(label_set)
            print();

            if len(channelsToID)>0:
                print("Enter acceptable channel indices to use for the given identifier. Use spaces to separate multiple indices.")
                print();

            displaySetSelection(label_list)
            print()

            if len(channelsToID)>0:

                toFile = {}; #dict();
                toFile['pathname']=path2check; # a string
                toFile['edfFiles']=edfFiles; # a list
                toFile['categories']=channelsToID; # a list of strings

                for ch in channelsToID:
                    indices = [int(num) for num in input(ch+': ').split()];
                    selectedLabels = [label_list[i] for i in indices]
                    print("Selected: ",selectedLabels)
                    toFile[ch] = selectedLabels;



                jsonStr = json.dumps(toFile,indent = 4,sort_keys = True)
                jsonFileOut.write_text(jsonStr);
                print(json.dumps(toFile))
                print();
                print('JSON data written to file:',jsonFileOut)
