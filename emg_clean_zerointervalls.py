#!/usr/bin/env python

import sys
import os.path

if sys.version_info[0] < 3:
    import Tkinter as Tk
    from tkFileDialog import askopenfilename
    from tkMessageBox import showerror
else:
    import tkinter as Tk
    from tkinter.filedialog import askopenfilename
    from tkinter.messagebox import showerror

import numpy as np


#CONFIG CONSTANTS
ECG_ZERO_THRES = 0.01    
ECG_ZERO_MIN_LENGTH = 500
ECG_ZERO_EDGE_DELTA = 2500
    
PEAKS_ZERO_MIN_LENGTH = 1500
PEAKS_ZERO_EDGE_DELTA = 2500
PEAKS_ZERO_NOJUMP_LENGTH = 60*500


def writeCSV(filename_in, filename_out, usage_new, jump_indices):

    jump_col = -1
        
    is_header_line = True
    i = 0
    with open(filename_in,'r') as fin:
        with open(filename_out,'w') as fout:
            for line in fin:
                if is_header_line:
                    header_columns = line.rstrip('\r\n').split(",")                                        
                    if jump_indices:
                        if '"jump_ibi"' in header_columns:
                            jump_col = header_columns.index('"jump_ibi"')
                        else:
                            print('ERROR: Expected column "jump_ibi" not found!')
                            sys.exit(-1)
                    
                    new_header_line = line.rstrip('\r\n')
                    new_header_line = new_header_line.replace('"usage_total"', '"usage_total_orig"')
                    if jump_indices:
                        new_header_line = new_header_line.replace('"jump_ibi"', '"jump_ibi_orig"')
                        new_header_line = '{},"usage_total","jump_ibi"\r\n'.format(new_header_line)
                    else:
                        new_header_line = '{},"usage_total"\r\n'.format(new_header_line)
                    fout.write(new_header_line)
                    is_header_line = False
                else:
                    if jump_indices:
                        if i in jump_indices:
                            fout.write('{},{},{}\r\n'.format(line.rstrip('\r\n'), str(usage_new[i]), '1'))
                        else:
                            tmp = line.rstrip('\r\n').split(",")
                            fout.write('{},{},{}\r\n'.format(line.rstrip('\r\n'), str(usage_new[i]), tmp[jump_col]))
                        
                    else:
                        fout.write('{},{}\r\n'.format(line.rstrip('\r\n'), str(usage_new[i])))
                    
                    i+=1

    return i


def readData(filename):
    ecg_col = -1
    peaks_col = -1
    usage_col = -1
    
    ecg_data = []
    peaks_data = []
    usage_data = []
    
    is_header_line = True
    i = 0
    with open(filename,'r') as fin:
        for line in fin:
            if is_header_line:
                header_columns = line.rstrip('\r\n').split(",")

                if '"f"' in header_columns:
                    ecg_col = header_columns.index('"f"')
                else:
                    print('ERROR: Expected column "f" not found!')
                    sys.exit(-1)
                
                if '"peaks_raw"' in header_columns:
                    peaks_col = header_columns.index('"peaks_raw"')
                else:
                    print('ERROR: Expected column "peaks_raw" not found!')
                    sys.exit(-1)
                
                if '"usage_total"' in header_columns:
                    usage_col = header_columns.index('"usage_total"')
                else:
                    print('ERROR: Expected column "usage_total" not found!')
                    sys.exit(-1)
                
                is_header_line = False
            else:
                tmp = line.rstrip('\r\n').split(",")
                ecg_data.append(float(tmp[ecg_col]))
                peaks_data.append(float(tmp[peaks_col]))
                usage_data.append(float(tmp[usage_col]))
                i+=1
                        
    print("reading done")
    return [ecg_data, peaks_data, usage_data, i]



if __name__ == "__main__":
    print('Number of arguments: {} arguments.'.format(len(sys.argv)))
    print('Argument List: {}'.format(str(sys.argv)))

    if (len(sys.argv)) > 1:
        filename = sys.argv[1]
    else:
        root = Tk.Tk()
        filename = askopenfilename()
        if not filename:
            sys.exit(0)
    
    while not os.path.isfile(filename):
        root = Tk.Tk()
        showerror("File not found", "Sorry, file '{}' was not found, try again".format(filename))
        filename = askopenfilename()
        if not filename:
            sys.exit(0)
    
    if root:
        root.quit()
        root.destroy()
    
    print('Processing file: {}'.format(filename))
    
    [ecg, peaks_raw, usage_total, file_length] = readData(filename)
    
    ecg_zero_mask = np.ones(len(ecg))
    peaks_zero_mask = np.ones(len(ecg))
    peaks_jump_idx = []

    #check ecg for zero intervalls    
    ecg_abs = np.absolute(ecg)
    ecg_zero = (ecg_abs < ECG_ZERO_THRES)*1
    
    ecgdiff = np.diff(ecg_zero)
    idx1 = np.nonzero(ecgdiff>0)[0]
    idx2 = np.nonzero(ecgdiff<0)[0]

    if len(idx1)>0 or len(idx2)>0:
        if len(idx2)>0 and len(idx1) == 0:
            idx1 = [0]
        elif len(idx1)>0 and len(idx2) == 0:
            idx2 = [len(ecg)-1]
        else:
            if idx2[0] < idx1[0]:
                idx1 = np.insert(idx1, 0, 0)
                
            if len(idx2) < len(idx1):
                idx2 = np.append(idx2, len(ecg)-1)
            
        for i in range(0, len(idx1)):
            if idx2[i]-idx1[i] >= ECG_ZERO_MIN_LENGTH:
                ecg_zero_mask[max(0, idx1[i]-ECG_ZERO_EDGE_DELTA):min(idx2[i]+ECG_ZERO_EDGE_DELTA, len(ecg)-1)] = 0

    #check peaks for no-activity intervalls
    peak_idx = np.nonzero(peaks_raw)[0]
    peak_dist = np.diff(peak_idx)
    
    no_peak_interval_idx = np.nonzero(peak_dist>=PEAKS_ZERO_MIN_LENGTH)[0]    

    for hb in no_peak_interval_idx:
        peaks_zero_mask[max(0, peak_idx[hb]-PEAKS_ZERO_EDGE_DELTA):min(peak_idx[hb+1]+PEAKS_ZERO_EDGE_DELTA, len(ecg)-1)] = 0
        if peak_dist[hb]<PEAKS_ZERO_NOJUMP_LENGTH:
            peaks_jump_idx.append(peak_idx[hb])

    #finish
    mask_all = np.logical_and(ecg_zero_mask,peaks_zero_mask)*1
    
    usage_new = np.logical_and(usage_total,mask_all)*1
    
    if np.sum(usage_total-usage_new) != 0:
        print('Changes found, saving to file ...')
        filename_out = filename.replace('.csv', '_tmpNEW.csv')
        new_file_length = writeCSV(filename, filename_out, usage_new, peaks_jump_idx)
        if new_file_length == file_length:
            os.rename(filename_out, filename)
            print('Finished')
        else:
            print('ALERT: Something might have gone wrong, "{}" seems to be incomplete'.format(filename_out))
            print('       Please check and rename it yourself')
    else:
        print('No changes found, closing without saving')
