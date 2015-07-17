import sys

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


import numpy as np
from numpy.fft import fft, ifft, fftshift
import copy
from scipy.interpolate import interp1d
from scipy import signal

import math

import argparse


def lowpass(data, cutoff):
	fourier = fftshift(fft(np.array(data,dtype=float)))
	timestep = 0.002
	freq = np.fft.fftfreq(len(data), d=timestep)
	freq = fftshift(freq)

	i_low = np.where(freq > -abs(cutoff))[0][0]
	i_high = np.where(freq > abs(cutoff))[0][0]
    
	result = np.zeros((1,len(data)), dtype=complex)[0]
	result[i_low:i_high] = fourier[i_low:i_high]

	result_shift = fftshift(np.array(result))
	data_filt = ifft(np.array(result_shift,dtype=complex))

	return data_filt.real


def HMS(seconds, pos):
	"""Customized x-axis ticks 
	
	Keyword arguments:
	seconds -- input in secs
	pos -- somehow required argument (matplotlib)
	"""
	seconds = int(seconds)
	hours = seconds // 3600
	seconds -= 3600 * hours
	minutes = seconds // 60
	seconds -= 60 * minutes
	if hours == 0:
		if minutes == 0:
			return "%ds" % (seconds)
		return "%dm%02ds" % (minutes, seconds)
	return "%dh%02dm" % (hours, minutes)


def readData(filename):
	infile = open(filename)
	infile.readline()
	f_dia = []
	rms_dia = []
	time_axis = []
	time_as_string = []
	motiondetection_usage = []
	count = 0
	for l in infile:
		tmp = l.split(",")
		if len(tmp) < 4:
			continue
		f_dia.append(float(tmp[2]))
		rms_dia.append(float(tmp[3]))
		time_axis.append(count/500.0)
		time_as_string.append(tmp[0])
		if len(tmp) >= 5:
			motiondetection_usage.append(int(tmp[4]))
		else:
			motiondetection_usage.append(1)
		count += 1

	print ("reading done")
	return [f_dia, rms_dia, time_axis, motiondetection_usage, time_as_string]


def thresholdRMSD(rmsd_data, cutoff, delta):
	print ("RMSD Filter")
	index = np.zeros((1,len(rmsd_data)), dtype=int)[0]
	index[rmsd_data<cutoff]=1
	idxdiff = np.diff(index)
	idx1 = np.nonzero(idxdiff<0)[0]
	for i in idx1:
		index[max(0,i+1-delta):i+1]=0
	idx2 = np.nonzero(idxdiff>0)[0]
	for i in idx2:
		index[i+1:min(len(index),i+1+delta)]=0
	return index




def findPeaks(data, lower_bound):
	peaks = np.zeros(len(data))
	peaks[data>lower_bound] = 1
	return peaks


def cleanPeaks(peaks, window):
	result = copy.deepcopy(peaks)
	j = 0
	for i in np.nonzero(result>0)[0]:
		if i >= j:
			j = min(len(result), i+peak_clean_range)
			result[i+1:j] = 0
	return result


def detectQRS(emg):
	## QRS-Detection
	# Based on DF1 (adapted to sampling rate) from Paper
	# "A Comparison of the Noise Sensitivity of Nine QRS Detection Algorithms"
	# Friesen et al., IEEE 1990
	# CAUTION: Hardcoded for a sampling frequency of 500Hz!

	# differentiator with 62.5 Hz notch filter
	print("start QRS")
	y0 = np.zeros((1,len(emg)), dtype=float)[0]
	
	for n in range(8, len(emg)-6):
         y0[n] = emg[n] - emg[n-8]

	# lowpass filter
	y1 = np.zeros((1,len(emg)), dtype=float)[0]
	for n in range(5, len(emg)-1):
         y1[n] = sum(y0[n-5:n]*np.array([1, 4, 6, 4, 1]))

	# detection thresholds
	thresUP = np.mean(y1)+np.std(y1)      # upper threshold
	thresDOWN = np.mean(y1)+np.std(y1)    # lower threshold
	bound = 80          # search window size (=160ms)

	# QRS detection
	cand = np.nonzero(y1[0:-bound-1] > thresUP)[0]
	qrs = [];
	for c in range(0, len(cand)-1):
         i = cand[c];
         if qrs:
             if i < qrs[-1]+bound:
                 continue;
         qrsflag = 0
         j = []
         k = []
         l = []
         j = np.nonzero(y1[i+1:i+bound] < thresDOWN)[0]
         if len(j) > 0:
             qrsflag = 1
             k = np.nonzero(y1[i+j[0]:i+bound] > thresUP)[0]
             if len(k) > 0:
                 l = np.nonzero(y1[i+j[0]+k[0]:i+bound] < thresDOWN)[0]
                 if len(l) > 0:
                     if len(np.nonzero(y1[i+j[0]+k[0]+l[0]:i+bound] > thresUP)[0]) > 0:
                         qrsflag = 0;

         if qrsflag > 0:
             qrs.append(i)
	
	print("QRS done")
	return qrs

def getTimeString(t):
	seconds = int(t)
	hours = seconds // 3600
	seconds -= 3600 * hours
	minutes = seconds // 60
	seconds -= 60 * minutes
	return str(hours) + ":" + str(minutes) + ":" + str(seconds)


def writeCSV(filename, time_as_string, usage_md, usage_rmsd, f, rmsd, beat, beat_raw, qrs_beat, lp_all, lp_filtered, ibint_peak, ibint_qrs, usage_bib, usage_total, jump_vec, ibint_tacho=None):
	print ("writing output")
	
	content = []

	header = '"time","usage_md","usage_rmsd","f","rmsd","beat","peaks_raw","qrs_beat","lp_all","lp_filtered","ibint_peak","ibint_qrs","usage_rmsd","usage_total","jump_ibi"'

	if ibint_tacho != None:
		header += ',"ibint_tacho"'

	header += '\r\n'
	#outfile.write(header)

	content.append(header)

	for i in range(0,len(time_as_string)):
		if (i>0) and (np.fmod(i,10000) < 0.001):
			sys.stdout.write("\r\t\t%d%%" % float((100.0*i)/len(time_as_string)) )
			sys.stdout.flush()
		result = time_as_string[i] + ","
		result += str(usage_md[i]) + ","
		result += str(usage_rmsd[i]) + ","
		result += str(f[i]) + ","
		result += str(rmsd[i]) + ","
		result += str(beat[i]) + ","
		result += str(beat_raw[i]) + ","
		result += str(qrs_beat[i]) + ","
		result += str(lp_all[i]) + ","
		result += str(lp_filtered[i]) + ","
		result += str(ibint_peak[i]) + ","
		result += str(ibint_qrs[i]) + ","
		result += str(int(usage_bib[i])) + ","
		result += str(int(usage_total[i])) + ","
		result += str(int(jump_vec[i]))
		
		if ibint_tacho != None:
			result += "," + str(ibint_tacho[i])
		
		result += "\r\n"
		#outfile.write(result)
		content.append(result)

	outfile = open(filename, 'w')
	outfile.writelines(content)
	outfile.close()
	print ("\r\ndone")


def getBeatVectorsForInt(data_x, data_y):
	result_x = [0]
	result_y = [0]
	for i in range(0,len(data_y)):
		if data_y[i] == 1:
			dx = 1000
			for j in range(1,1000):
				if  (i+j)<len(data_y) and data_y[i+j] == 1:
					dx = j
					break
			if dx < 1000:
				result_x.append(data_x[i])
				result_y.append(1.0/(float(dx)/500))

	result_x.append(beat_int_x[-1]+5)
	result_y.append(beat_int_y[-1])
	result_y[0] = result_y[1] - 0.1
	return [result_x, result_y]


def extractHFFromFile(filename):
	infile = open(filename)
	infile.readline()
	x = []
	y = []
	count = 0
	for l in infile:
		tmp = l.split(',')
		x.append(count/5.0)
		y.append(float(tmp[2])/60.0)
		count += 1
	infile.close()
	return [x,y]

def getDiffVec(v1, v2, v3):
	result = []
	for i in range(0,len(v1)):
		tmp = []
		tmp.append(abs(v1[i]-v2[i]))
		tmp.append(abs(v2[i]-v3[i]))
		tmp.append(abs(v1[i]-v3[i]))
		tmp = sorted(tmp)
		result.append(tmp[0])
	return result

def getUsageVec(vec_in, th, delta):
	result = np.ones(len(vec_in))
	for i in range(0,len(vec_in)):
		if vec_in[i] > th:
			#for j in range( max(0,i-int(1.2*delta))  ,  min(i+int(1.2*delta),len(vec_in))    ):
			#	result[j] = 0
			result[max(0,i-int(1.2*delta)):min(i+int(1.2*delta),len(vec_in))] = 0
	return list(result)

def generateDefensiveUsageVector(movie_vec, rmsd_vec, bib_vec):
	result = []
	print("debug", len(movie_vec), len(rmsd_vec), len(bib_vec))
	for i in range(0,len(movie_vec)):
		if movie_vec[i] == 1 and rmsd_vec[i] == 1 and bib_vec[i] == 1:
			result.append(1)
		else:
			result.append(0)
	return result

def generateIBIJumpVector(movie_vec, rmsd_vec, bib_vec, delta, diff_vec):
	print ("jump vector creation")
	result = []
	for i in range(0,len(movie_vec)):
		if movie_vec[i] == 1 and rmsd_vec[i] == 1 and bib_vec[i] == 0:
			result.append(1)
		else:
			result.append(0)

	i = 10 * 500
	ind = []
	while i < len(result):
		if result[i] == 1:
			ind.append(i + delta/2)
			while (i<len(result)-1) and (result[i]==1):
				result[i] = 0
				i += 1
		i += 1

	result2 = list(np.zeros(len(movie_vec)))

	d = 2000
	for i in ind:
		if np.max(diff_vec[i-d:i+d]) > 0.2:
			if int(i) < len(result2):
				result2[int(i)] = 1
				minutes = int(i/500)//60
				sec = int(60*((i/500)/60.0 - minutes))
				print ("\tjump-position found:", minutes, "mins", sec, "secs")
	print ("done")
	return result2


def peak_correction(peaks, f):
	dx = 15
	ind = []
	for i in range(2*dx,len(peaks)-3*dx-3):
		if peaks[i] == 1:
			ind.append(i)

	left = 0
	right = 0
	for i in ind:
		tmp = f[i-dx:i+dx+1]
		ind_temp = np.argmin(tmp)-dx+i
		tmp2 = f[ind_temp:ind_temp+2*dx]
		ind_temp2 = np.argmax(tmp2)+ind_temp
		if (ind_temp2-ind_temp) > dx/2:
			left += 1
			tmp2 = f[ind_temp-2*dx:ind_temp]
			offset = len(tmp2) - np.argmax(tmp2)
			ind_temp2 = ind_temp - offset
		else:
			right += 1

	
	print ("left:", left)
	print ("right", right)
	
	ind_corr = []
	for i in ind:
		tmp = f[i-dx:i+dx+1]
		ind_temp = np.argmin(tmp)-dx+i
		tmp2 = f[ind_temp:ind_temp+2*dx]
		ind_temp2 = np.argmax(tmp2)+ind_temp
		if left>right:
			tmp2 = f[ind_temp-2*dx:ind_temp]
			offset = len(tmp2) - np.argmax(tmp2)
			ind_temp2 = ind_temp - offset
		ind_corr.append(ind_temp2)

	result = list(np.zeros(len(peaks)))
	for i in ind_corr:
		result[i] = 1
	
	return result


def peak_correction2(peaks, f, mode, offset):
	dx = 12
	ind = []
	for i in range(2*dx,len(peaks)-3*dx-3):
		if peaks[i] == 1:
			ind.append(i)

	ind_corr = []
	for i in ind:
		tmp = f[i-dx+offset:i+dx+1+offset]
		if mode == 1:
			ind_new = np.argmax(tmp)-dx+i+offset
		else:
			ind_new = np.argmin(tmp)-dx+i+offset
		ind_corr.append(ind_new)
	
	result = list(np.zeros(len(peaks)))
	for i in ind_corr:
		result[i] = 1
	
	return result


if __name__ == "__main__":
	#print 'Number of arguments:', len(sys.argv), 'arguments.'
	#print 'Argument List:', str(sys.argv)

	parser = argparse.ArgumentParser(description='Run EMG Analysis')
	parser.add_argument('filename', metavar='Input File', type=str, nargs=1, help='Name of the input file (CSV-File)')
	parser.add_argument('-l', '--limit', default=0, type=int, required=False)
	parser.add_argument('-p', '--print_pdf', action='store_true', default=False, dest='boolean_switch_pdf', help='Set a switch to true')
	parser.add_argument('-b', '--print_beat', action='store_true', default=False, dest='boolean_switch_beat', help='Set a switch')
	parser.add_argument('-hf', '--extra-file', default='', type=str, required=True)
	
	parser.add_argument('-min', '--min', action='store_true', default=False, dest='boolean_switch_min', help='Set a switch')
	parser.add_argument('-max', '--max', action='store_true', default=False, dest='boolean_switch_max', help='Set a switch')
	parser.add_argument('-off', '--offset', default='0', type=int, required=False)
	parser.add_argument('-sd', '--sd_th', default='1', type=float, required=False)

	args = parser.parse_args()

	if args.extra_file != '':
		print ("extra input file given:", args.extra_file)

	filename = args.filename[0]

	#Flag for writing PDFs (or just CSV)
	writePDF = args.boolean_switch_pdf
	writeBeat = args.boolean_switch_beat
	sd_th = args.sd_th

	print(args.boolean_switch_max, args.boolean_switch_min, args.offset)

	if args.boolean_switch_max:
		manual_peak_mode = 1
	elif args.boolean_switch_min:
		manual_peak_mode = 0
	else:
		manual_peak_mode = None

	print("manual_peak", manual_peak_mode)
	
	manual_offset = int(args.offset)

	if writePDF:
		pdf_file = filename.rsplit(".",1)[0] + ".pdf"
		print ("pdf-file:", pdf_file)

	pdf_file_peakfitting = filename.rsplit(".",1)[0] + "_peak_fitting.pdf"
	print ("pdf-file2:", pdf_file_peakfitting)

	csv_out_file = filename.rsplit(".",1)[0] + "_eval.csv"
	print ("csv-out-file:", csv_out_file)
	
	[y_raw, rmsd, x, usage_vec, t_string] = readData(filename)

	limit_manual = min(int(args.limit * 60 * 500), len(y_raw))
	if limit_manual == 0:
		limit_manual = len(y_raw)

	limit = int(60 * 500 * (limit_manual // (60*500) ))

	raw_length = len(y_raw)
	y_raw = y_raw[:limit]
	lowpass_y = lowpass(y_raw,10.0)

	y = []
	for i in range(0,len(y_raw)):
		y.append((y_raw[i]-lowpass_y[i]))

	print ('RMSD Stat (mean,std)', np.mean(rmsd), np.std(rmsd))
	print ('F Stat (mean,std)', np.mean(y), np.std(y))
	
	if sd_th != 1:
		rmsd_cutoff = np.float64(sd_th)
	else:
		rmsd_cutoff = np.float64(min(np.mean(rmsd)+0.6*np.std(rmsd), 2*np.mean(rmsd)))

	print ("rmsd-cutoff", rmsd_cutoff)

	peak_cutoff =  2*np.std(y)
	print ("peak-cutoff", peak_cutoff)

	lp = 1.8
	print ("low-pass freq", lp)

	y = y[:limit]
	rmsd = rmsd[:limit]
	x = x[:limit]
	usage_vec = usage_vec[:limit]
	t_string = t_string[:limit]

	if args.extra_file != '':
		[hf_x, hf_y] = extractHFFromFile(args.extra_file)

		print(len(hf_x), len(hf_y), raw_length)

		hf_x = hf_x[:limit]
		hf_y = hf_y[:limit]

		hf_x.append(x[-1] + 100000)
		hf_y.append(2)
		f_hf = interp1d(hf_x, hf_y)

	delta = 4000	
	index = thresholdRMSD(rmsd, rmsd_cutoff, delta)

	peak_clean_range = int(500 // 4)

	if (np.max(y) != np.min(y)):
		peak_value = 0
		y_squared = np.abs(np.array(y))
		th = 10.0*np.std(np.abs(np.array(y)))
		#th = 5000
		n_peaks_vec = []
		th_vec = []
		peak_old = -10
		d_th = min(th*0.005,10)
		count = 0
		while peak_value < 3.5:
			count += 1
			peaks = findPeaks(y_squared, th)
			peaks_cleaned = cleanPeaks(peaks, peak_clean_range)
			peaks_indexed = np.array(index) * np.array(peaks_cleaned)
			peak_value = 500 * np.mean(np.array(peaks_indexed)) / np.mean(np.array(index))

			if math.isnan(peak_value):
				print("NAN issue")
				sys.exit(-1)

			print ("\tpeak value", peak_value, th)
			n_peaks_vec.append(peak_value)
			th_vec.append(th)

			if count > 200 and peak_value<0.2:
				d_th = 1
			if count > 400 and peak_value<0.2:
				d_th = 0.2
			
			if peak_value > 1 and abs(peak_value-peak_old) < 0.00001:
				break

			peak_old = peak_value
			th -= d_th

		plt.plot(range(0,len(n_peaks_vec)), n_peaks_vec)
			
		slope = []
		for i in range(0,len(n_peaks_vec)-1):
			if n_peaks_vec[i] > 1:
				slope.append(abs(n_peaks_vec[i]-n_peaks_vec[i+1]))
			else:
				slope.append(100)
		th = th_vec[slope.index(min(slope))]

		ind = th_vec.index(th)
		plt.plot(ind, n_peaks_vec[ind], 'o')
		plt.title("Optimal TH = " + str(th))
		plt.savefig(pdf_file_peakfitting)

	else:
		print ('no values')
		th = 1
		sys.exit(-1)

	peaks = findPeaks(np.abs(np.array(y)), th)
	peaks_cleaned = cleanPeaks(peaks, peak_clean_range)

	w = 100
	ave_peak = np.zeros(w)
	ave_peak2 = np.zeros(w)
	n = 0
	m = 0
	plt.figure()
	for i in range(w,len(peaks_cleaned)-w):
		if peaks_cleaned[i] == 1:
			ave_peak += np.array(y[i-w//2:i+w//2])
			ave_peak2 += np.array(y[i-w//2:i+w//2])
			n += 1
			m += 1
			if m == 50:
				plt.plot(ave_peak2/m, 'b', alpha=0.1)
				ave_peak2 = np.zeros(w)
				m = 0
	ave_peak /= n
	plt.savefig(filename.rsplit(".",1)[0] + "_all_beat.pdf")

	beat_ave_val = np.mean(ave_peak)

	pdf_file_beat = filename.rsplit(".",1)[0] + "_ave_beat.pdf"
	plt.figure()
	plt.plot(ave_peak)
	plt.plot(w//2, ave_peak[w//2], 'or', alpha=0.6)

	if ave_peak[w//2] < beat_ave_val:
		plt.title('going for min')
		corr_mode = 0
	else:
		plt.title('going for max')
		corr_mode = 1

	plt.savefig(pdf_file_beat)

	#peaks_cleaned = peak_correction(peaks_cleaned, y)

	if manual_peak_mode != None:
		corr_mode = manual_peak_mode
		corr_offset = manual_offset
	else:
		corr_offset = 0

	peaks_cleaned = peak_correction2(peaks_cleaned, y, corr_mode, corr_offset)

	peaks_indexed = np.array(index) * np.array(peaks_cleaned)

	w = 100
	ave_peak = np.zeros(w)
	m = 0
	plt.figure()
	for i in range(w,len(peaks_indexed)-w):
		if peaks_indexed[i] == 1:
			ave_peak += np.array(y[i-w//2:i+w//2])
			m += 1
			if m == 50:
				plt.plot(ave_peak/m, 'r', alpha=0.1)
				ave_peak = np.zeros(w)
				m = 0

	print ("saving corr_beat to:", ''.join(filename.split(".")[:-1]) + "_all_clean_beat.pdf")
	plt.savefig(filename.rsplit(".",1)[0] + "_all_clean_beat.pdf")

	if writeBeat:
		print('beat done')
		sys.exit(0)

		
	passed = lowpass(y, lp)
	passed_filtered = np.clip(passed,-3, 3) * np.array(index)

	beat_int_x_filt = []
	beat_int_y_filt = []
	for i in range(0,len(peaks_indexed)):
		if peaks_indexed[i] == 1:
			dx = 1000
			for j in range(1,1000):
				if  (i+j)<len(peaks_indexed) and peaks_indexed[i+j] == 1:
					dx = j
					break
			if dx < 1000:
				beat_int_x_filt.append(x[i])
				beat_int_y_filt.append(1.0/(float(dx)/500))

	beat_int_x = []
	beat_int_y = []
	for i in range(0,len(peaks_cleaned)):
		if peaks_cleaned[i] == 1:
			dx = 1000
			for j in range(1,1000):
				if  (i+j)<len(peaks_cleaned) and peaks_cleaned[i+j] == 1:
					dx = j
					break
			if dx < 1000:
				beat_int_x.append(x[i])
				beat_int_y.append(1.0/(float(dx)/500))

	int_x_peak, int_y_peak = getBeatVectorsForInt(x, peaks_cleaned)
	int_x_peak.append(int_x_peak[-1]+10000)
	int_y_peak.append(int_y_peak[-1])
	f_peak = interp1d(int_x_peak, int_y_peak)

	qrs_beats = detectQRS(y)
	qrs_peaks = np.zeros((1,len(y)), dtype=int)[0]
	for q in qrs_beats:
         qrs_peaks[q] = 1

	#qrs_peaks = peak_correction(list(qrs_peaks), y)

	int_x_qrs, int_y_qrs = getBeatVectorsForInt(x, qrs_peaks)
	int_x_qrs.append(int_x_qrs[-1] + 10000)
	int_y_qrs.append(int_y_qrs[-1])
	f_qrs = interp1d(int_x_qrs, int_y_qrs)

	print ("preparing vecs")
	
	v1 = f_peak(x)
	print ("\t1/6 done")

	v2 = f_qrs(x)
	print ("\t2/6 done")

	v3 = f_hf(x)
	print ("\t3/6 done")

	v4 = getDiffVec(v1, v2, v3)
	print ("\t4/6 done")

	v5 = signal.medfilt(v4,201)
	print ("\t5/6 done")

	v6 = getUsageVec(v5, 0.2, 2000)
	print ("\t6/6 done")

	print ("vecs ready")

	usage_final = generateDefensiveUsageVector(usage_vec, index, v6)
	jump_vec = generateIBIJumpVector(usage_vec, index, v6, delta, v5)

	qrs_peaks_indexed = np.array(usage_final) * np.array(qrs_peaks)
	peaks_indexed = np.array(usage_final) * np.array(peaks_indexed)

	if writePDF:
		print ("writing plots")
		pdf_pages = PdfPages(pdf_file)	

		dx = 60 * 500
		for i in range(0, limit, dx):
			sys.stdout.write("\r\t\t%d%%" % float((100.0*i)/limit) )
			sys.stdout.flush()
        
			index_low = i
			index_high = min(i+dx, limit)
        
			plt.rc('font', **{'size':'5'})
        
			fig, axes = plt.subplots(nrows=6, ncols=1)
			fig.tight_layout(pad=3.0, w_pad=4.0, h_pad=3.0)
        
			ax1 = plt.subplot(7, 1, 1)
			ax1.xaxis.set_major_formatter(plt.FuncFormatter(HMS))
			ax1.set_ylim([-2*np.std(np.array(y)), 2*np.std(np.array(y))])
			plt.plot(x[index_low:index_high], np.array(y[index_low:index_high]))
        
			plt.title("Raw signal")
        	
			ax = plt.subplot(7, 1, 2, sharex=ax1)
			ax.set_ylim([-0.1, 5])
			plt.plot(x[index_low:index_high], rmsd[index_low:index_high])
			plt.title("RMSD")
        
			ax = plt.subplot(7, 1, 3, sharex=ax1)
			ax.set_ylim([-1.5, 1.5])
			plt.plot(x[index_low:index_high], peaks_cleaned[index_low:index_high])
			plt.plot(x[index_low:index_high], -np.array(qrs_peaks[index_low:index_high]), 'r')
			plt.title("Beats")
        
			ax = plt.subplot(7, 1, 4, sharex=ax1)
			ax.set_ylim([-1, 3])
			#plt.plot(x[index_low:index_high], passed[index_low:index_high])
			plt.plot(x[index_low:index_high], v5[index_low:index_high])
			plt.grid()
        		#plt.title("Low-pass @1.5Hz")
			plt.title("IBI Similarity Measurement")
        
			ax = plt.subplot(7, 1, 5, sharex=ax1)
			ax.set_ylim([-3, 3])
			plt.plot(x[index_low:index_high], peaks_indexed[index_low:index_high], alpha=0.4)
			plt.plot(x[index_low:index_high], -np.array(qrs_peaks_indexed[index_low:index_high]), 'r', alpha=0.4)
			plt.plot(x[index_low:index_high], passed_filtered[index_low:index_high], 'g')
			plt.title("Beats and low-passed signal")
        	
			ax = plt.subplot(7, 1, 6, sharex=ax1)
			ax.set_ylim([-4, 8])
			plt.plot(x[index_low:index_high], np.array(usage_final[index_low:index_high]) + 6*np.ones(dx), linewidth=2, label='Final Usage')
			plt.plot(x[index_low:index_high], np.array(v6[index_low:index_high]) + 4*np.ones(dx), label='IBI Usage Vec')
			plt.plot(x[index_low:index_high], np.array(index[index_low:index_high]) + 2*np.ones(dx), label='EMG Usage Vector')
			plt.plot(x[index_low:index_high], usage_vec[index_low:index_high], label='Movie Usage Vector')
			#plt.plot(x[index_low:index_high], jump_vec[index_low:index_high])
			plt.legend(loc=4, fontsize=5, ncol=4)
			plt.grid()
			plt.title("Data quality indicator")
        
			#beat_int_filt_low = [n for n, j in enumerate(beat_int_x_filt) if j>(i/500)]
			#beat_int_filt_high = [n for n, j in enumerate(beat_int_x_filt) if j<(i/500) + dx/500]
        
			#beat_int_low = [n for n, j in enumerate(beat_int_x) if j>(i/500)]
			#beat_int_high = [n for n, j in enumerate(beat_int_x) if j<(i/500) + dx/500]
      
			#if len(beat_int_filt_low)>0 and len(beat_int_filt_high)>0:
			#	index_low_filt = beat_int_filt_low[0]
			#	index_high_filt = beat_int_filt_high[-1]
			#	index_low = beat_int_low[0]
			#	index_high = beat_int_high[-1]
			#	ax = plt.subplot(7, 1, 7, sharex=ax1)
			#	ax.set_ylim([0, 4])
			#	plt.plot(beat_int_x[index_low:index_high], beat_int_y[index_low:index_high], 'o', alpha=0.25)
			#	plt.plot(beat_int_x_filt[index_low_filt:index_high_filt], beat_int_y_filt[index_low_filt:index_high_filt], 'ob')
			#	plt.grid()
			#	plt.title("Beat frequencies")

			ax = plt.subplot(7, 1, 7, sharex=ax1)
			ax.set_ylim([-1, 5])
			#ax.set_ylim([2.2, 2.6])
			plt.plot(x[index_low:index_high], v1[index_low:index_high], alpha=0.4, label='IBInt-Peak')
			plt.plot(x[index_low:index_high], v2[index_low:index_high], 'g', alpha=0.4, label='IBInt-QRS')
			
			if args.extra_file != '':
				plt.plot(x[index_low:index_high], v3[index_low:index_high], 'r', alpha=0.4, label='IBInt-CardioTach')

			plt.grid()
			plt.legend(loc=4, fontsize=5, ncol=3)

			pdf_pages.savefig(fig)
			fig.clf()
			plt.clf()
        
		pdf_pages.close()
		print ("\r\ndone")

	if args.extra_file != '':
		#pass
		writeCSV(csv_out_file, t_string, usage_vec, index, y, rmsd, peaks_indexed, peaks_cleaned, qrs_peaks, passed, passed_filtered, f_peak(x), f_qrs(x), v6, usage_final, jump_vec, f_hf(x))
	else:
		writeCSV(csv_out_file, t_string, usage_vec, index, y, rmsd, peaks_indexed, peaks_cleaned, qrs_peaks, passed, passed_filtered, f_peak(x), f_qrs(x), v6, usage_final, jump_vec)

	found_peaks = []
	for i in range(0,len(peaks_indexed)):
		if peaks_cleaned[i] == 1:
			found_peaks.append(i)

	data = []
	diff_sum = 0
	for i in range(0,len(found_peaks)-1):
		diff = (found_peaks[i+1]-found_peaks[i])/500.0
		data.append([i, diff, diff_sum, usage_final[found_peaks[i]]])
		diff_sum += diff

	x = []; y = [];
	for d in data:
		x.append(d[2])
		y.append(d[1])

	plt.figure()
	plt.plot(x,y)
	plt.title("IBI All")
	pdf_file = filename.rsplit(".",1)[0] + "_ibi_all.pdf"
	plt.savefig(pdf_file)

	x = []; y = [];
	ind = 0
	for d in data:
		if d[3] == 1:
			x.append(ind)
			ind += 1
			y.append(d[1])

	w = 5
	t = 0.05
	outlay_x = []
	outlay_y = []
	for i in range(w,len(y)-w):
		mean = np.mean(y[i-w:i+w+1])
		q = abs(y[i]-mean)
		if q > t:
			outlay_x.append(i)
			outlay_y.append(y[i])
	
	print (len(outlay_x), "outlayers")

	plt.figure()
	plt.plot(x,y)
	plt.plot(outlay_x, outlay_y, 'or', alpha=0.6)
	plt.title("IBI Used")
	pdf_file = filename.rsplit(".",1)[0] + "_ibi_cleaned.pdf"
	plt.savefig(pdf_file)

	for ind in outlay_x:
		val_sum = 0
		n = 0
		for i in range(ind-5, ind+5+1):
			if (i!=ind) and (i not in outlay_x):
				val_sum += y[i]
				n += 1
		if n>0:
			y[ind] = float(val_sum) / float(n)

	plt.figure()
	plt.plot(x,y)
	plt.title("IBI Used")
	pdf_file = filename.rsplit(".",1)[0] + "_ibi_corrected.pdf"
	plt.savefig(pdf_file)

	print("Processing successfully done")



