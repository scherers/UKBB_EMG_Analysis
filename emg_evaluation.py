import sys

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
from numpy.fft import fft, ifft, fftshift
import copy
from scipy.interpolate import interp1d
from scipy import signal

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
	hours = seconds / 3600
	seconds -= 3600 * hours
	minutes = seconds / 60
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
	i = 0
	index = np.zeros((1,len(rmsd_data)), dtype=int)[0]
	while i < (len(rmsd_data)):
		if (i>0) and (np.fmod(i,1000) < 0.001):
			sys.stdout.write("\r\t\t%d%%" % float((100.0*i)/len(rmsd_data)) )
			sys.stdout.flush()
		if rmsd_data[i] < cutoff:
			index[i] = 1
			i += 1
		else:
			index[max(i-delta, 0):i] = 0
			i += delta
	print ('\ndone')
	return index


def findPeaks(data, index, lower_bound, upper_bound):
	cleaned_data = np.array(data) * np.array(index)
	peaks = []
	for i in range(0,len(cleaned_data)):
		if cleaned_data[i] > lower_bound:
			peaks.append(1)
		else:
			peaks.append(0)
	return peaks


def cleanPeaks(peaks, window):
	i = 0
	result = copy.deepcopy(peaks)
	while i < len(result):
		if result[i] > 0:
			for j in range(1,peak_clean_range):
				if i+j < len(result):
					result[i+j] = 0
			i += peak_clean_range
		else:
			i += 1
	return result


def detectQRS(emg):
	## QRS-Detection
	# Based on DF1 (adapted to sampling rate) from Paper
	# "A Comparison of the Noise Sensitivity of Nine QRS Detection Algorithms"
	# Friesen et al., IEEE 1990
	# CAUTION: Hardcoded for a sampling frequency of 500Hz!

	# differentiator with 62.5 Hz notch filter
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
         j = np.nonzero(y1[i+1:i+bound] < thresDOWN)[0][0]
         if j:
             qrsflag = 1
             k = np.nonzero(y1[i+j:i+bound] > thresUP)[0][0]
             if k:
                 l = np.nonzero(y1[i+j+k:i+bound] < thresDOWN)[0][0]
                 if l:
                     if np.nonzero(y1[i+j+k+l:i+bound] > thresUP)[0]:
                         qrsflag = 0;

         if qrsflag > 0:
             qrs.append(i)
	
	return qrs

def getTimeString(t):
	seconds = int(t)
	hours = seconds / 3600
	seconds -= 3600 * hours
	minutes = seconds / 60
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
		if (i>0) and (np.fmod(i,1000) < 0.001):
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
		#tmp.append(abs(v2[i]-v3[i]))
		tmp.append(abs(v1[i]-v3[i]))
		tmp = sorted(tmp)
		result.append(tmp[0])
	return result

def getUsageVec(vec_in, th, delta):
	result = list(np.ones(len(vec_in)))
	for i in range(0,len(vec_in)):
		if vec_in[i] > th:
			for j in range(max(0,i-int(1.2*delta)),min(i+int(1.2*delta),len(vec_in))):
				result[j] = 0
	return result

def generateDefensiveUsageVector(movie_vec, rmsd_vec, bib_vec):
	result = []
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
			while result[i] == 1:
				result[i] = 0
				i += 1
		i += 1

	result2 = list(np.zeros(len(result)))

	d = 2000
	for i in ind:
		if np.max(diff_vec[i-d:i+d]) > 0.2:
			result2[int(i)] = 1
			minutes = (i/500)/60
			sec = int(60*((i/500)/60.0 - minutes))
			print ("\tjump-position found:", minutes, "mins", sec, "secs")
	print ("done")
	return result2

def peak_correction(peaks, f):
	dx = 25	
	ind = []
	for i in range(dx,len(peaks)-dx-1):
		if peaks[i] == 1:
			ind.append(i)
	
	ind_corr = []
	for i in ind:
		tmp = f[i-dx:i+dx+1]
		ind_corr.append(np.argmin(tmp)-25+i)

	result = list(np.zeros(len(peaks)))
	for i in ind_corr:
		print i
		result[i] = 1
	
	print len(f), len(result)
	return result


if __name__ == "__main__":
	#print 'Number of arguments:', len(sys.argv), 'arguments.'
	#print 'Argument List:', str(sys.argv)

	parser = argparse.ArgumentParser(description='Run EMG Analysis')
	parser.add_argument('filename', metavar='Input File', type=str, nargs=1, help='Name of the input file (CSV-File)')
	parser.add_argument('-l', '--limit', default=0, type=int, required=False)
	parser.add_argument('-p', '--print_pdf', action='store_true', default=False, dest='boolean_switch_pdf', help='Set a switch to true')
	parser.add_argument('-hf', '--extra-file', default='', type=str, required=False)
	args = parser.parse_args()

	if args.extra_file != '':
		print ("extra input file given:", args.extra_file)

	filename = args.filename[0]

	#Flag for writing PDFs (or just CSV)
	writePDF = args.boolean_switch_pdf

	if writePDF:
		pdf_file = ''.join(filename.split(".")[:-1]) + ".pdf"
		print ("pdf-file:", pdf_file)

	pdf_file_peakfitting = ''.join(filename.split(".")[:-1]) + "_peak_fitting.pdf"
	print ("pdf-file2:", pdf_file_peakfitting)

	csv_out_file = ''.join(filename.split(".")[:-1]) + "_eval.csv"
	print ("csv-out-file:", csv_out_file)
	
	[y_raw, rmsd, x, usage_vec, t_string] = readData(filename)

	if args.extra_file != '':
		[hf_x, hf_y] = extractHFFromFile(args.extra_file)
		f_hf = interp1d(hf_x, hf_y)
	

	limit_manual = min(int(args.limit * 60 * 500), len(y_raw))
	if limit_manual == 0:
		limit_manual = len(y_raw)

	limit = int(60 * 500 * (limit_manual / (60*500) ))

	y_raw = y_raw[:limit]
	lowpass_y = lowpass(y_raw,0.6)

	y = []
	for i in range(0,len(y_raw)):
		y.append((y_raw[i]-lowpass_y[i]))

	print ('RMSD Stat (mean,std)', np.mean(rmsd), np.std(rmsd))
	print ('F Stat (mean,std)', np.mean(y), np.std(y))
	
	rmsd_cutoff = min(np.mean(rmsd)+np.std(rmsd), 2*np.mean(rmsd))
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

	delta = 4000	
	index = thresholdRMSD(rmsd, rmsd_cutoff, delta)

	peak_clean_range = int(500 / 5)

	if abs(np.mean(y)) > 0:
		peak_value = 0
		th = 10.0*np.std(np.array(y)**2)
		#th = 5000
		n_peaks_vec = []
		th_vec = []
		count = 0
		peak_old = -10
		while peak_value < 4:
			peaks = findPeaks(np.array(y)**2, np.ones(len(y)), th, 10000)
			peaks_cleaned = cleanPeaks(peaks, peak_clean_range)
			peaks_indexed = np.array(index) * np.array(peaks_cleaned)
			peak_value = 500 * np.mean(np.array(peaks_indexed)) / np.mean(np.array(index))
			print ("\tpeak value", peak_value, th)
			n_peaks_vec.append(peak_value)
			th_vec.append(th)
			
			if peak_value > 1 and abs(peak_value-peak_old) < 0.001:
				break

			peak_old = peak_value
			count += 1
			th *= 0.9

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
		plt.savefig(pdf_file_peakfitting)

	else:
		print ('no values')
		th = 1

	peaks = findPeaks(np.array(y)**2, np.ones(len(y)), th, 10000)
	peaks_cleaned = cleanPeaks(peaks, peak_clean_range)

	peaks_cleaned = peak_correction(peaks_cleaned, y)

	peaks_indexed = np.array(index) * np.array(peaks_cleaned)
		
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
	f_peak = interp1d(int_x_peak, int_y_peak)

	qrs_beats = detectQRS(y)
	qrs_peaks = np.zeros((1,len(y)), dtype=int)[0]
	for q in qrs_beats:
         qrs_peaks[q] = 1

	int_x_qrs, int_y_qrs = getBeatVectorsForInt(x, qrs_peaks)
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
			plt.plot(x[index_low:index_high], -qrs_peaks[index_low:index_high], 'r')
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
			plt.plot(x[index_low:index_high], -qrs_peaks_indexed[index_low:index_high], 'r', alpha=0.4)
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
		writeCSV(csv_out_file, t_string, usage_vec, index, y, rmsd, peaks_indexed, peaks_cleaned, qrs_peaks, passed, passed_filtered, f_peak(x), f_qrs(x), v6, usage_final, jump_vec, f_hf(x))
	else:
		writeCSV(csv_out_file, t_string, usage_vec, index, y, rmsd, peaks_indexed, peaks_cleaned, qrs_peaks, passed, passed_filtered, f_peak(x), f_qrs(x), v6, usage_final, jump_vec)

	tmp = []
	for i in range(0,len(peaks_indexed)):
		if peaks_indexed[i] == 1:
			tmp.append(i)

	tmp2 = []
	for i in range(0,len(tmp)-1):
		if usage_final[tmp[i]] == 1 and usage_final[tmp[i+1]] == 1:
			diff = (tmp[i+1]-tmp[i])/500.0
			if diff < 5:
				tmp2.append(diff)
	
	tmp4 = []
	tmp5 = []
	tmp6 = []
	w = 5
	t = 0.05
	for i in range(w,len(tmp2)-w):
		mean = np.mean(tmp2[i-w:i+w+1])
		q = abs(tmp2[i]-mean)
		if q > t:
			tmp4.append(i)
			tmp5.append(tmp2[i])
		else:
			tmp6.append(tmp2[i])

	plt.figure()
	plt.plot(tmp2)

	print (len(tmp4), "outlayers found")
	plt.plot(tmp4, tmp5, 'or', alpha=0.6)
	pdf_file3 = ''.join(filename.split(".")[:-1]) + "_ibi.pdf"
	plt.savefig(pdf_file3)

	plt.figure()
	plt.plot(tmp6)
	pdf_file4 = ''.join(filename.split(".")[:-1]) + "_ibi_clean.pdf"
	plt.savefig(pdf_file4)

	
