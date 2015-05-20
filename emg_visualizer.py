#!/usr/bin/env python

# TODO not python3 ready yet

import sys
import os.path

if sys.version_info[0] < 3:
    import Tkinter as Tk
    from tkFileDialog import askopenfilename
    from tkMessageBox import showerror, askyesno
else:
    import tkinter as Tk
    from tkinter.filedialog import askopenfilename
    from tkinter.messagebox import showerror, askyesno

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import SpanSelector
matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np


class DataManager:
    header_columns = []
    file_length = 0
    header_line = ''
    data_lines = []
    
    dx = 60 * 500
    overlap = 500
    current_page = 1
    pages = []
    num_pages = [] 
    
    def low_Idx(self):
        return max(0, self.pages[self.current_page-1]-self.overlap)
        
    def high_Idx(self):
        return min(self.low_Idx()+self.dx+(2*self.overlap), self.file_length-1)
        
    def low_Xlimit(self):
        return (self.pages[self.current_page-1]/500)-self.overlap/500
        
    def high_Xlimit(self):
        return ((self.pages[self.current_page-1]+self.dx)/500)+self.overlap/500
    
    def __init__(self, filename):
        infile = open(filename)
        self.header_line = infile.readline()
        self.header_columns = self.header_line.split(",")
        self.data_lines = infile.readlines()
        self.file_length = len(self.data_lines)
        
        self.pages = range(0, self.file_length, self.dx)
        self.num_pages = len(self.pages)

    def readData(self, col1, col2, col_motion, col_rmsd):
        # Zerobased counting of columns (First column = 0)
        data1 = []
        data2 = []
        motion_usage = []
        rmsd_usage = []
        time_axis = []
        count = 0
        for l in self.data_lines:
            tmp = l.split(",")
            data1.append(float(tmp[col1]))
            data2.append(float(tmp[col2]))
            motion_usage.append(int(tmp[col_motion]))
            rmsd_usage.append(int(tmp[col_rmsd]))
            time_axis.append(count/500.0)
            count += 1

        print("reading done")
        return [time_axis, data1, data2, motion_usage, rmsd_usage]

    def writeCSV(self, filename, manual_usg):
        outfile = open(filename, 'w')
    
        print("writing output")
        
        header = '{},"manual_usage"\r\n'.format(self.header_line.rstrip('\r\n'))
        outfile.write(header)
    
        for i in range(0,self.file_length):
            if (i>0) and (np.fmod(i,1000) < 0.001):
                sys.stdout.write("\r\t\t%d%%" % float((100.0*i)/self.file_length) )
                sys.stdout.flush()
            result = self.data_lines[i].rstrip('\r\n') + ","
            result += str(manual_usg[i]) + "\r\n"
            outfile.write(result)
        outfile.close()
        print("\r\ndone")


class VisualizerWindow:
    root = []
    dataMgr = []
    
    #graphs
    fig = []
    ax1 = []
    ax2 = []
    ax3 = []    
    
    #data
    x = []
    y = []
    rmsd = []
    md_usage = []
    rmsd_usage = []
    manual_usage = []   
    
    #debug
    show_debug_msg = False
    
    
    def __init__(self, filename):        
        self.root = Tk.Tk()
        self.root.wm_title("EMG-Visualization-Tool")
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
     
        if not filename:
             filename = askopenfilename()
             if not filename:
                 sys.exit(0)
    
        while not os.path.isfile(filename):
             showerror("File not found", "Sorry, file '{}' was not found, try again".format(filename))
             filename = askopenfilename()
             if not filename:
                 sys.exit(0)
        
        self.dataMgr = DataManager(filename)        
        
        self.csv_out_file = ''.join(filename.split(".")[:-1]) + "_usage.csv"
        print("csv-out-file:".format(self.csv_out_file))
        
        emg_col = self.dataMgr.header_columns.index('"f"')
        rmsd_col = self.dataMgr.header_columns.index('"rmsd"')
        md_usage_col = self.dataMgr.header_columns.index('"usage_md"')
        rmsd_usage_col = self.dataMgr.header_columns.index('"usage_rmsd"')
        
        [self.x, self.y, self.rmsd, self.md_usage, self.rmsd_usage] = self.dataMgr.readData(emg_col,rmsd_col,md_usage_col,rmsd_usage_col)
    
        self.manual_usage = np.array(self.rmsd_usage[:])


        self.root.wm_title("EMG-Visualization-Tool: {}".format(filename))
        
        self.fig = plt.figure(figsize=(17,8), dpi=80, tight_layout=True)     
        canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        canvas.show()
        canvas.mpl_connect('key_press_event', self.on_key_event)
     
        print("displaying plots")
    
        # Graphs
        gs = gridspec.GridSpec(3,1,
                       height_ratios=[2,2,1]
                       )

        self.ax1 = plt.subplot(gs[0])
        self.ax2 = plt.subplot(gs[1], sharex=self.ax1)
        self.ax3 = plt.subplot(gs[2], sharex=self.ax1)
        self.plotPage()
    
        # GUI Elements
        self.progress_label = Tk.Label(self.root, text="Page {}/{}".format(self.dataMgr.current_page, self.dataMgr.num_pages))
    
        self.button_prev = Tk.Button(master=self.root, text='Prev', command=self._prev)
        self.button_next = Tk.Button(master=self.root, text='Next', command=self._next)
    
        self.button_add = Tk.Button(master=self.root, text='Add', command=self._add)
        self.button_del = Tk.Button(master=self.root, text='Del', command=self._del)
        self.button_save = Tk.Button(master=self.root, text='Save', command=self._save)
        self.button_quit = Tk.Button(master=self.root, text='Quit', command=self._quit)
        
        # Selection
        self.span1 = SpanSelector(self.ax1, self.onselectAdd, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='green') )
        self.span2 = SpanSelector(self.ax2, self.onselectAdd, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='green') )
        self.span3 = SpanSelector(self.ax3, self.onselectAdd, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='green') )

        # GUI Layout
        self.button_prev.grid(column=0, row=0)
        self.button_next.grid(column=3, row=0)
        self.progress_label.grid(column=1, row=0, columnspan=2)
        canvas.get_tk_widget().grid(column=0, row=1, columnspan=4)
        canvas._tkcanvas.grid(column=0, row=1, columnspan=4)
        self.button_add.grid(column=0, row=2)
        self.button_del.grid(column=1, row=2)
        self.button_save.grid(column=2, row=2)
        self.button_quit.grid(column=3, row=2)
    
        Tk.mainloop()       
        

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

    def plotPage(self):
        index_low = self.dataMgr.low_Idx()
        index_high = self.dataMgr.high_Idx()
        
        if self.show_debug_msg:
            print("index_low: {} | index_high: {}".format(index_low, index_high))
    
        self.ax1.clear()
        self.ax1.xaxis.set_major_formatter(plt.FuncFormatter(self.HMS))
        self.ax1.plot(self.x[index_low:index_high], np.array(self.y[index_low:index_high]))
        self.ax1.set_ylim([-2*np.std(np.array(self.y)), 2*np.std(np.array(self.y))])
        self.ax1.set_title("Raw signal")
        
        self.ax2.clear()
        self.ax2.plot(self.x[index_low:index_high], self.rmsd[index_low:index_high])
        self.ax2.set_ylim([-0.1, 5])
        self.ax2.set_title("RMSD")
        
        self.plotAx3()
        
        self.fig.canvas.draw()
        
    
    def plotAx3(self):
        index_low = self.dataMgr.low_Idx()
        index_high = self.dataMgr.high_Idx()

        if self.show_debug_msg:
            print("plotAx3: index_low: {} | index_high: {}".format(index_low, index_high))
        diff_vec = np.diff(self.manual_usage[index_low:index_high])
        starts = np.nonzero(diff_vec>0)[0]+index_low
        ends = np.nonzero(diff_vec<0)[0]+index_low           
        if self.manual_usage[index_low] == 1:
            starts = np.insert(starts, 0, 0)
        if self.manual_usage[index_high-1] == 1:
            ends = np.append(ends, index_high-1)
        
        diff_vecM = np.diff(self.md_usage[index_low:index_high])
        startsM = np.nonzero(diff_vecM>0)[0]+index_low
        endsM = np.nonzero(diff_vecM<0)[0]+index_low
        if self.md_usage[index_low] == 1:
            startsM = np.insert(startsM, 0, 0)
        if self.md_usage[index_high-1] == 1:
            endsM = np.append(endsM, index_high-1)
        
        diff_vecR = np.diff(self.rmsd_usage[index_low:index_high])
        startsR = np.nonzero(diff_vecR>0)[0]+index_low
        endsR = np.nonzero(diff_vecR<0)[0]+index_low
        if self.rmsd_usage[index_low] == 1:
            startsR = np.insert(startsR, 0, 0)
        if self.rmsd_usage[index_high-1] == 1:
            endsR = np.append(endsR, index_high-1)

        
        self.ax3.clear()
        self.ax3.set_title("Data quality indicator")
        self.ax3.set_ylim(0,1)
        
        for s in range(0, len(starts)):
            if self.show_debug_msg:
                print("%\t axvspan: {} - {}".format(float(starts[s])/500, float(ends[s])/500))
            self.ax3.axvspan(float(starts[s])/500, float(ends[s])/500, color='green', alpha=0.5)
            
        for r in range(0, len(startsR)):
            if self.show_debug_msg:
                print("%\t axvspanR: {} - {}".format(float(startsR[r])/500, float(endsR[r])/500))
            self.ax3.axvspan(float(startsR[r])/500, float(endsR[r])/500, ymin=0.1, ymax=0.2, color='green')
            
        for m in range(0, len(startsM)):
            if self.show_debug_msg:
                print("%\t axvspanM: {} - {}".format(float(startsM[m])/500, float(endsM[m])/500))
            self.ax3.axvspan(float(startsM[m])/500, float(endsM[m])/500, ymin=0, ymax=0.1, color='blue')
            
        self.ax1.set_xlim(self.dataMgr.low_Xlimit(),self.dataMgr.high_Xlimit())
        
    
    def onselectAdd(self, xmin, xmax):
        minIdx = max(0, round(xmin*500))
        maxIdx = min(self.dataMgr.file_length-1, round(xmax*500))
        if self.show_debug_msg:
            print("ADD: xmin: {} | xmax: {}".format(xmin, xmax))
            print("ADD: minIdx: {} | maxIdx: {}".format(minIdx, maxIdx))

        self.manual_usage[minIdx:maxIdx] = 1
        self.plotAx3()
        self.fig.canvas.draw()
        
    def onselectDel(self, xmin, xmax):
        minIdx = max(0, round(xmin*500))
        maxIdx = min(self.dataMgr.file_length-1, round(xmax*500))
        if self.show_debug_msg:
            print("DEL: xmin: {} | xmax: {}".format(xmin, xmax))
            print("DEL: minIdx: {} | maxIdx: {}".format(minIdx, maxIdx))
        
        self.manual_usage[minIdx:maxIdx] = 0
        self.plotAx3()
        self.fig.canvas.draw()
    
    
    def on_key_event(self, event):
        if self.show_debug_msg:
            print('you pressed %s'%event.key)
        if event.key == 's':
            self._save()
        elif event.key == 'left':
            self._prev()
        elif event.key == 'right':
            self._next()
        elif event.key == 'up':
            self._add()
        elif event.key == 'down':
            self._del()
    
    def _quit(self):
        self.root.quit()     # stops mainloop
        self.root.destroy()  # this is necessary on Windows to prevent
                        # Fatal Python Error: PyEval_RestoreThread: NULL tstate
    
    def _prev(self):
        if self.show_debug_msg:
            print('_prev()')
        if self.dataMgr.current_page > 1:
            self.dataMgr.current_page = max(1, self.dataMgr.current_page-1)
            if self.show_debug_msg:
                print(self.dataMgr.current_page)
            self.plotPage()
            self.progress_label["text"] ="Page {}/{}".format(self.dataMgr.current_page, self.dataMgr.num_pages)
    
    def _next(self):
        if self.show_debug_msg:
            print('next()')
        if self.dataMgr.current_page < self.dataMgr.num_pages:
            self.dataMgr.current_page = min(self.dataMgr.current_page+1, self.dataMgr.num_pages)
            if self.show_debug_msg:
                print(self.dataMgr.current_page)
            self.plotPage()
            self.progress_label["text"] ="Page {}/{}".format(self.dataMgr.current_page, self.dataMgr.num_pages)
    
    def _add(self):
        if self.show_debug_msg:
            print('_add()')
        self.span1 = SpanSelector(self.ax1, self.onselectAdd, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='green') )
        self.span2 = SpanSelector(self.ax2, self.onselectAdd, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='green') )
        self.span3 = SpanSelector(self.ax3, self.onselectAdd, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='green') )
        self.fig.canvas.draw()
    
    def _del(self):
        if self.show_debug_msg:
            print('_del()')
        self.span1 = SpanSelector(self.ax1, self.onselectDel, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='red') )
        self.span2 = SpanSelector(self.ax2, self.onselectDel, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='red') )
        self.span3 = SpanSelector(self.ax3, self.onselectDel, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='red') )
        self.fig.canvas.draw()
    
    def _save(self):
        if self.show_debug_msg:
            print('_save()')
        if os.path.isfile(self.csv_out_file):
            print('File "{}" already exists'.format(self.csv_out_file))
            if not askyesno('Overwrite File?', 'File "{}" already exists. Overwrite?'.format(self.csv_out_file)):
                return
        print('save {}'.format(self.csv_out_file))
        self.dataMgr.writeCSV(self.csv_out_file, self.manual_usage)



if __name__ == "__main__":
    print('Number of arguments: {} arguments.'.format(len(sys.argv)))
    print('Argument List: {}'.format(str(sys.argv)))

 
    if (len(sys.argv)) > 1:
         gui = VisualizerWindow(sys.argv[1])
    else:
         gui = VisualizerWindow([])


