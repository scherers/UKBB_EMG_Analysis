#!/usr/bin/env python

import sys
import os.path

if sys.version_info[0] < 3:
    import Tkinter as Tk
    import ttk
    from tkFileDialog import askopenfilename, asksaveasfilename
    from tkMessageBox import showerror, askyesno
else:
    import tkinter as Tk
    from tkinter import ttk
    from tkinter.filedialog import askopenfilename, asksaveasfilename
    from tkinter.messagebox import showerror, askyesno

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import SpanSelector, MultiCursor
matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np


class DataManager:
    header_columns = []
    file_length = 0
    header_line = ''
    data_lines = []
    
    plot_columns = []
    usage_columns = []
    jump_columns = []
    
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
        
    def getPageNumByX(self, x_val):
        return 1+int(np.floor(x_val/60))
    
    def __init__(self, filename):
        self.infile = open(filename)
        self.header_line = self.infile.readline()
        self.header_columns = self.header_line.rstrip('\r\n').split(",")

        for item in self.header_columns:
            if item == '"time"':
                continue
            elif item.startswith('"usage_'):
                self.usage_columns.append(item)
            elif item.startswith('"jump_'):
                self.jump_columns.append(item)
            else:
                self.plot_columns.append(item)
        

    def readData(self, plot_cols, usage_cols, progress_label, usg_manual_col, jmp_col):
        # Zerobased counting of columns (First column = 0)
        del_usage_manual_col = -1
        if '"usage_manual"' in self.header_columns:
            del_usage_manual_col = self.header_columns.index('"usage_manual"')
        plot_data = []
        for i in range(0,len(plot_cols)):
            plot_data.append([])
        usage_data = []
        for i in range(0,len(usage_cols)):
            usage_data.append([])
        usage_manual = []
        time_axis = []
        jmp_idxs = []
        count = 0
        self.data_lines = []
        for l in self.infile:
            tmp = l.rstrip('\r\n').split(",")
            for p in range(0,len(plot_cols)):
                plot_data[p].append(float(tmp[plot_cols[p]]))
            for u in range(0,len(usage_cols)):
                usage_data[u].append(int(tmp[usage_cols[u]]))
            if usg_manual_col >= 0:
                usage_manual.append(int(tmp[usg_manual_col]))
            else:
                usage_manual.append(1)
            time_axis.append(count/500.0)
            if jmp_col >= 0 and int(tmp[jmp_col]) == 1:
                jmp_idxs.append(count/500.0)
            count += 1
            if del_usage_manual_col > -1:
                del tmp[del_usage_manual_col]
                self.data_lines.append(','.join(tmp)+'\r\n')
            else:
                self.data_lines.append(l)
            if count%(500*60) == 0:
                progress_label['text'] = 'loading ... ({} minutes loaded)'.format(count/500/60)
                progress_label.update()

        self.file_length = len(self.data_lines)

        self.pages = range(0, self.file_length, self.dx)
        self.num_pages = len(self.pages)

        print("reading done")
        progress_label['text'] = 'loading done, preparing plots ...'
        progress_label.update()
        return [time_axis, plot_data, usage_data, np.array(usage_manual), jmp_idxs]

    def writeCSV(self, filename, manual_usg):
        outfile = open(filename, 'w')
    
        print("writing output")
        
        header = '{},"usage_manual"\r\n'.format(self.header_line.replace(',"usage_manual"', '').rstrip('\r\n'))
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
    ax4 = []
    
    ylim_ax1 = []
    ylim_ax2 = []
    ylim_ax3 = []
    
    #data
    x = []
    usage_manual = []   
    
    #debug
    show_debug_msg = False
    
    #autosave (-1 for disable)
    autosave = 30

    
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
        
        self.root.wm_title("EMG-Visualization-Tool: {}".format(filename))
        self.dataMgr = DataManager(filename)     
        
        self.csv_out_file = ''.join(filename.split(".")[:-1]) + "_usage.csv"
        
        while os.path.isfile(self.csv_out_file):
            print('File "{}" already exists'.format(self.csv_out_file))
            if askyesno('Overwrite File?', 'File "{}" already exists. Overwrite?'.format(self.csv_out_file)):
                break
            else:
                new_out_file = asksaveasfilename(initialfile=self.csv_out_file)
                if new_out_file:
                    self.csv_out_file = new_out_file
                else:
                    sys.exit(0)
                    
        print("csv-out-file: {}".format(self.csv_out_file))
        
        self.configFrame = Tk.Frame(self.root, height=500, width=400)

        Tk.Label(self.configFrame, text="\r\nPlot 1").pack()
        self.plot1_select = ttk.Combobox(self.configFrame, values=self.dataMgr.plot_columns, state="readonly")
        self.plot1_select.pack()
        
        if '"f"' in self.dataMgr.plot_columns:
            self.plot1_select.current(self.dataMgr.plot_columns.index('"f"'))
        else:
            self.plot1_select.current(0)
        

        Tk.Label(self.configFrame, text="Plot 2").pack()
        self.plot2_select = ttk.Combobox(self.configFrame, values=self.dataMgr.plot_columns, state="readonly")
        self.plot2_select.pack()

        if '"rmsd"' in self.dataMgr.plot_columns:
            self.plot2_select.current(self.dataMgr.plot_columns.index('"rmsd"'))
        else:
            self.plot2_select.current(0)

        
        Tk.Label(self.configFrame, text="Plot 3").pack()
        self.plot3_select = ttk.Combobox(self.configFrame, values=self.dataMgr.plot_columns, state="readonly")
        self.plot3_select.pack()

        if '"beat"' in self.dataMgr.plot_columns:
            self.plot3_select.current(self.dataMgr.plot_columns.index('"beat"'))
        else:
            self.plot3_select.current(0)
        
        
        Tk.Label(self.configFrame, text="\r\nUsage Plot").pack()

        self.usage_plots = {}

        for usg in self.dataMgr.usage_columns:
            if not usg == '"usage_manual"':
                chkbx_var = Tk.IntVar()
                chkbx_var.set(1)
                usg_check = ttk.Checkbutton(self.configFrame, text=usg, variable=chkbx_var)
                usg_check.pack()
                self.usage_plots[usg] = chkbx_var

        Tk.Label(self.configFrame, text="\r\nLoad/copy \"usage_manual\" from").pack()
        self.usg_man_select = ttk.Combobox(self.configFrame, values=self.dataMgr.usage_columns, state="readonly")
        self.usg_man_select.pack()
        
        if '"usage_manual"' in self.dataMgr.usage_columns:
            self.usg_man_select.current(self.dataMgr.usage_columns.index('"usage_manual"'))
        elif '"usage_total"' in self.dataMgr.usage_columns:
            self.usg_man_select.current(self.dataMgr.usage_columns.index('"usage_total"'))
        else:
            self.usg_man_select.current(0)


        Tk.Label(self.configFrame, text="\r\nJump Column").pack()

        if len(self.dataMgr.jump_columns) == 0:
            self.jmp_select = ttk.Combobox(self.configFrame, values=['--none--'], state="readonly")
        else: 
            self.jmp_select = ttk.Combobox(self.configFrame, values=self.dataMgr.jump_columns, state="readonly")
        self.jmp_select.current(0)        
        self.jmp_select.pack()

        
        Tk.Label(self.configFrame, text="").pack()

        button_continue = Tk.Button(self.configFrame, text='Continue', command=self._CfgContinue)
        button_continue.pack()
        
        Tk.Label(self.configFrame, text="").pack()
        self.loading_label = Tk.Label(self.configFrame, text="")
        self.loading_label.pack()
        
        self.configFrame.pack()

        self.visualizerFrame = Tk.Frame(self.root)
        
        
        # Figure with Subplots
        self.fig = plt.figure(figsize=(17,8), dpi=80, tight_layout=True)
        gs = gridspec.GridSpec(4,1, height_ratios=[3,2,2,1])
        self.ax1 = plt.subplot(gs[0])
        self.ax2 = plt.subplot(gs[1], sharex=self.ax1)
        self.ax3 = plt.subplot(gs[2], sharex=self.ax1)
        self.ax4 = plt.subplot(gs[3], sharex=self.ax1)   
        
        canvas = FigureCanvasTkAgg(self.fig, master=self.visualizerFrame)
        canvas.show()
        canvas.mpl_connect('key_press_event', self.on_key_event)
        canvas.mpl_connect('button_press_event', self.on_button_event)
    
        # GUI Elements
        self.mode_label = Tk.Label(self.visualizerFrame, text="Mode: ADD", background="green")
    
        self.progress_label = Tk.Label(self.visualizerFrame, text="Page {}/{}".format(self.dataMgr.current_page, self.dataMgr.num_pages))
    
        button_prev = Tk.Button(master=self.visualizerFrame, text='Prev', command=self._prev)
        button_next = Tk.Button(master=self.visualizerFrame, text='Next', command=self._next)
    
        button_zoom_in = Tk.Button(master=self.visualizerFrame, text='Zoom In', command=self._zoom_in)
        button_zoom_out = Tk.Button(master=self.visualizerFrame, text='Zoom Out', command=self._zoom_out)    
    
        button_add = Tk.Button(master=self.visualizerFrame, text='Add', command=self._add)
        button_del = Tk.Button(master=self.visualizerFrame, text='Del', command=self._del)
        button_save = Tk.Button(master=self.visualizerFrame, text='Save', command=self._save)
        button_quit = Tk.Button(master=self.visualizerFrame, text='Quit', command=self._quit)
        
        # Selection
        self.multi_cursor = MultiCursor(self.fig.canvas, (self.ax1, self.ax2, self.ax3, self.ax4), useblit=True, horizOn=False, vertOn=True, color='g', lw=1)
        self.span1 = SpanSelector(self.ax1, self.onselectAdd, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='green') )
        self.span2 = SpanSelector(self.ax2, self.onselectAdd, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='green') )
        self.span3 = SpanSelector(self.ax3, self.onselectAdd, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='green') )
        self.span4 = SpanSelector(self.ax4, self.onselectAdd, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='green') )

        # GUI Layout
        button_zoom_in.grid(column=0, row=0)
        button_zoom_out.grid(column=1, row=0)
        button_prev.grid(column=3, row=0)
        self.progress_label.grid(column=4, row=0)
        button_next.grid(column=5, row=0)
        canvas.get_tk_widget().grid(column=0, row=1, columnspan=6)
        canvas._tkcanvas.grid(column=0, row=1, columnspan=6)
        button_add.grid(column=0, row=2)
        button_del.grid(column=1, row=2)
        self.mode_label.grid(column=2, row=2, columnspan=2)
        button_save.grid(column=4, row=2)
        button_quit.grid(column=5, row=2)
    
        Tk.mainloop()       


    def _CfgContinue(self):
        self.loading_label['text'] = 'loading ...'
        self.loading_label.update()
        
        self.plot_names = [self.plot1_select.get(),
                      self.plot2_select.get(),
                      self.plot3_select.get()]
        
        self.plot_cols = []              
        for pn in self.plot_names:
            self.plot_cols.append(self.dataMgr.header_columns.index(pn))

        self.usage_names = []
        self.usage_cols = []
        for k,v in self.usage_plots.items():
            if v.get() == 1:
                self.usage_names.append(k)
                self.usage_cols.append(self.dataMgr.header_columns.index(k))
        
        if self.usg_man_select.get() in self.dataMgr.header_columns:
            usg_manual_col = self.dataMgr.header_columns.index(self.usg_man_select.get())
        else:
            usg_manual_col = -1
            
        if self.jmp_select.get() in self.dataMgr.jump_columns:
            jmp_col = self.dataMgr.header_columns.index(self.jmp_select.get())
        else:
            jmp_col = -1
        
        [self.x, self.plot_data, self.usage_data, self.usage_manual, self.jump_positions] = self.dataMgr.readData(self.plot_cols,self.usage_cols,self.loading_label, usg_manual_col, jmp_col)

        self.configFrame.pack_forget()
        self.visualizerFrame.pack()

        print("displaying plots")
        
        self.ylim_ax1 = self.calc_ylims(self.plot_data[0])
        self.ylim_ax2 = self.calc_ylims(self.plot_data[1])
        self.ylim_ax3 = self.calc_ylims(self.plot_data[2])
        
        self.loadPage(1)


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
        
        
    def calc_ylims(self, data):
        [cnt, edge] = np.histogram(data, 25)
        s = len(data)
        thres = 0.975*s
        i = 0
        j = len(cnt)-1
        while True:
            if cnt[i] < cnt[j]:
                if s-cnt[i] < thres:
                    break
                else:
                    s -= cnt[i]
                    i += 1
            else:
                if s-cnt[j] < thres:
                    break
                else:
                    s -= cnt[j]
                    j -= 1
        
        return [min(0, edge[i]), max(1, edge[j+1])]


    def plotPage(self):
        index_low = self.dataMgr.low_Idx()
        index_high = self.dataMgr.high_Idx()
        
        if self.show_debug_msg:
            print("index_low: {} | index_high: {}".format(index_low, index_high))
    
        self.ax1.clear()
        self.ax1.xaxis.set_major_formatter(plt.FuncFormatter(self.HMS))
        self.ax1.plot(self.x[index_low:index_high], np.array(self.plot_data[0][index_low:index_high]))
        self.ax1.set_ylim(self.ylim_ax1)
        self.ax1.set_title(self.plot_names[0])
        
        self.ax2.clear()
        self.ax2.plot(self.x[index_low:index_high], self.plot_data[1][index_low:index_high])
        self.ax2.set_ylim(self.ylim_ax2)
        self.ax2.set_title(self.plot_names[1])
        
        self.ax3.clear()
        self.ax3.plot(self.x[index_low:index_high], self.plot_data[2][index_low:index_high])
        self.ax3.set_ylim(self.ylim_ax3)
        self.ax3.set_title(self.plot_names[2])
        
        self.plotUsages()
        
        self.fig.canvas.draw()
        
    
    def plotUsages(self):
        index_low = self.dataMgr.low_Idx()
        index_high = self.dataMgr.high_Idx()
        
        self.ax4.clear()
        self.ax4.set_ylim(0,len(self.usage_names)+2)
        self.ax4.set_yticks([], minor=False)     
        
        colors = ['#483D8B', '#228B22', '#B22222', '#8A2BE2', '#808000', '#FF4500', '#DA70D6', '#FFA500']
                
        self.ax4.fill_between(self.x[index_low:index_high], 0, (len(self.usage_names)+2)*np.array(self.usage_manual[index_low:index_high]), facecolor='#7fbf7f', edgecolor='None')
        
        self.ax4.plot(self.jump_positions, [len(self.usage_names)+1]*len(self.jump_positions), 'r*')        
        
        for u in range(0,len(self.usage_data)):
            self.ax4.fill_between(self.x[index_low:index_high], u, u+np.array(self.usage_data[u][index_low:index_high]), facecolor=colors[u], edgecolor='None')

        patches = [mpatches.Patch(color='green', alpha=0.5, label='usage_manual')]
        
        for i in range(0,len(self.usage_names)):
            patches.append(mpatches.Patch(color=colors[i], label=self.usage_names[i]))

        plt.legend(bbox_to_anchor=(0., 1.0, 1., .102), loc=3,
           ncol=5, mode="expand", borderaxespad=0., handles=patches)

        self.ax1.set_xlim(self.dataMgr.low_Xlimit(),self.dataMgr.high_Xlimit())

    
    def onselectAdd(self, xmin, xmax):
        minIdx = max(0, round(xmin*500))
        maxIdx = min(self.dataMgr.file_length-1, round(xmax*500))
        if self.show_debug_msg:
            print("ADD: xmin: {} | xmax: {}".format(xmin, xmax))
            print("ADD: minIdx: {} | maxIdx: {}".format(minIdx, maxIdx))

        self.usage_manual[minIdx:maxIdx] = 1
        self.plotUsages()
        self.fig.canvas.draw()


    def onselectDel(self, xmin, xmax):
        minIdx = max(0, round(xmin*500))
        maxIdx = min(self.dataMgr.file_length-1, round(xmax*500))
        if self.show_debug_msg:
            print("DEL: xmin: {} | xmax: {}".format(xmin, xmax))
            print("DEL: minIdx: {} | maxIdx: {}".format(minIdx, maxIdx))
        
        self.usage_manual[minIdx:maxIdx] = 0
        self.plotUsages()
        self.fig.canvas.draw()


    def onselectZoom(self, xmin, xmax):
        if self.show_debug_msg:
            print("ZOOM: xmin: {} | xmax: {}".format(xmin, xmax))
        
        self.plotUsages()
        self.ax1.set_xlim(xmin,xmax)
        self.fig.canvas.draw()
    
    
    def on_key_event(self, event):
        if self.show_debug_msg:
            print('you pressed %s'%event.key)
        if event.key == 'a':
            self._prev()
        elif event.key == 'd':
            self._next()
        elif event.key == 'w':
            self._add()
        elif event.key == 's':
            self._del()
        elif event.key == 'q':
            self._zoom_in()
        elif event.key == 'e':
            self._zoom_out()
        elif event.key == 'r':
            self._save()
        elif event.key == 'x':
            self._prevJump()
        elif event.key == 'c':
            self._nextJump()
        elif event.key == 'left':
            self._prev()
        elif event.key == 'right':
            self._next()
        elif event.key == 'up':
            self._add()
        elif event.key == 'down':
            self._del()
            

    def on_button_event(self, event):
        if self.show_debug_msg:
            print('you clicked %s'%event.button)
        if event.button == 3:   #right mouse button
            self._zoom_out()
        elif event.button == 2: #middle mouse button (scroll wheel)
            self._zoom_in()


    def _quit(self):
        self.root.quit()     # stops mainloop
        self.root.destroy()  # this is necessary on Windows to prevent
                        # Fatal Python Error: PyEval_RestoreThread: NULL tstate
    
    def _prev(self):
        if self.show_debug_msg:
            print('_prev()')
        if self.dataMgr.current_page > 1:
            self.loadPage(self.dataMgr.current_page-1)

    def _next(self):
        if self.show_debug_msg:
            print('next()')
        if self.dataMgr.current_page < self.dataMgr.num_pages:
            self.loadPage(self.dataMgr.current_page+1)


    def _prevJump(self):
        if self.show_debug_msg:
            print('_prevJump()')
        if self.dataMgr.current_page > 1:
            for p in reversed(self.jump_positions):
                num = self.dataMgr.getPageNumByX(p)
                if num < self.dataMgr.current_page:
                    self.loadPage(num)
                    break


    def _nextJump(self):
        if self.show_debug_msg:
            print('nextJump()')
        if self.dataMgr.current_page < self.dataMgr.num_pages:
            for p in self.jump_positions:
                num = self.dataMgr.getPageNumByX(p)
                if num > self.dataMgr.current_page:
                    self.loadPage(num)
                    break
            

    def loadPage(self, page_num):
        if self.autosave > -1 and page_num % self.autosave == 0:
            if self.show_debug_msg:
                print('autosaving on page {}'.format(page_num))
            self._save()
        self.dataMgr.current_page = min(max(1, page_num), self.dataMgr.num_pages)
        if self.show_debug_msg:
            print('loadPage(): {}'.format(self.dataMgr.current_page))
        self.plotPage()
        self.progress_label["text"] ="Page {}/{}".format(self.dataMgr.current_page, self.dataMgr.num_pages)


    def _add(self):
        if self.show_debug_msg:
            print('_add()')
        if float(matplotlib.__version__[0:3])>=1.4:
            self.multi_cursor.disconnect()
        self.multi_cursor = MultiCursor(self.fig.canvas, (self.ax1, self.ax2, self.ax3, self.ax4), useblit=True, horizOn=False, vertOn=True, color='g', lw=1)
        self.span1.disconnect_events()
        self.span2.disconnect_events()
        self.span3.disconnect_events()
        self.span4.disconnect_events()
        self.span1 = SpanSelector(self.ax1, self.onselectAdd, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='green') )
        self.span2 = SpanSelector(self.ax2, self.onselectAdd, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='green') )
        self.span3 = SpanSelector(self.ax3, self.onselectAdd, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='green') )
        self.span4 = SpanSelector(self.ax4, self.onselectAdd, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='green') )

        self.mode_label['text'] = 'Mode: ADD'
        self.mode_label['bg'] = 'green'
        self.fig.canvas.draw()


    def _del(self):
        if self.show_debug_msg:
            print('_del()')
        if float(matplotlib.__version__[0:3])>=1.4:
            self.multi_cursor.disconnect()
        self.multi_cursor = MultiCursor(self.fig.canvas, (self.ax1, self.ax2, self.ax3, self.ax4), useblit=True, horizOn=False, vertOn=True, color='r', lw=1)
        self.span1.disconnect_events()
        self.span1.disconnect_events()
        self.span2.disconnect_events()
        self.span3.disconnect_events()
        self.span4.disconnect_events()
        self.span1 = SpanSelector(self.ax1, self.onselectDel, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='red') )
        self.span2 = SpanSelector(self.ax2, self.onselectDel, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='red') )
        self.span3 = SpanSelector(self.ax3, self.onselectDel, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='red') )
        self.span4 = SpanSelector(self.ax4, self.onselectDel, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='red') )

        self.mode_label['text'] = 'Mode: DEL'
        self.mode_label['bg'] = 'red'
        self.fig.canvas.draw()

  
    def _zoom_in(self):
        if self.show_debug_msg:
            print('_zoom_in()')
        if float(matplotlib.__version__[0:3])>=1.4:
            self.multi_cursor.disconnect()
        self.multi_cursor = MultiCursor(self.fig.canvas, (self.ax1, self.ax2, self.ax3, self.ax4), useblit=True, horizOn=False, vertOn=True, color='b', lw=1)
        self.span1.disconnect_events()
        self.span1.disconnect_events()
        self.span2.disconnect_events()
        self.span3.disconnect_events()
        self.span4.disconnect_events()
        self.span1 = SpanSelector(self.ax1, self.onselectZoom, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='blue') )
        self.span2 = SpanSelector(self.ax2, self.onselectZoom, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='blue') )
        self.span3 = SpanSelector(self.ax3, self.onselectZoom, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='blue') )
        self.span4 = SpanSelector(self.ax4, self.onselectZoom, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='blue') )

        self.mode_label['text'] = 'Mode: ZOOM'
        self.mode_label['bg'] = 'blue'
        self.fig.canvas.draw()


    def _zoom_out(self):
        if self.show_debug_msg:
            print('_zoom_out()')

        self.plotUsages()
        self.fig.canvas.draw()


    def _save(self):
        if self.show_debug_msg:
            print('_save()')
        plt.text(20, 20, 'autosaving...', fontsize=46, color='r', weight='bold', ha='center', va='top')
        self.fig.canvas.draw()
        self.dataMgr.writeCSV(self.csv_out_file, self.usage_manual)


if __name__ == "__main__":
    print('Number of arguments: {} arguments.'.format(len(sys.argv)))
    print('Argument List: {}'.format(str(sys.argv)))

    if (len(sys.argv)) > 1:
         gui = VisualizerWindow(sys.argv[1])
    else:
         gui = VisualizerWindow([])


