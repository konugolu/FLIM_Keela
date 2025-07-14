
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 12:22:32 2025

@author: keela
"""

import time
import ctypes as ct
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ctypes import byref
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from scipy.optimize import minimize
from scipy.signal import convolve
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os


# -------------- PicoHarp Measurement Class --------------
class PicoHarpGUI:
    def __init__(self, root):
        #Initialise GUI
        self.root = root
        self.running = False
        self.time_values = []
        self.counts = []

        self.frame = ttk.Frame(root)
        self.frame.pack(fill="both", expand=True)

        # Top horizontal layout
        top_frame = ttk.Frame(self.frame)
        top_frame.pack(fill="x", pady=10)
        
        # Bottom horizontal layout
        bottom_frame = ttk.Frame(self.frame)
        bottom_frame.pack(fill="both", expand=True, pady=10)

        # Left: Input + Buttons
        control_frame = ttk.Frame(top_frame)
        control_frame.grid(row=0, column=0, sticky="nw", padx=10)

        tk.Label(control_frame, text="Sync Offset:").grid(row=0, column=0, sticky="w")
        self.sync_offset_entry = tk.Entry(control_frame, width=8)
        self.sync_offset_entry.grid(row=0, column=1)
        self.sync_offset_entry.insert(0, "31000")

        tk.Label(control_frame, text="Measurement Time (s):").grid(row=1, column=0, sticky="w")
        self.meas_time_entry = tk.Entry(control_frame, width=8)
        self.meas_time_entry.grid(row=1, column=1)
        self.meas_time_entry.insert(0, "1")

        tk.Button(control_frame, text="Start", command=self.start_measurement).grid(row=2, column=0, pady=2, sticky="ew")
        tk.Button(control_frame, text="Stop", command=self.stop_measurement).grid(row=2, column=1, pady=2, sticky="ew")
        
        self.enable_fit = tk.BooleanVar()
        tk.Checkbutton(control_frame, text="Enable Live Fitting", variable=self.enable_fit).grid(row=5, column=0, columnspan=2)
        
        # ROI and Fit Range Inputs
        tk.Label(control_frame, text="ROI Start (ns):").grid(row=6, column=0, sticky="w")
        self.roi_start_entry = tk.Entry(control_frame, width=8)
        self.roi_start_entry.grid(row=6, column=1)
        self.roi_start_entry.insert(0, "0")

        tk.Label(control_frame, text="ROI Stop (ns):").grid(row=7, column=0, sticky="w")
        self.roi_stop_entry = tk.Entry(control_frame, width=8)
        self.roi_stop_entry.grid(row=7, column=1)
        self.roi_stop_entry.insert(0, "30")

        tk.Label(control_frame, text="Fit Start (%):").grid(row=8, column=0, sticky="w")
        self.fit_start_entry = tk.Entry(control_frame, width=8)
        self.fit_start_entry.grid(row=8, column=1)
        self.fit_start_entry.insert(0, "80")

        tk.Label(control_frame, text="Fit Stop (%):").grid(row=9, column=0, sticky="w")
        self.fit_stop_entry = tk.Entry(control_frame, width=8)
        self.fit_stop_entry.grid(row=9, column=1)
        self.fit_stop_entry.insert(0, "1")


        self.count_rate = tk.StringVar()
        tk.Label(control_frame, textvariable=self.count_rate, fg="blue").grid(row=10, column=0, columnspan=2, pady=(10, 0))


        # Right: IRF Plot
        irf_plot_frame = ttk.Frame(top_frame)
        irf_plot_frame.grid(row=0, column=1, sticky="nsew", padx=10)
        tk.Button(irf_plot_frame, text="Load IRF", command=self.load_irf).pack(pady=(0, 5))
        tk.Label(irf_plot_frame, text="Instrument Response Function (IRF)").pack()
        self.fig_irf, self.ax_irf = plt.subplots(figsize=(4.5, 2.5))
        self.canvas_irf = FigureCanvasTkAgg(self.fig_irf, master=irf_plot_frame)
        self.canvas_irf.get_tk_widget().pack()


        # Bottom: Measurement Plot (left)
        meas_frame = ttk.Frame(bottom_frame)
        meas_frame.grid(row=0, column=0, padx=10)

        tk.Label(meas_frame, text="Live Measurement").pack()
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=meas_frame)
        self.canvas.get_tk_widget().pack()

        tk.Button(meas_frame, text="Save Data", command=self.save_data).pack(pady=5)

        # Bottom: Fit Plot (right)
        fit_frame = ttk.Frame(bottom_frame)
        fit_frame.grid(row=0, column=1, padx=10)

        tk.Label(fit_frame, text="Live Fit to IRF").pack()
        self.fig_fit, self.ax_fit = plt.subplots(figsize=(5, 3))
        self.canvas_fit = FigureCanvasTkAgg(self.fig_fit, master=fit_frame)
        self.canvas_fit.get_tk_widget().pack()

        self.fit_result_label = tk.Label(fit_frame, text="", fg="green")
        self.fit_result_label.pack()
        
        tk.Button(fit_frame, text="Save Fit Data", command=self.save_fit_data).pack(pady=5)



    def picoharp_measurement_rt(self, syncoffset, measurement_time_sec, update_interval=1):
        #Functions from PHlib library
        LIB_VERSION = "3.0"
        HISTCHAN = 65536
        MAXDEVNUM = 8
        MODE_HIST = 0
        FLAG_OVERFLOW = 0x0040

        offset = 0
        syncDivider = 8
        CFDZeroCross0 = 13
        CFDLevel0 = 231
        CFDZeroCross1 = 20
        CFDLevel1 = 189

        self.counts = (ct.c_uint * HISTCHAN)()
        resolution = ct.c_double()
        errorString = ct.create_string_buffer(b"", 40)
        phlib = ct.CDLL("phlib64.dll")

        def tryfunc(retcode, funcName):
            if retcode < 0:
                phlib.PH_GetErrorString(errorString, ct.c_int(retcode))
                raise RuntimeError(f"PH_{funcName} error {retcode} ({errorString.value.decode('utf-8')}).")

        dev_id = -1
        for i in range(MAXDEVNUM):
            retcode = phlib.PH_OpenDevice(ct.c_int(i), ct.create_string_buffer(b"", 8))
            if retcode == 0:
                dev_id = i
                break
        if dev_id == -1:
            raise RuntimeError("No device available.")

        tryfunc(phlib.PH_Initialize(ct.c_int(dev_id), ct.c_int(MODE_HIST)), "Initialize")
        tryfunc(phlib.PH_Calibrate(ct.c_int(dev_id)), "Calibrate")
        tryfunc(phlib.PH_SetSyncDiv(ct.c_int(dev_id), ct.c_int(syncDivider)), "SetSyncDiv")
        tryfunc(phlib.PH_SetInputCFD(ct.c_int(dev_id), ct.c_int(0), ct.c_int(CFDLevel0), ct.c_int(CFDZeroCross0)), "SetInputCFD")
        tryfunc(phlib.PH_SetInputCFD(ct.c_int(dev_id), ct.c_int(1), ct.c_int(CFDLevel1), ct.c_int(CFDZeroCross1)), "SetInputCFD")
        tryfunc(phlib.PH_SetBinning(ct.c_int(dev_id), ct.c_int(0)), "SetBinning")
        tryfunc(phlib.PH_SetOffset(ct.c_int(dev_id), ct.c_int(offset)), "SetOffset")
        tryfunc(phlib.PH_SetSyncOffset(ct.c_int(dev_id), ct.c_int(syncoffset)), "SetSyncOffset")
        tryfunc(phlib.PH_GetResolution(ct.c_int(dev_id), byref(resolution)), "GetResolution")

        tryfunc(phlib.PH_ClearHistMem(ct.c_int(dev_id), ct.c_int(0)), "ClearHistMem")

        time_per_bin_ns = resolution.value / 1000
        self.time_values = [i * time_per_bin_ns for i in range(HISTCHAN)]  # Store in self
        
        #Continue updating for live fitting (until 'Stop Measurement' pressed)
        def update_plot():
            if not self.running:
                return  

            tryfunc(phlib.PH_ClearHistMem(ct.c_int(dev_id), ct.c_int(0)), "ClearHistMem")  
            meas_time_ms = int(measurement_time_sec * 1000)  
            tryfunc(phlib.PH_StartMeas(ct.c_int(dev_id), ct.c_int(meas_time_ms)), "StartMeas")
            time.sleep(measurement_time_sec)  
            tryfunc(phlib.PH_GetHistogram(ct.c_int(dev_id), byref(self.counts), ct.c_int(0)), "GetHistogram")

            count_values = np.array([self.counts[i] for i in range(HISTCHAN)])

            self.count_rate.set(f"Count Rate: {np.sum(count_values) / measurement_time_sec:.2E} cps")

            self.ax.clear()
            self.ax.plot(self.time_values, count_values, label="Photon Counts", color='blue')
            self.ax.set_yscale("log")
            self.ax.set_xlim(0, 50)
            self.ax.legend()
            self.canvas.draw()

            tryfunc(phlib.PH_StopMeas(ct.c_int(dev_id)), "StopMeas")

            self.root.after(int(update_interval * 1000), update_plot)  
            
            if self.enable_fit.get() and hasattr(self, "irf_counts") and hasattr(self, "irf_time"):
                try:
                    self.perform_live_fit(np.array(self.time_values), count_values)
                except Exception as e:
                    print(f"Live fitting failed: {e}")


        return update_plot

    def start_measurement(self):
        self.running = True
        syncoffset = int(self.sync_offset_entry.get())
        measurement_time_sec = float(self.meas_time_entry.get())  
        update_plot = self.picoharp_measurement_rt(syncoffset, measurement_time_sec)
        self.root.after(100, update_plot)  

    def stop_measurement(self):
        self.running = False

    #Save live fit data
    def save_data(self):
        if not self.time_values or not self.counts:
            messagebox.showerror("Error", "No data to save!")
            return

        filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not filename:
            return

        with open(filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Time (ns)", "Counts"])
            for t, c in zip(self.time_values, self.counts):
                csv_writer.writerow([t, c])

        messagebox.showinfo("Success", f"Data saved to {filename}")
        
    def load_irf(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        try:
            irf_data = np.loadtxt(file_path, delimiter=",", skiprows=1)
            if irf_data.shape[1] != 2:
                raise ValueError("CSV should have two columns: time and intensity.")

            time_irf = irf_data[:, 0]
            counts_irf = irf_data[:, 1]
            
            self.irf_time = time_irf
            self.irf_counts = counts_irf


            self.ax_irf.clear()
            self.ax_irf.plot(time_irf, counts_irf, label="IRF", color='green')
            self.ax_irf.set_title("Instrument Response Function")
            self.ax_irf.set_xlabel("Time (ns)")
            self.ax_irf.set_ylabel("Counts")
            self.ax_irf.set_yscale("log")
            self.ax_irf.set_xlim(0, 100)

            self.ax_irf.legend()
            self.canvas_irf.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load IRF: {e}")
            
    # Live fitting
    def perform_live_fit(self, t, counts_obs):
        from scipy.optimize import minimize
        from scipy.signal import convolve

        irf_counts = self.irf_counts
        t_irf = self.irf_time

        # Background subtraction (same as in LifetimeFittingGUI)
        t_bg_start, t_bg_stop = 0, 3
        bg_indices = (t >= t_bg_start) & (t <= t_bg_stop)
        background_value_irf = np.mean(irf_counts[bg_indices])
        background_value_obs = np.mean(counts_obs[bg_indices])
        irf_counts_bg = irf_counts - background_value_irf
        obs_counts_bg = counts_obs - background_value_obs

        # Moving average 
        window_size = 1
        irf_ma = np.convolve(irf_counts_bg, np.ones(window_size) / window_size, mode='same')
        obs_ma = np.convolve(obs_counts_bg, np.ones(window_size) / window_size, mode='same')

        # Define ROI range from user inputs
        roi_start_ns = float(self.roi_start_entry.get())
        roi_stop_ns = float(self.roi_stop_entry.get())
        fit_start_perc = float(self.fit_start_entry.get()) / 100
        fit_stop_perc = float(self.fit_stop_entry.get()) / 100


        roi_start_idx = np.searchsorted(t, roi_start_ns)
        roi_stop_idx = np.searchsorted(t, roi_stop_ns)

        t_roi = t[roi_start_idx:roi_stop_idx]
        obs_roi = obs_ma[roi_start_idx:roi_stop_idx]
        irf_roi = irf_ma[roi_start_idx:roi_stop_idx]

        max_count = np.max(obs_roi)
        lower_limit = fit_stop_perc * max_count
        upper_limit = fit_start_perc * max_count

        try:
            start_idx = np.where(obs_roi > upper_limit)[0][0]
            stop_idx = np.where(obs_roi[start_idx:] < lower_limit)[0][0] + start_idx
        except IndexError:
            return  # Avoid crashing if no points meet the criteria

        def decay(t_roi, tau):
            
            if tau <= 0 or np.isnan(tau) or np.isinf(tau):
                return np.ones_like(t_roi) * np.nan
            return np.exp(-t_roi / tau)

            return np.exp(-t_roi / tau)

        def model(t_roi, irf_roi, tau):
            decay_curve = decay(t_roi, tau)
            conv = convolve(decay_curve, irf_roi, mode='full')
            return conv[:len(t_roi)]

        def chi2(params):
            tau, shift = params
            shift_int = int(round(shift))
            irf_shifted = np.roll(irf_roi, shift_int)
            pred = model(t_roi, irf_shifted, tau)
            pred *= np.sum(obs_roi) / np.sum(pred)

            pred_crop = pred[start_idx:stop_idx]
            obs_crop = obs_roi[start_idx:stop_idx]
            pred_crop[pred_crop <= 0] = 1e-10

            return np.sum((obs_crop - pred_crop) ** 2 / (pred_crop + 1e-10))

        result = minimize(
            chi2, [1.0, 0.0],
            bounds=[(0.01, 15), (-200, 200)],
            method="Powell"
        )

        fitted_tau = result.x[0]
        shift = int(round(result.x[1]))
        irf_shifted = np.roll(irf_roi, shift)
        fitted_counts = model(t_roi, irf_shifted, fitted_tau)
        fitted_counts *= np.sum(obs_roi) / np.sum(fitted_counts)
        time_shift_ns = shift * (t[1] - t[0])  
        


        # Plot
        self.ax_fit.clear()
        self.ax_fit.plot(t_roi, obs_roi, 'o', markersize=3, alpha=0.6, label="Observed")
        self.ax_fit.plot(t_roi, fitted_counts, '-', label=f"Fitted (τ = {fitted_tau:.2f} ns)", linewidth=2)
        self.ax_fit.axvline(t_roi[start_idx], color='red', linestyle='--')
        self.ax_fit.axvline(t_roi[stop_idx], color='red', linestyle='--')
        self.ax_fit.set_yscale("log")
        self.ax_fit.set_xlim(roi_start_ns, roi_stop_ns)
        self.ax_fit.set_ylim(1, 1000)
        self.ax_fit.legend()
        self.canvas_fit.draw()
        
         # Calculate Pearson Chi-Squared and Reduced Pearson Chi-Squared
        observed_counts_cropped = obs_roi[start_idx:stop_idx]
        fitted_counts_cropped = fitted_counts[start_idx:stop_idx]
        
        pearson_chi2 = np.sum((observed_counts_cropped - fitted_counts_cropped) ** 2 / fitted_counts_cropped)
        degrees_of_freedom = len(observed_counts_cropped) - 2  # Adjust for cropped range
        reduced_pearson_chi2 = pearson_chi2 / degrees_of_freedom

       
        # Store data for saving
        self.last_fit_time = t_roi
        self.last_fit_observed = obs_roi
        self.last_fit_irf = irf_shifted
        self.last_fit_fitted = fitted_counts
        self.last_fit_tau = fitted_tau
        self.last_fit_shift = time_shift_ns
        self.last_fit_chi2 = reduced_pearson_chi2

        # Fit result
        self.fit_result_label.config(
            text=f"τ: {fitted_tau:.2f} ns | Shift: {time_shift_ns:.2f} ns | χ²: {reduced_pearson_chi2:.2f}"
        )
        
    def save_fit_data(self):
        try:
            # Ensure latest fit values exist
            if not hasattr(self, "last_fit_tau") or not hasattr(self, "last_fit_shift") or not hasattr(self, "last_fit_chi2"):
                messagebox.showerror("Error", "No fit data available.")
                return

            filename = filedialog.asksaveasfilename(defaultextension=".csv",
                                                    filetypes=[("CSV files", "*.csv")])
            if not filename:
                return

            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Time (ns)", "Counts", "IRF Used", "Fitted Counts", "Tau (ns)", "Shift (ns)", "Chi^2"])

                for i in range(len(self.last_fit_time)):
                    writer.writerow([
                        self.last_fit_time[i],
                        self.last_fit_observed[i],
                        self.last_fit_irf[i],
                        self.last_fit_fitted[i],
                        self.last_fit_tau if i == 0 else "",
                        self.last_fit_shift if i == 0 else "",
                        self.last_fit_chi2 if i == 0 else ""
                    ])

            messagebox.showinfo("Saved", f"Fit data saved to:\n{filename}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save fit data: {e}")




       
# -------------- Lifetime Fitting Class --------------
class LifetimeFittingGUI:
    def __init__(self, root):
        self.root = root
        self.root.winfo_toplevel().title("Fluorescence Lifetime Fitting")



        # File selection
        self.irf_file = None
        self.obs_file = None

        tk.Label(root, text="IRF File:").grid(row=0, column=0, padx=5, pady=5)
        self.irf_entry = tk.Entry(root, width=50, state="readonly")
        self.irf_entry.grid(row=0, column=1, padx=5, pady=5)
        tk.Button(root, text="Browse", command=self.upload_irf).grid(row=0, column=2, padx=5, pady=5)

        tk.Label(root, text="Observed File:").grid(row=1, column=0, padx=5, pady=5)
        self.obs_entry = tk.Entry(root, width=50, state="readonly")
        self.obs_entry.grid(row=1, column=1, padx=5, pady=5)
        tk.Button(root, text="Browse", command=self.upload_obs).grid(row=1, column=2, padx=5, pady=5)

        # ROI selection
        tk.Label(root, text="ROI Start (ns):").grid(row=2, column=0, padx=5, pady=5)
        self.roi_start_entry = tk.Entry(root, width=10)
        self.roi_start_entry.grid(row=2, column=1, padx=5, pady=5)
        self.roi_start_entry.insert(0, "0")  

        tk.Label(root, text="ROI Stop (ns):").grid(row=3, column=0, padx=5, pady=5)
        self.roi_stop_entry = tk.Entry(root, width=10)
        self.roi_stop_entry.grid(row=3, column=1, padx=5, pady=5)
        self.roi_stop_entry.insert(0, "30")  

        # Fitting range selection
        tk.Label(root, text="Fit Start (%):").grid(row=4, column=0, padx=5, pady=5)
        self.fit_start_entry = tk.Entry(root, width=10)
        self.fit_start_entry.grid(row=4, column=1, padx=5, pady=5)
        self.fit_start_entry.insert(0, "80")  

        tk.Label(root, text="Fit Stop (%):").grid(row=5, column=0, padx=5, pady=5)
        self.fit_stop_entry = tk.Entry(root, width=10)
        self.fit_stop_entry.grid(row=5, column=1, padx=5, pady=5)
        self.fit_stop_entry.insert(0, "1")  

        # Run button
        self.run_button = tk.Button(root, text="Run Fitting", command=self.run_fitting, state=tk.DISABLED)
        self.run_button.grid(row=6, column=0, columnspan=3, pady=10)

        # Plot 
        self.figure, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().grid(row=7, column=0, columnspan=3, pady=10)
        
        # Save Plot button (initially disabled)
        self.save_button = tk.Button(root, text="Save Plot", command=self.save_plot, state=tk.DISABLED)
        self.save_button.grid(row=8, column=0, columnspan=3, pady=10)
        
        # Save CSV button (initially disabled)
        self.save_csv_button = tk.Button(root, text="Save CSV", command=self.save_fit_csv, state=tk.DISABLED)
        self.save_csv_button.grid(row=10, column=0, columnspan=3, pady=(0, 5))
        
        # Results Display (move it just below)
        self.result_label = tk.Label(root, text="Results will be displayed here", fg="blue", font=("Arial", 12, "bold"))
        self.result_label.grid(row=11, column=0, columnspan=3, pady=10)



    def upload_irf(self):
        self.irf_file = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if self.irf_file:
            self.irf_entry.config(state="normal")
            self.irf_entry.delete(0, tk.END)
            self.irf_entry.insert(0, os.path.basename(self.irf_file))
            self.irf_entry.config(state="readonly")

            # Load and plot IRF data
            df = pd.read_csv(self.irf_file)
            self.t_irf = df["Time (ns)"].values
            self.irf_counts = df["Counts"].values

            self.ax.clear()
            self.ax.plot(self.t_irf, self.irf_counts, label="IRF", color='blue')
            self.ax.set_yscale("log")
            self.ax.set_xlim(0, 100)
            self.ax.set_title("Uploaded Data")
            self.ax.legend()
            self.canvas.draw()

            self.check_ready()

    def upload_obs(self):
        self.obs_file = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if self.obs_file:
            self.obs_entry.config(state="normal")
            self.obs_entry.delete(0, tk.END)
            self.obs_entry.insert(0, os.path.basename(self.obs_file))
            self.obs_entry.config(state="readonly")

            # Load and plot Observed data
            df = pd.read_csv(self.obs_file)
            self.t_obs = df["Time (ns)"].values
            self.observed_counts = df["Counts"].values

            self.ax.clear()
            self.ax.plot(self.t_obs, self.observed_counts, label="Observed", color='orange')
            if hasattr(self, 't_irf') and hasattr(self, 'irf_counts'):  # If IRF is already uploaded, overlay both plots
                self.ax.plot(self.t_irf, self.irf_counts, label="IRF", color='blue')

                self.ax.set_yscale("log")
                self.ax.set_xlim(0, 100)
                self.ax.set_title("Uploaded Data")
                self.ax.legend()
                self.canvas.draw()

                self.check_ready()


    def check_ready(self):
        if self.irf_file and self.obs_file:
            self.run_button.config(state=tk.NORMAL)
            
    def save_plot(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG Files", "*.png"),
                                                                ("PDF Files", "*.pdf"),
                                                                ("JPEG Files", "*.jpg"),
                                                                ("All Files", "*.*")])
        if file_path:
            self.figure.savefig(file_path, dpi=300)
            messagebox.showinfo("Save Plot", "Plot saved successfully!")
            
    def save_fit_csv(self):
        try:
            if not hasattr(self, "last_fit_tau"):
                messagebox.showerror("Error", "No fitted data to save.")
                return
    
            file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                     filetypes=[("CSV Files", "*.csv")])
            if not file_path:
                return
    
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Time (ns)", "Observed Counts", "IRF Used", "Fitted Counts", "Tau (ns)", "Shift (ns)", "Chi^2"])
                for i in range(len(self.last_fit_time)):
                    writer.writerow([
                        self.last_fit_time[i],
                        self.last_fit_observed[i],
                        self.last_fit_irf[i],
                        self.last_fit_fitted[i],
                        self.last_fit_tau if i == 0 else "",
                        self.last_fit_shift if i == 0 else "",
                        self.last_fit_chi2 if i == 0 else ""
                    ])
            messagebox.showinfo("Saved", f"Fit data saved to:\n{file_path}")
    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save CSV: {e}")


    def run_fitting(self):
        df = pd.read_csv(self.irf_file)
        dff = pd.read_csv(self.obs_file)

        t = df["Time (ns)"].values
        observed_counts = dff["Counts"].values
        irf_counts = df["Counts"].values

        # Background subtraction 
        t_bg_start, t_bg_stop = 0, 3  
        bg_indices = (t >= t_bg_start) & (t <= t_bg_stop)
        background_value_irf = np.mean(irf_counts[bg_indices])
        background_value_obs = np.mean(observed_counts[bg_indices])
        irf_counts_bg = irf_counts - background_value_irf
        observed_counts_bg = observed_counts - background_value_obs

        # Apply moving average filter 
        window_size = 1  
        irf_counts_ma = np.convolve(irf_counts_bg, np.ones(window_size) / window_size, mode='same')
        observed_counts_ma = np.convolve(observed_counts_bg, np.ones(window_size) / window_size, mode='same')

        # Get ROI inputs
        roi_start_ns = float(self.roi_start_entry.get())
        roi_stop_ns = float(self.roi_stop_entry.get())
        fit_start_perc = float(self.fit_start_entry.get()) / 100  
        fit_stop_perc = float(self.fit_stop_entry.get()) / 100    

        roi_start_idx = np.searchsorted(t, roi_start_ns)
        roi_stop_idx = np.searchsorted(t, roi_stop_ns)

        t_roi = t[roi_start_idx:roi_stop_idx]
        observed_counts_roi = observed_counts_ma[roi_start_idx:roi_stop_idx]
        irf_counts_roi = irf_counts_ma[roi_start_idx:roi_stop_idx]

        max_count = np.max(observed_counts_roi)
        lower_limit = fit_stop_perc * max_count
        upper_limit = fit_start_perc * max_count

        start_idx = np.where(observed_counts_roi > upper_limit)[0][0]
        stop_idx = np.where(observed_counts_roi[start_idx:] < lower_limit)[0][0] + start_idx

        def fluorescence_decay(t_roi, tau):
            
            if tau <= 0 or np.isnan(tau) or np.isinf(tau):
                return np.ones_like(t_roi) * np.nan
            return np.exp(-t_roi / tau)


        def convolved_decay(t_roi, irf_counts_roi, tau):
            ideal_decay = fluorescence_decay(t_roi, tau)
            convolved = convolve(ideal_decay, irf_counts_roi, mode="full")
            return convolved[:len(t_roi)]

        def chi_squared_residuals(params, t_roi, observed_counts_roi, irf_counts_roi, start_idx, stop_idx):
            tau, shift = params
            shift_int = int(round(shift))
            irf_counts_shifted = np.roll(irf_counts_roi, shift_int)
            
            
            
            predicted_counts = convolved_decay(t_roi, irf_counts_shifted, tau)
            predicted_counts *= np.sum(observed_counts_roi) / np.sum(predicted_counts)
            
            # Store the best-fit values for later retrieval
            chi_squared_residuals.best_fit = predicted_counts
            chi_squared_residuals.best_irf_shifted = irf_counts_shifted  # Store shifted IRF
            
            
            # Crop for fitting range
            observed_counts_cropped = observed_counts[start_idx:stop_idx]
            predicted_counts_cropped = predicted_counts[start_idx:stop_idx]

            # Avoid division by zero
            predicted_counts_cropped[predicted_counts_cropped <= 0] = 1e-10  

            chi_squared_value = np.sum(((observed_counts_cropped - predicted_counts_cropped) ** 2) / (predicted_counts_cropped + 1e-10))
            return chi_squared_value
        
        result = minimize(
            chi_squared_residuals,
            [1, 0],
            args=(t_roi, observed_counts_roi, irf_counts_roi, start_idx, stop_idx),
            bounds=[(0, 15), (-500, 500)],
            method="Powell"
        )

        # Extract fitted parameters
        fitted_tau = result.x[0]
        shift_int = int(round(result.x[1]))  
        time_shift_ns = shift_int * (t[1] - t[0])  
        
        # Use the stored best-fit values instead of recomputing
        fitted_counts = chi_squared_residuals.best_fit  
        irf_counts_shifted = chi_squared_residuals.best_irf_shifted 

        self.ax.clear()
        self.ax.plot(t_roi, observed_counts_roi, 'o', label="Observed (ROI)", color = "cornflowerblue", markersize=3, alpha=0.6)
        self.ax.plot(t_roi, irf_counts_roi, '-', color = "darkgreen", markersize=3, alpha=0.6)
        self.ax.plot(t_roi, fitted_counts, '-', label=f"Fitted (τ = {fitted_tau:.2f} ns)", color = "darkorange", linewidth=2)
        self.ax.axvline(t_roi[start_idx], color='red', linestyle='--', label=f"Start Index ({t_roi[start_idx]:.2f} ns)")
        self.ax.axvline(t_roi[stop_idx], color='red', linestyle='--', label=f"Stop Index ({t_roi[stop_idx]:.2f} ns)")
        self.ax.set_yscale("log")
        self.ax.set_ylim(1, 10000)
        self.ax.set_title("Fluorescence Decay Fit")
        self.ax.set_xlabel("Time (ns)")
        self.ax.set_ylabel("Counts")
        self.ax.legend()
        self.canvas.draw()
        
        self.save_button.config(state=tk.NORMAL)  # Enable the save button after plotting
        self.save_csv_button.config(state=tk.NORMAL)




        
        # Calculate Pearson Chi-Squared and Reduced Pearson Chi-Squared
        observed_counts_cropped = observed_counts[start_idx:stop_idx]
        fitted_counts_cropped = fitted_counts[start_idx:stop_idx]
        
        pearson_chi2 = np.sum((observed_counts_cropped - fitted_counts_cropped) ** 2 / fitted_counts_cropped)
        degrees_of_freedom = len(observed_counts_cropped) - 2  # Adjust for cropped range
        reduced_pearson_chi2 = pearson_chi2 / degrees_of_freedom

        self.result_label.config(text=f"Fitted Lifetime (τ): {fitted_tau:.2f} ns | Shift: {time_shift_ns:.2f} ns | χ²: {reduced_pearson_chi2:.2f}")
        
        # Store data for export
        self.last_fit_time = t_roi
        self.last_fit_observed = observed_counts_roi
        self.last_fit_irf = irf_counts_shifted
        self.last_fit_fitted = fitted_counts
        self.last_fit_tau = fitted_tau
        self.last_fit_shift = time_shift_ns
        self.last_fit_chi2 = reduced_pearson_chi2



# -------------- TMF8828 Raspberry Pi Class --------------
class TMF8828RaspberryPiGUI:
    def __init__(self, root):
        self.root = root
        self.root.winfo_toplevel().title("TMF8828 Raspberry Pi Measurement")

        self.frame = ttk.Frame(root)
        self.frame.pack(fill="both", expand=True)

        # Placeholder for future implementation
        tk.Label(self.frame, text="TMF8828 Raspberry Pi Measurement will be implemented here.").pack(pady=20)
        tk.Button(self.frame, text="Back to Main", command=self.root.destroy).pack(pady=10)

    
# -------------- Main Application --------------
class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fluorescence Lifetime & PicoHarp")

        notebook = ttk.Notebook(root)
        self.picoharp_tab = ttk.Frame(notebook)
        self.lifetime_tab = ttk.Frame(notebook)
        self.tmf8828_rasp_tab = ttk.Frame(notebook)

        notebook.add(self.picoharp_tab, text="PicoHarp Measurement")
        notebook.add(self.lifetime_tab, text="Lifetime Fitting")
        notebook.add(self.tmf8828_rasp_tab, text="TMF8828 Raspberry Pi Measurement")
        notebook.pack(expand=True, fill="both")

        PicoHarpGUI(self.picoharp_tab)
        LifetimeFittingGUI(self.lifetime_tab)
        TMF8828RaspberryPiGUI(self.tmf8828_rasp_tab)


#Run app
root = tk.Tk()
app = MainApp(root)
root.mainloop()