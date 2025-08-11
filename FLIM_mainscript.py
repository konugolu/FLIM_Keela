
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 12:22:32 2025

@author: keela
"""

import queue
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

from TMF8828PiDataReader import DataReader
from contini_model_panel import ContiniModelPanel
from diffusion_equation.diffusion_equation import Contini1997
from diffusion_equation.fit import convolve_irf_with_model
from fitting_worker import FittingWorker
from test_contini_model import GEOMETRY


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
        meas_settings_frame = ttk.Frame(top_frame)
        meas_settings_frame.grid(row=0, column=0, sticky="nw", padx=10)

        tk.Label(meas_settings_frame, text="Sync Offset:").grid(row=0, column=0, sticky="w")
        self.sync_offset_entry = tk.Entry(meas_settings_frame, width=8)
        self.sync_offset_entry.grid(row=0, column=1)
        self.sync_offset_entry.insert(0, "31000")

        tk.Label(meas_settings_frame, text="Measurement Time (s):").grid(row=1, column=0, sticky="w")
        self.meas_time_entry = tk.Entry(meas_settings_frame, width=8)
        self.meas_time_entry.grid(row=1, column=1)
        self.meas_time_entry.insert(0, "1")

        tk.Button(meas_settings_frame, text="Start", command=self.start_measurement).grid(row=2, column=0, pady=2, sticky="ew")
        tk.Button(meas_settings_frame, text="Stop", command=self.stop_measurement).grid(row=2, column=1, pady=2, sticky="ew")
        
        self.enable_fit = tk.BooleanVar()
        tk.Checkbutton(meas_settings_frame, text="Enable Live Fitting", variable=self.enable_fit).grid(row=5, column=0, columnspan=2)
        
        # ROI and Fit Range Inputs
        tk.Label(meas_settings_frame, text="ROI Start (ns):").grid(row=6, column=0, sticky="w")
        self.roi_start_entry = tk.Entry(meas_settings_frame, width=8)
        self.roi_start_entry.grid(row=6, column=1)
        self.roi_start_entry.insert(0, "0")

        tk.Label(meas_settings_frame, text="ROI Stop (ns):").grid(row=7, column=0, sticky="w")
        self.roi_stop_entry = tk.Entry(meas_settings_frame, width=8)
        self.roi_stop_entry.grid(row=7, column=1)
        self.roi_stop_entry.insert(0, "30")

        tk.Label(meas_settings_frame, text="Fit Start (%):").grid(row=8, column=0, sticky="w")
        self.fit_start_entry = tk.Entry(meas_settings_frame, width=8)
        self.fit_start_entry.grid(row=8, column=1)
        self.fit_start_entry.insert(0, "80")

        tk.Label(meas_settings_frame, text="Fit Stop (%):").grid(row=9, column=0, sticky="w")
        self.fit_stop_entry = tk.Entry(meas_settings_frame, width=8)
        self.fit_stop_entry.grid(row=9, column=1)
        self.fit_stop_entry.insert(0, "1")


        self.count_rate = tk.StringVar()
        tk.Label(meas_settings_frame, textvariable=self.count_rate, fg="blue").grid(row=10, column=0, columnspan=2, pady=(10, 0))


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
#TODO: options to save the meas curve                       DONE
#TODO: options to save the fitted curve
#TODO: option to take 1 measurement and save as irf         
#TODO: live mua musp,                                       DONE
#TODO: smart crop, 80% 1%                                   DONE
class TMF8828RaspberryPiGUI:
    def __init__(self, root):
        self.root = root
        self.root.winfo_toplevel().title("TMF8828 Raspberry Pi Measurement")

        self.frame = ttk.Frame(root)
        self.frame.pack(fill="both", expand=True)

        self.plot_data_queue = queue.Queue()
        self.selected_channels = set() 
        # self.selected_channels.add(1)  # Default to channel 1
        self.status_queue = queue.Queue()
        self.fit_data_queue = queue.Queue()
        self.reader = DataReader(self.plot_data_queue, self.fit_data_queue, self.selected_channels, self.status_queue)

        self.fitting_worker = None
        self.left_frame = ttk.Frame(self.frame)
        self.left_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")    
        self.right_frame = ttk.Frame(self.frame)
        self.right_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        # top left corner
        self.model_frame = tk.LabelFrame(self.left_frame, text="Model", padx=5, pady=5, bg="white")
        self.model_frame.pack(fill="both", expand=True, padx=5, pady=5)
        # bottom left corner
        self.control_frame = tk.LabelFrame(self.left_frame, text="Control Panel", padx=5, pady=5, bg="white")
        self.control_frame.pack(fill="both", expand=True, padx=5, pady=5)
        # top right corner
        self.graph_frame = tk.LabelFrame(self.right_frame, text="Graph", padx=5, pady=5, bg="white")
        self.graph_frame.pack(fill="both", expand=True, padx=5, pady=5)


        self.build_control_panel()
        self.build_channel_selector()
        self.build_model_panel()

        self.status_labels = {}
        self.build_status_display()
        self.update_status_display()

        # self.fig, self.ax = plt.subplots()
        # self.bars = self.ax.bar(np.arange(128), np.zeros(128))
        # self.ax.set_ylim(0, 100)

        # self.fig, self.ax = plt.subplots()
        self.fig, (self.ax, self.residual_ax) = plt.subplots(2, 1, figsize=(6, 6), sharex=True, height_ratios=[3, 1])
        self.x_data = np.arange(128)
        self.y_data = np.zeros(128)
        self.line, = self.ax.plot(self.x_data, self.y_data, color='blue', label='Measurement Curve')
        self.model_line, = self.ax.plot(self.x_data, np.zeros_like(self.x_data), color='red', linestyle='--', label='Fitted Model')
        self.ax.legend()
        self.ax.set_ylim(0, 100)
        self.ax.set_xlim(0, 127)
        self.ax.set_xlabel("Time Bins")
        self.ax.set_ylabel("Intensity")
        self.ax.set_title("Time of Flight Graph")
        self.residual_line, = self.residual_ax.plot(self.x_data, np.zeros_like(self.x_data), color='green', label='Residual')
        self.residual_ax.axhline(0, color='gray', linestyle='--')
        self.residual_ax.set_ylabel("Residual")
        self.residual_ax.set_xlabel("Time Bins")
        self.residual_ax.legend()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, padx=5, pady=5)

        self.update_plot()

    def start_fitting_worker(self):
        if not self.contini_model_panel.get_irf():
            messagebox.showerror("Error", "No IRF loaded. Please load an IRF before starting live fitting.")
            return
        # Starts the fitting worker if not already running.
        if self.fitting_worker is None:
            self.fitting_worker = FittingWorker(
                data_queue=self.fit_data_queue,
                get_settings_fn=self.contini_model_panel.get_settings,
                irf=self.contini_model_panel.irf,  # Assuming loaded
                result_callback=self.handle_fit_result,
                interval=0.5  # every 1 second
            )
            self.fitting_worker.start()
            print("Fitting worker started.")

    #callback function for FittingWorker, if you want to do something with the fit result this function will be called whenever a fit result is available
    def handle_fit_result(self, fit_result): 

        print("Live Fit Result:", fit_result) 
        # update the fit result label
        self.fit_result_var.set(f"μa: {fit_result['mua']:.4f}, μs': {fit_result['musp']:.4f}")

        # create the fitting curve using the model
        mua, musp = fit_result["mua"], fit_result["musp"]
        settings = self.contini_model_panel.get_settings()
        rho = float(settings['rho'])
        t = settings['t']
        s = float(settings['s'])
        n1 = float(settings['n1'])
        n2 = float(settings['n2'])
        phantom = settings['phantom']
        mua_independent = settings['mua_independent'].lower() == 'true'
        m = int(settings['m'])
        output = Contini1997([rho], t, s, mua, musp, n1, n2, phantom, mua_independent, m)["total"][0][0]
        model_conv = convolve_irf_with_model(self.contini_model_panel.irf, output, geometry=GEOMETRY.REFLECTANCE, offset=0, normalize_irf=True, normalize_model=True, denest_contini_output=False)
        
        model_conv = model_conv/max(model_conv) #NORMALIZING
        self.model_line.set_ydata(model_conv)

        if len(model_conv) == len(self.y_data):
            residual = self.y_data - model_conv
            self.residual_line.set_ydata(residual)
            self.residual_ax.set_ylim(residual.min() * 1.1, residual.max() * 1.1)

        self.canvas.draw()
        print("Model line updated with new fit result.")

    def update_plot(self):
        while not self.plot_data_queue.empty():
            line = self.plot_data_queue.get()
            parts = line.strip().split(";")
            if len(parts) == 129 and parts[0].startswith("#HLONG"):
                try:
                    values = np.array(list(map(int, parts[1:])))
                    # for bar, val in zip(self.bars, values): #this one is for bar graph
                    #     bar.set_height(val)
                    # self.ax.set_ylim(0, values.max() * 1.1)
                    values = values / max(values) #NORMALIZING
                    self.y_data = values
                    self.line.set_ydata(self.y_data)
                    # self.ax.set_ylim(0, max(100, values.max() * 1.1))  # Keep minimum Y max
                    self.ax.set_ylim(0, values.max() * 1.1)  # Keep minimum Y max
                    self.canvas.draw()
                except Exception as e:
                    print(f"Error: {e}")
        self.root.after(100, self.update_plot)
    
    """
    Status Display Methods
    """
    def build_status_display(self):
        status_frame = tk.LabelFrame(self.graph_frame, text="Sensor Status", padx=5, pady=5, bg="white")
        status_frame.grid(row=2, column=0, padx=5, pady=5, sticky="w")

        # Create a StringVar to hold the dynamic text
        self.fit_result_var = tk.StringVar()
        self.fit_result_var.set("μa: ---, μs': ---")
        # Label bound to the StringVar
        fit_label = tk.Label(self.graph_frame, textvariable=self.fit_result_var, font=("Arial", 14))
        fit_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")

        status_fields = [
            "Timestamp", "Iterations", "Threshold", "SPAD Map ID",
            "Measurement Range", "Operation Mode", "De-scattering", "Short Range Mode"
        ]

        # Top row: first 5 fields
        for idx, field in enumerate(status_fields[:5]):
            label = tk.Label(status_frame, text=f"{field}: N/A", anchor="w", bg="white")
            label.grid(row=0, column=idx, sticky="w", padx=5)
            self.status_labels[field] = label

        # Bottom row: remaining 4 fields
        for idx, field in enumerate(status_fields[5:]):
            label = tk.Label(status_frame, text=f"{field}: N/A", anchor="w", bg="white")
            label.grid(row=1, column=idx, sticky="w", padx=5)
            self.status_labels[field] = label

    def update_status_display(self):
        while not self.status_queue.empty():
            line = self.status_queue.get()
            if line.startswith("#ITT"):
                self.parse_and_update_status(line)

        # Schedule next check
        self.root.after(1000, self.update_status_display)  # update every 1 second

    def parse_and_update_status(self, line):
        parts = line.strip().split(";")
        if len(parts) != 9:
            return  # Invalid format

        status_values = {
            "Timestamp": parts[1],
            "Iterations": parts[2],
            "Threshold": parts[3],
            "SPAD Map ID": parts[4],
            "Measurement Range": parts[5],
            "Operation Mode": parts[6],
            "De-scattering": "Enabled" if parts[7] == "1" else "Disabled",
            "Short Range Mode": "Enabled" if parts[8] == "1" else "Disabled"
        }

        for key, value in status_values.items():
            self.status_labels[key].config(text=f"{key}: {value}")



    """
    Channel Selector Methods
    These methods handle the creation, destruction, and management of the channel selector frame.
    """
    def build_channel_selector(self):
        self.channel_frame = tk.LabelFrame(self.control_frame, text="TDC Channel", padx=5, pady=5, bg="white")
        self.channel_frame.grid(row=0, column=3, rowspan=6,columnspan=3, padx=5, pady=5)

        # Canvas and scrollbar setup
        canvas = tk.Canvas(self.channel_frame, height=120, bg="white")
        scrollbar = tk.Scrollbar(self.channel_frame, orient="vertical", command=canvas.yview)
        self.inner_frame = tk.Frame(canvas, bg="white")
        self.inner_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # apply button to update the selected channels
        apply_btn = tk.Button(self.inner_frame, text="Apply", command=self.apply_selected_channels, bg="white", fg="black")
        apply_btn.pack(pady=(10, 0))
        
        # Checkboxes for channels
        self.channel_vars = []
        for i in range(10):
            var = tk.IntVar()
            chk = tk.Checkbutton(self.inner_frame, text=f"Channel {i}", variable=var, bg="white")
            chk.pack(anchor="w")
            self.channel_vars.append(var)

    def destroy_channel_selector(self):
        """Destroys the channel selector frame and its widgets"""
        for widget in self.channel_frame.winfo_children():
            widget.destroy()
        self.channel_frame.destroy()

    def get_selected_channels(self):
        """Returns a list of selected channel numbers"""
        return [i for i, var in enumerate(self.channel_vars) if var.get()]
    
    def apply_selected_channels(self):
        """Updates the selected channels in data reader with the current selection (channel_vars)"""
        self.selected_channels.clear()
        self.selected_channels.update(self.get_selected_channels())
        print(f"Selected channels updated: {sorted(self.selected_channels)}")

        # self.build_graphs_for_selected_channels()


    """
    Control Frame Methods
    These methods handle the creation, destruction, and management of the control frame.
    """
    def build_control_panel(self):
        meas_settings_frame = tk.LabelFrame(self.control_frame, text="Measurement Settings", padx=5, pady=5, bg="white")
        meas_settings_frame.grid(row=0, column=0, rowspan=6,columnspan=3, padx=5, pady=5)

        # Iterations
        tk.Label(meas_settings_frame, text="Iterations (0-65535):", bg="white").grid(row=0, column=0, sticky="w")
        self.iterations_var = tk.IntVar()
        tk.Entry(meas_settings_frame, textvariable=self.iterations_var, width=10).grid(row=0, column=1)
        tk.Button(meas_settings_frame, text="Set", command=self.set_iterations).grid(row=0, column=2)

        # Threshold
        tk.Label(meas_settings_frame, text="Threshold (0-255):", bg="white").grid(row=1, column=0, sticky="w")
        self.threshold_var = tk.IntVar()
        tk.Entry(meas_settings_frame, textvariable=self.threshold_var, width=10).grid(row=1, column=1)
        tk.Button(meas_settings_frame, text="Set", command=self.set_threshold).grid(row=1, column=2)

        # Short Range Mode
        tk.Label(meas_settings_frame, text="Short Range Mode (0-Off, 1-On):", bg="white").grid(row=2, column=0, sticky="w")
        self.short_range_var = tk.IntVar()
        tk.Entry(meas_settings_frame, textvariable=self.short_range_var, width=10).grid(row=2, column=1)
        tk.Button(meas_settings_frame, text="Set", command=self.set_short_range).grid(row=2, column=2)

        # Operation Mode
        tk.Label(meas_settings_frame, text="Operation Mode (0-3):", bg="white").grid(row=3, column=0, sticky="w")
        self.operation_mode_var = tk.IntVar()
        tk.Entry(meas_settings_frame, textvariable=self.operation_mode_var, width=10).grid(row=3, column=1)
        tk.Button(meas_settings_frame, text="Set", command=self.set_operation_mode).grid(row=3, column=2)

        # Histogram Mode
        tk.Label(meas_settings_frame, text="Histogram Mode (0-3):", bg="white").grid(row=4, column=0, sticky="w")
        self.histogram_mode_var = tk.IntVar()
        tk.Entry(meas_settings_frame, textvariable=self.histogram_mode_var, width=10).grid(row=4, column=1)
        tk.Button(meas_settings_frame, text="Set", command=self.set_histogram_mode).grid(row=4, column=2)
        
        # ROW 6: Start Reader and Toggle Measurement buttons and take one measurement button
        self.start_reader_button = tk.Button(self.control_frame, text="Connect", command=self.start_reader)
        self.start_reader_button.grid(row=6, column=0, padx=5, pady=5)   
        self.toggle_measurement_button = tk.Button(self.control_frame, text="Toggle Measurement", command=self.reader.toggle_measurement, state=tk.DISABLED)
        self.toggle_measurement_button.grid(row=6, column=1, padx=5, pady=5)

        # ROW 7: File selection and saving options and start live fitting button
        self.file_path_var = tk.StringVar()
        self.file_path_var.set("No file selected")
        self.file_path_label = tk.Label(self.control_frame, textvariable=self.file_path_var, bg="white")
        self.file_path_label.grid(row=7, column=0, padx=5, pady=5, sticky="w")
        self.file_path_button = tk.Button(self.control_frame, text="Select File", command=self.select_file)
        self.file_path_button.grid(row=7, column=1, padx=5, pady=5)
        self.save_meas_checkbox = tk.Checkbutton(self.control_frame, text="Save Measurement to CSV", bg="white", command=self.reader.toggle_saving_to_csv)
        self.save_meas_checkbox.grid(row=7, column=2, padx=5, pady=5)
        tk.Button(self.control_frame, text="Start Live Fitting", command=self.start_fitting_worker).grid(row=7, column=3, padx=5, pady=5)
    
    def start_reader(self):
            self.reader.start()
            self.start_reader_button.config(state=tk.DISABLED)
            self.toggle_measurement_button.config(state=tk.NORMAL)

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.file_path_var.set(os.path.basename(file_path))
            self.reader.set_file_path(file_path)
        else:
            self.file_path_var.set("No file selected")

    def set_iterations(self):
        value = self.iterations_var.get()
        self.reader.set_number_of_iterations(value)

    def set_threshold(self):
        value = self.threshold_var.get()
        self.reader.set_object_detection_threshold(value)

    def set_short_range(self):
        value = self.short_range_var.get()
        self.reader.set_short_range_mode(value)

    def set_operation_mode(self):
        value = self.operation_mode_var.get()
        self.reader.set_operation_mode(value)

    def set_histogram_mode(self):
        value = self.histogram_mode_var.get()
        self.reader.set_histogram_mode(value)

    """
    Contini Model Panel Frame Methods
    These methods handle the creation, destruction, and management of the model panel frame.
    """
    def build_model_panel(self):
        self.contini_model_panel = ContiniModelPanel(self.model_frame)
        panel_frame = self.contini_model_panel.get_frame()
        panel_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

    
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

if __name__ == "__main__":
    #Run app
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()