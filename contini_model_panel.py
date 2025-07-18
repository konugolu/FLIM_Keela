import csv
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import filedialog
from diffusion_equation.diffusion_equation import Contini1997
from diffusion_equation.fit import model, GEOMETRY

class ContiniModelPanel:
    """
    A class to represent the Contini model panel for diffusion equations.
    Provides a Tkinter GUI frame to input parameters and compute results.
    """

    def __init__(self, root):
        self.root = root
        self.frame = ttk.Frame(root)

        self.irf = None

        self.params = {
            'rho': '0',
            'time_step (ns)': '0.19',
            'num_bins': '128',
            's (slab thickness mm)': '18e3',
            'mua': '0.1e-4',
            'musp': '5e-4',
            'n1 (external n)': '1',
            'n2 (diffusing n)': '1.4',
            'phantom': 'slab',
            'mua_independent (True/False)': 'True',
            'm (num imaginary sources)': '400'
        }

        self.entries = {}

        for i, (label, default) in enumerate(self.params.items()):
            ttk.Label(self.frame, text=label).grid(row=i, column=0, sticky='w', padx=5, pady=5)
            entry = ttk.Entry(self.frame)
            entry.insert(0, default)
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.entries[label] = entry

        self.contini_button = ttk.Button(self.frame, text="Contini", command=self.compute_contini)
        self.contini_button.grid(row=len(self.params), column=0, columnspan=1, pady=10)

        self.compute_convolution_button = ttk.Button(self.frame, text="Contini & Convolve IRF", command=self.compute_convolution_with_irf)
        self.compute_convolution_button.grid(row=len(self.params), column=1, columnspan=1, pady=10)

        load_irf_button = ttk.Button(self.frame, text="Load IRF from CSV", command=self.load_irf)
        load_irf_button.grid(row=len(self.params)+1, column=0, columnspan=2, pady=10)


    def compute_contini(self):
        """
        handles contini button, reads parameters from entries, computes the Contini1997 model
        Return Value: A dictionary with only the "total" key, which contains 2 arrays, both array size are (1, num_bins)
        and the first array is the reflectance and the second array is the transmittance.
        """
        try:
            rho = float(self.entries['rho'].get())
            time_step = float(self.entries['time_step (ns)'].get())
            num_bins = int(self.entries['num_bins'].get())
            t = [time_step * i for i in range(num_bins)]

            s = float(self.entries['s (slab thickness mm)'].get())
            mua = float(self.entries['mua'].get())
            musp = float(self.entries['musp'].get())
            n1 = float(self.entries['n1 (external n)'].get())
            n2 = float(self.entries['n2 (diffusing n)'].get())
            phantom = self.entries['phantom'].get()
            mua_independent = self.entries['mua_independent (True/False)'].get().lower() == 'true'
            m = int(self.entries['m (num imaginary sources)'].get())

            result = Contini1997([rho], t, s, mua, musp, n1, n2, phantom, mua_independent, m)["total"]
            # messagebox.showinfo("Result", f"Contini1997 Output:\n{result}")
            print("Contini1997 Output:", result) 
            print(len(result[0][0]))
            
            # plot both result[0][0] and result[1][0] as subplots in one figure
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            axs[0].plot(result[0][0])
            axs[0].set_title("Reflectance AKA result[0][0]")
            axs[0].set_ylabel("Value")
            axs[1].plot(result[1][0])
            axs[1].set_title("Transmittance AKA result[1][0]")
            axs[1].set_xlabel("Time Bin")
            axs[1].set_ylabel("Value")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to compute: {e}")

    def compute_convolution_with_irf(self):
        try:
            rho = float(self.entries['rho'].get())
            time_step = float(self.entries['time_step (ns)'].get())
            num_bins = int(self.entries['num_bins'].get())
            t = [time_step * i for i in range(num_bins)]

            s = float(self.entries['s (slab thickness mm)'].get())
            mua = float(self.entries['mua'].get())
            musp = float(self.entries['musp'].get())
            n1 = float(self.entries['n1 (external n)'].get())
            n2 = float(self.entries['n2 (diffusing n)'].get())
            phantom = self.entries['phantom'].get()
            mua_independent = self.entries['mua_independent (True/False)'].get().lower() == 'true'
            m = int(self.entries['m (num imaginary sources)'].get())

            result = model(self.irf, [rho], t, s, mua, musp, n1, n2, phantom, mua_independent, m, GEOMETRY.REFLECTANCE, offset=0)
            print("Convolution Result:", result)
            print(len(result))

            #plot the convolved result
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 4))
            plt.plot(result)
            plt.title("Convolved Result (Reflectance)")
            plt.xlabel("Time Bin")
            plt.ylabel("Value")
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to compute: {e}")
        
    def load_irf(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        try:
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    self.irf = [float(value) for value in row]
                    break  # Only read the first line

            # messagebox.showinfo("IRF Loaded", f"IRF loaded with {len(self.irf)} time bins.")
            print("IRF Loaded", f"IRF loaded with {len(self.irf)} time bins.")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 4))
            plt.plot(self.irf)
            plt.title("Loaded IRF")
            plt.xlabel("Time Bin")
            plt.ylabel("IRF Value")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load IRF: {e}")

    def get_frame(self):
        return self.frame
    


if __name__ == '__main__':
    root = tk.Tk()
    root.title("Contini1997 Model Panel")
    panel = ContiniModelPanel(root)
    panel.get_frame().pack(padx=10, pady=10)
    root.mainloop()