from enum import IntEnum

from diffusion_equation.diffusion_equation import Contini1997


GEOMETRY = IntEnum('GEOMETRY', {'REFLECTANCE':0, 'TRANSMITTANCE':1})
import numpy as np
import csv
import os

mu_list = [
    (1.24, 188.37),
    (2.67, 92.50),
    (0.61, 241.29),
    (2.03, 289.43),
    (1.78, 66.91),
    (0.83, 138.75),
    (2.48, 271.64),
    (0.94, 210.86),
    (1.35, 59.02),
    (2.89, 172.18)
]

mu_list_divided_by_100 = [(round(mua / 100, 4), round(musp / 100, 4)) for mua, musp in mu_list]
print("mu_list_divided_by_100:", mu_list_divided_by_100)

phantom_type = ["slab", "semiinf"]

params = {
    'rho': '15',
    'time_step (ns)': '0.004', #4e-12
    'num_bins': '4096',
    's': '1',
    'mua': '0.01', # mm^{-1}
    'musp': '1', # mm^{-1}
    'n1': '1',
    'n2': '1.41',
    'phantom': 'slab',
    'mua_independent': 'True',
    'm': '200',
    'geometry': GEOMETRY.REFLECTANCE,  # Measurement geometry
    't' : None, # this is calculated from time_step and num_bins
}

time_step = float(params['time_step (ns)'])
num_bins = int(params['num_bins'])
t = [1e-9 if i == 0 else time_step * i for i in range(num_bins)] # don't use 0 for time bin 0 otherwise it would cause error
params['t'] = t

rho = float(params['rho'])
t = params['t']
s = float(params['s'])
# mua = float(params['mua'])
# musp = float(params['musp'])
n1 = float(params['n1'])
n2 = float(params['n2'])
# phantom = params['phantom']
mua_independent = params['mua_independent'].lower() == 'true'
m = int(params['m'])

#placeholder
mua = None
musp = None
phantom = None

dir_name = "C:/Users/USER/Desktop/work related/Tyndall Intern/FLIM_Keela/output/"

for mua, musp in mu_list_divided_by_100:
    for phantom in phantom_type:
        print(f"Fitting for mua: {mua}, musp: {musp}, phantom: {phantom}")
        output = Contini1997([rho], t, s, mua, musp, n1, n2, phantom, mua_independent, m) 
        result = output["total"][0][0] #just get the reflectance for the first rho
        csv_file = os.path.join(dir_name, "python_contini_master_2.csv")
        write_header = not os.path.exists(csv_file)

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if write_header:
                header = ['mua', 'musp', 'phantom'] + [f'bin_{i}' for i in range(len(result))]
                writer.writerow(header)
            row = [mua, musp, phantom] + list(result)
            writer.writerow(row)

for mua, musp in mu_list_divided_by_100:
    pass
        
    
