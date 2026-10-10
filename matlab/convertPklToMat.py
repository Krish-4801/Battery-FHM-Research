#!/usr/bin/env python3
"""
Convert Data.pkl into MATLAB-accessible Data.mat.
All MATLAB variable / field names use underscores instead of dots to be strictly
valid MATLAB identifiers (e.g., 'T288_C0_5_X' instead of 'T288_C0.5_X').

Saves:
  1. inputs_all (N x 4): [x, time, C_rate, Temperature]
  2. cs_all     (N x 1): Active solid concentration
  3. ce_all     (N x 1): Electrolyte concentration
  4. phie_all   (N x 1): Electrolyte potential
  5. phis_all   (N x 1): Solid phase potential
  6. JLi_all    (N x 1): Intercalation current density
  7. Per-condition evaluation profiles (times, X, and time-resolved variables)
     under valid field prefixes such as 'T298_C1_X', 'T298_C0_5_X', 'T298_C1_t{t}_cs', etc.
"""

import os
import sys
import pickle
import numpy as np
import scipy.io as sio

def make_valid_matlab_tag(c_val):
    # Convert numbers like 0.5 to '0_5', 1 to '1', etc.
    s = str(c_val)
    return s.replace('.', '_')

def convert_pkl_to_mat(pkl_path, mat_path):
    print(f"Loading {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    temperatures = [288.0, 298.0, 308.0, 318.0]
    C_rates = [0.5, 1, 2, 4, 6, 8, 10]

    input_list = []
    cs_list = []
    ce_list = []
    phie_list = []
    phis_list = []
    JLi_list = []

    mat_dict = {}

    print("Extracting datasets for all conditions...")
    for T in temperatures:
        for C in C_rates:
            T_str = f"T{T}"
            C_str = f"C{C}"
            if T_str in data and C_str in data[T_str]:
                time_data = data[T_str][C_str]
                # Spatial coordinate grid
                X_vals = np.array(time_data[0]["X"], dtype=np.float32).flatten()
                
                # Valid MATLAB prefix (no dots)
                c_tag = make_valid_matlab_tag(int(C) if float(C).is_integer() else C)
                t_tag = make_valid_matlab_tag(int(T) if float(T).is_integer() else T)
                eval_prefix = f"T{t_tag}_C{c_tag}"
                mat_dict[f"{eval_prefix}_X"] = X_vals

                time_keys = []
                for time_val_str, variables in time_data.items():
                    if time_val_str != "X":
                        try:
                            time_val = int(time_val_str)
                            time_keys.append(time_val)

                            ce_data = np.array(variables["ce"], dtype=np.float32).flatten()
                            cs_data = np.array(variables["cs"], dtype=np.float32).flatten()
                            phie_data = np.array(variables["pe"], dtype=np.float32).flatten()
                            phis_data = np.array(variables["ps"], dtype=np.float32).flatten()
                            JLi_data = np.array(variables["J_Li"], dtype=np.float32).flatten()

                            # Save time-resolved profile for evaluateResults (valid identifier)
                            t_prefix = f"{eval_prefix}_t{time_val}"
                            mat_dict[f"{t_prefix}_cs"] = cs_data
                            mat_dict[f"{t_prefix}_ce"] = ce_data
                            mat_dict[f"{t_prefix}_pe"] = phie_data
                            mat_dict[f"{t_prefix}_ps"] = phis_data
                            mat_dict[f"{t_prefix}_J_Li"] = JLi_data

                            for x_idx, x_val in enumerate(X_vals):
                                input_list.append([x_val, float(time_val), float(C), float(T)])
                                cs_list.append(cs_data[x_idx])
                                ce_list.append(ce_data[x_idx])
                                phie_list.append(phie_data[x_idx])
                                phis_list.append(phis_data[x_idx])
                                JLi_list.append(JLi_data[x_idx])
                        except ValueError:
                            print(f"Skipping key {time_val_str}")
                
                mat_dict[f"{eval_prefix}_times"] = np.array(sorted(time_keys), dtype=np.float32)

    print("Formatting arrays into tensors...")
    mat_dict["inputs_all"] = np.array(input_list, dtype=np.float32)
    mat_dict["cs_all"]     = np.array(cs_list, dtype=np.float32).reshape(-1, 1)
    mat_dict["ce_all"]     = np.array(ce_list, dtype=np.float32).reshape(-1, 1)
    mat_dict["phie_all"]   = np.array(phie_list, dtype=np.float32).reshape(-1, 1)
    mat_dict["phis_all"]   = np.array(phis_list, dtype=np.float32).reshape(-1, 1)
    mat_dict["JLi_all"]    = np.array(JLi_list, dtype=np.float32).reshape(-1, 1)

    print(f"Total samples processed: {mat_dict['inputs_all'].shape[0]}")
    print(f"Writing to {mat_path} (compressed)...")
    sio.savemat(mat_path, mat_dict, do_compression=True)
    print("Conversion complete! All field names are valid MATLAB identifiers.")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(base_dir, '..'))

    pkl_file = os.path.join(project_dir, 'Data.pkl')
    mat_file = os.path.join(project_dir, 'Data.mat')

    if len(sys.argv) > 1:
        pkl_file = sys.argv[1]
    if len(sys.argv) > 2:
        mat_file = sys.argv[2]

    convert_pkl_to_mat(pkl_file, mat_file)
