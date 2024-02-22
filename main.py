import os
import sys
import time
import subprocess
import mlippy
from ikdtools.io.mlip.cfg import write_cfg, read_cfg
import numpy as np
import copy
from mpi4py import MPI
from pot import read_untrained_MTP, write_MTP, generate_random_numbers
from opt import optimization_nelder, optimization_bfgs, optimization_DE, optimization_sa
from itertools import combinations_with_replacement
from Moo import optimization_GA

def configuration_set(input_cfg, species=['H']):
    Training_set = read_cfg(input_cfg, ":", species)
    current_set = copy.deepcopy(Training_set)
    return Training_set, current_set


def target_value(Training_set):
    # Extract energies and forces from the training set
    Target_energies = [atom.calc.results['free_energy'] for atom in Training_set]
    Target_forces = [atom.calc.results['forces'] for atom in Training_set]
    Target_stress =   [atom.calc.results['stress'] for atom in Training_set]
    return np.array(Target_energies), np.array(Target_forces), np.array(Target_stress)


def calculate_energy_force_stress(atom, potential):
    atom.calc = potential
    energy = atom.get_potential_energy()
    force = atom.get_forces()
    stress = atom.get_stress() # stress property not implemented in morse
    #stress = [0, 0, 0, 0, 0, 0]  # workaround
    return energy, force, stress


def current_value(current_set, potential):
    rank = comm.Get_rank()
    size = comm.Get_size()
    # Initialize lists to store energies, forces, and stresses
    current_energies = []
    current_forces = []
    current_stress = []

    if isinstance(current_set, list):
        atoms = current_set
    else:
        atoms = [current_set]

    # Determine the chunk of atoms to process for each MPI process
    chunk_size = len(atoms) // size
    remainder = len(atoms) % size
    start_idx = rank * chunk_size
    end_idx = (rank + 1) * chunk_size + (remainder if rank == size - 1 else 0)
    local_atoms = atoms[start_idx:end_idx]

    # Perform local calculations
    local_results = []
    for atom in local_atoms:
        local_results.append(calculate_energy_force_stress(atom, potential))

    # Gather results from all processes
    all_results = comm.gather(local_results, root=0)

    # Process results on root process
    if rank == 0:
        for result_list in all_results:
            for energy, force, stress in result_list:
                current_energies.append(energy)
                current_forces.append(force)
                current_stress.append(stress)

    # Broadcast the processed results to all processes
    current_energies = comm.bcast(current_energies, root=0)
    current_forces = comm.bcast(current_forces, root=0)
    current_stress = comm.bcast(current_stress, root=0)

    return np.array(current_energies), np.array(current_forces), np.array(current_stress)


def mytarget(parameters, *args):
    Target_energies, Target_forces, Target_stress, global_weight, configuration_weight, current_set = args
    GEW, GFW, GSW = global_weight

    # potential = force_field(parameters)
    potential = MTP_field(parameters)
    current_energies, current_forces, current_stress = current_value(current_set, potential)

    # Calculate the energy difference
    energy_difference = configuration_weight * (current_energies - Target_energies)
    energy_scalar_difference = np.sum(energy_difference ** 2)

    # Calculate the force difference
    force_difference = [np.square(np.linalg.norm(configuration_weight[i] * (current_forces[i] - Target_forces[i]),axis=1)) for i in range(len(Target_forces))]
    force_scalar_difference = np.sum(np.concatenate(force_difference))

    # Calculate the stress difference
    stress_difference = [np.square(np.linalg.norm(configuration_weight[j] * (current_stress[j] - Target_stress[j]))) for j in range(len(Target_stress))]
    stress_scalar_difference = np.sum(stress_difference)

    return GEW * energy_scalar_difference + GFW * force_scalar_difference + GSW * stress_scalar_difference


#def RMSE(reference_set, current_set, potential):
#    current_energies, current_forces, current_stress = current_value(current_set, potential)
#    Target_energies, Target_forces, Target_stress = target_value(reference_set)
#
#    error_energy = [((current_energies[i] - Target_energies[i]) / len(current_set[i])) ** 2 for i in range(len(current_set))]
#    error_force = [np.sum(current_forces[i] - Target_forces[i]) / (3 * len(current_set[i])) ** 2 for i in range(len(current_set))]
#    error_stress = [np.sum(current_stress[i] - Target_stress[i]) / 6 ** 2 for i in range(len(current_set))]
#
#    RMSE_energy = (np.sum(error_energy) * 1000) / len(current_set)
#    RMSE_force = (np.sum(error_force)) / len(current_set)
#    RMSE_stress = (np.sum(error_stress)) / len(current_set)
#
#    print("RMSE Energy per atom (meV/atom):", RMSE_energy)
#    print("RMSE force per atom (eV/Ang):", RMSE_force)
#    print("RMSE stress (GPa):", RMSE_force*0.1)

def RMSE(cfg,pot):
    ts = mlippy.ase_loadcfgs(cfg)
    mlip = mlippy.initialize()
    mlip = mlippy.mtp()
    mlip.load_potential("Test.mtp")
    opts = {}
    mlip.add_atomic_type(1)
    potential = mlippy.MLIP_Calculator(mlip, opts)
    errors=mlippy.ase_errors(mlip,ts)
    print("RMSE Energy per atom (meV/atom):",1000*float(errors['Energy per atom: RMS absolute difference']))
    print("RMSE force per atom (eV/Ang):",float(errors['Forces: RMS absolute difference']))
    print("RMSE stress (GPa):", float(errors['Stresses: RMS absolute difference']))
    return errors






def MTP_field(parameters):
    # Assuming read_untrained_MTP, write_MTP, and combinations_with_replacement are defined elsewhere
    yaml_data = read_untrained_MTP(untrained_mtp)
    species_count = int(yaml_data['species_count'])
    max_dist = int(float(yaml_data['max_dist']))
    min_dist = int(float(yaml_data['min_dist']))

    scaling = parameters[0]
    length_moment = int(yaml_data['alpha_scalar_moments'])
    moment_coeffs = parameters[1:length_moment + 1]
    species_coeffs = parameters[length_moment + 1:length_moment + 1 + species_count]
    total_radial = parameters[length_moment + 1 + species_count:]

    species_pairs = combinations_with_replacement(range(species_count), 2)

    radial_coeffs = np.array(total_radial).reshape(-1, int(yaml_data['radial_basis_size'])).tolist()
    #if rank==0:
    file = "Test.mtp"
    #else:
    #    file = "test.mtp"
    write_MTP(file, species_count, min_dist, max_dist, scaling, radial_coeffs, species_coeffs, moment_coeffs, yaml_data)

    mlip = mlippy.initialize()
    mlip = mlippy.mtp()
    mlip.load_potential(file)
    opts = {}
    mlip.add_atomic_type(1)
    potential = mlippy.MLIP_Calculator(mlip, opts)

    return potential
#======================================================================================================================================================
if __name__ == "__main__":
    import os
    import subprocess
    import time
    
    # Other imports...
    
    if __name__ == "__main__":
        start_time = time.time()
        current_directory = os.getcwd()
        global untrained_mtp 
        global comm
        global size
        global rank
        global cfg_file
         
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        cfg_file= current_directory+"/final.cfg"
        untrained_mtp = current_directory+"/02.mtp"
        
        #if rank!=0:
        #    sys.stdout = None
        Training_set, current_set = configuration_set(cfg_file, species=['H'])
        Target_energies, Target_forces, Target_stress = target_value(Training_set)

        global_weight = [1, 0.01, 0]  
        configuration_weight = np.ones(len(Training_set))  

        yaml_data = read_untrained_MTP(untrained_mtp)
        species_count = int(yaml_data['species_count'])

        # Create folders for each rank
        folder_name = f"rank_{rank}"
        folder_path = os.path.join(current_directory, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    
        # Change working directory to the created folder
    
        # Rest of the code...
        os.chdir(folder_path) 
    #    for i in np.arange(1,100):
	
        species_pairs = combinations_with_replacement(range(species_count), 2)
        w_cheb = species_count+int(yaml_data['alpha_scalar_moments'])
        cheb=len(list(species_pairs)) * int(yaml_data['radial_funcs_count']) * int(yaml_data['radial_basis_size'])
        #global bounds
       # global lower_bounds
       # global upper_bounds
        bounds=[(-1000,1000)]+[(-5,5)]*w_cheb+[(-0.1,0.1)]*cheb
        #lower_bounds = [item[0] for item in bounds]
        #upper_bounds = [item[1] for item in bounds]


        initial_guess = [1000]+[5]*w_cheb +generate_random_numbers(cheb, -0.1, 0.1, 10)
        



        optimized_parameters = optimization_GA(mytarget,initial_guess,bounds,Target_energies, Target_forces, Target_stress,global_weight, configuration_weight, current_set)
        
#==============================================================================================
        #initial_guess=[-7.54050095e+00,-1.92261361e-04,-1.68294883e+00,7.18632606e-01, -7.05708440e-01,1.23713720e+00,7.27847678e-02,2.29800048e-02,8.02591439e-01,-3.11064787e-01,-1.32991017e-01] 
        initial_guess=optimized_parameters
        optimized_parameters = optimization_nelder(mytarget,initial_guess,bounds, Target_energies, Target_forces, Target_stress,
                       global_weight, configuration_weight, current_set)
 
        #initial_guess=optimized_parameters
        
        #optimized_parameters = optimization_bfgs(mytarget,initial_guess, Target_energies, Target_forces, Target_stress,global_weight, configuration_weight, current_set)
        # Change back to the original directory after processing
#====================================================================================================    
        end_time = time.time()
        elapsed_time = end_time - start_time
            #Calculate RMSE
        #print(optimized_parameters)
        potential = MTP_field(optimized_parameters)
        RMSE(cfg_file,"Test.mtp")

        print("Total time taken:", elapsed_time, "seconds")
        os.chdir(current_directory)
        comm.Barrier()
        MPI.Finalize()
