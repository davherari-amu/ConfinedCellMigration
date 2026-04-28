# Libraries {{{
import os, sys, shutil
from pdb import set_trace

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import log, io

import numpy as np

import copy

from datetime import datetime
import getopt, json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from python_utils.gspde import GSPDE, Dynamics, forces, ecm_funcs
from python_utils import mesh_utils
from python_utils.mesh_utils import MakeCircle
from python_utils.gspde.multi_gspde import Cell
from python_utils.misc_utils import mprint, PlotCircles, PlotPolygons, CombineClasses, SelectFunction, SelectClass
# }}}

# Single cell function {{{
def single_cell(params):

    # Setting {{{
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    log.set_log_level(log.LogLevel.WARNING)
    np.random.seed(rank)
    PETSc.Sys.popErrorHandler()  # make PETSc throw exceptions instead of aborting
    # }}}

    # Internal parameters {{{
    # Reference values
    xref = 8.0 # µm
    # Basic dynamics
    cell_dynamics = ["SizeControlDynamics"]
    nucl_dynamics = ["SizeControlDynamics", "NucleusToPMDynamics"]
    cort_dynamics = ["SizeControlDynamics"]
    # }}}

    # Organise parameters {{{
    # Make a copy of parameters
    params_copy = copy.deepcopy(params)
    # Time {{{
    timeParams = params["time"]
    Ttot = timeParams["Ttot"]
    dt = timeParams["dt"]
    print_each = timeParams["print_each"]
    print_each_report = timeParams.get("print_each_report", 1)
    # }}}
    # Cell {{{
    paramsCell = params["cell"]
    # Initialisation of parameters {{{
    cell_params = {
            # Communicator from MPI
            "comm" : comm,
            # Time step
            "dt" : dt
            }
    # }}}
    # Contact {{{
    contactParams = params["contact"]
    contact_tol = contactParams["tol"]
    contact_stiffness = contactParams["stiffness"]
    # Fill to cell_params
    cell_params["self_tol"] = contact_tol
    cell_params["self_stiffness"] = contact_stiffness
    # }}}
    # Mesh {{{
    meshparamsCell = paramsCell["mesh"]
    # Geometry
    radius_ce = meshparamsCell.get("radius", 1.0)
    # Mesh size
    lc_ce = meshparamsCell["lc"]
    meshOrder_ce = meshparamsCell["meshOrder"]
    meshFunction_ce = SelectFunction(meshparamsCell.get("function", "MakeCircle"), mesh_utils)
    # Fill to cell_params
    aRef_ce = np.pi*radius_ce**2.0
    periRef_ce = 2.0*np.pi*radius_ce
    Href_ce = 1.0/radius_ce
    cell_params["model_mesh"] = io.gmshio.model_to_mesh(
                meshFunction_ce(radius_ce, lc_ce, meshOrder = meshOrder_ce),
                comm, 0, gdim = 2)
    cell_params["meshOrder"] = meshOrder_ce
    cell_params["aRef"] = aRef_ce
    cell_params["periRef"] = periRef_ce
    cell_params["Href"] = Href_ce
    # }}}
    # Material {{{
    materialparamsCell = paramsCell["material"]
    bendingTensionRatio_cell = materialparamsCell["bendingTensionRatio"]
    area_stiffness_cell = materialparamsCell["area_stiffness"]
    peri_stiffness_cell = materialparamsCell["peri_stiffness"]
    peri_max_factor_cell = materialparamsCell.get("peri_max_factor", 1.0)
    # Fill to cell_params
    cell_params["bendingTensionRatio"] = bendingTensionRatio_cell
    cell_params["area_stiffness"] = area_stiffness_cell
    cell_params["peri_stiffness"] = peri_stiffness_cell
    cell_params["peri_max_factor"] = peri_max_factor_cell
    # }}}
    # Protrusions {{{
    if "protrusions" in paramsCell:
        protrusionsParams = paramsCell["protrusions"]
        # Add dynamics
        cell_dynamics.append(protrusionsParams["dynamics"])
        prot_tim_min = protrusionsParams["prot_tim_min"]
        prot_mag_min = protrusionsParams["prot_mag_min"]
        prot_wid_min = protrusionsParams["prot_wid_min"]
        prot_theta_pi_min = protrusionsParams["prot_theta_pi_min"]
        prot_tim_max = protrusionsParams.get("prot_tim_max", prot_tim_min)
        prot_mag_max = protrusionsParams.get("prot_mag_max", prot_mag_min)
        prot_wid_max = protrusionsParams.get("prot_wid_max", prot_wid_min)
        prot_theta_pi_max = protrusionsParams.get("prot_theta_pi_max", prot_theta_pi_min)
        if "prot_freq" in protrusionsParams:
            prot_freq = protrusionsParams["prot_freq"]
            prot_tim_mean = 0.5*(prot_tim_min + prot_tim_max)
            prot_num = int(prot_freq*Ttot/prot_tim_mean)
        else:
            prot_num = protrusionsParams["prot_num"]
        # Define force class
        ForceClass = CombineClasses([protrusionsParams["force_class"]], forces)
        # Get ea_params initialisation
        ea_params = protrusionsParams.get("ea_params", {})
        # Set ea_params {{{
        tims = np.random.uniform(prot_tim_min, prot_tim_max, prot_num)
        mags = np.random.uniform(prot_mag_min, prot_mag_max, prot_num)
        wids = np.random.uniform(prot_wid_min, prot_wid_max, prot_num)
        thetas = np.random.uniform(prot_theta_pi_min, prot_theta_pi_max, prot_num)*np.pi
        t0s = np.random.rand(prot_num)*Ttot
        ea_params["N"]          = prot_num
        ea_params["Ts"]         = tims
        ea_params["hs"]         = mags
        ea_params["ws"]         = wids
        ea_params["thetas"]     = thetas
        ea_params["t0s"]        = t0s
        ea_params["ForceClass"] = ForceClass
        if not "startAt" in ea_params:
            ea_params["startAt"] = -(prot_tim_max + prot_tim_min)/2.0
        # Fill to cell_params
        cell_params["ea_params"] = ea_params
        # }}}
    # }}}
    # ECM {{{
    if "ecm" in paramsCell:
        ecmParams = paramsCell["ecm"]
        # Add dynamics
        cell_dynamics.append(ecmParams["dynamics"])
        # Get ecm_params
        ecm_params = ecmParams.get("ecm_params", {})
        ecm_make_func = SelectFunction(ecmParams["ecm_make_func"], ecm_funcs)
        ecm_make_func_kwargs = ecmParams["ecm_make_func_kwargs"]
        ecm_points = ecm_make_func(**ecm_make_func_kwargs)
        # Fill to cell_params
        ecm_params["ecm_stiffness"] = contact_stiffness
        ecm_params["ecm_points"] = ecm_points
        cell_params["ecm_params"] = ecm_params
    # }}}
    # Pressure {{{
    if "pressure" in paramsCell:
        pressureParams = paramsCell["pressure"]
        # Add dynamics
        cell_dynamics.append(pressureParams["dynamics"])
        # Function
        pressure_class = SelectClass(pressureParams["class"], forces)
        pressure_class_kwargs = pressureParams["class_kwargs"]
        # Input
        pressure_func = pressure_class(**pressure_class_kwargs)
        # Fill to cell_params
        cell_params["pressure_func"] = pressure_func
    # }}}
    # }}}
    # Nucleus {{{
    if "nucleus" in params:
        # Initialisation of nucl_params
        nucl_params = {
                # Communicator from MPI
                "comm" : comm,
                # Time step
                "dt" : dt
                }
        # Update membrane dynamics
        cell_dynamics.append("MembraneNucleusDynamics")
        # Get external parameters
        paramsNucleus = params["nucleus"]
        memb_nucl_spring_stiffness = paramsNucleus.get("spring_stiffness", 0.0)
        # Fill to cell and nucleus parameters
        cell_params["spring_stiffness"] = memb_nucl_spring_stiffness
        nucl_params["spring_stiffness"] = memb_nucl_spring_stiffness
        # Contact {{{
        nucl_params["self_tol"] = contact_tol
        nucl_params["self_stiffness"] = contact_stiffness
        # }}}
        # Mesh {{{
        meshparamsNucleus = paramsNucleus.get("mesh", {})
        # Geometry
        radius_nu = meshparamsNucleus.get("radius", radius_ce/2.0)
        radiusRatio_nucl = radius_ce/radius_nu
        # Mesh size
        lc_nu = meshparamsNucleus.get("lc", lc_ce)
        meshOrder_nu = meshparamsNucleus.get("meshOrder", meshOrder_ce)
        meshFunction_nu = SelectFunction(meshparamsNucleus.get("function", "MakeCircle"), mesh_utils)
        # Fill to nucl_params
        aRef_nu = np.pi*radius_nu**2.0
        periRef_nu = 2.0*np.pi*radius_nu
        Href_nu = 1.0/radius_nu
        nucl_params["model_mesh"] = io.gmshio.model_to_mesh(
                meshFunction_nu(radius_nu, lc_nu, meshOrder = meshOrder_nu),
                comm, 0, gdim = 2)
        nucl_params["meshOrder"] = meshOrder_nu
        nucl_params["aRef"] = aRef_nu
        nucl_params["periRef"] = periRef_nu
        nucl_params["Href"] = Href_nu
        # }}}
        # Material {{{
        materialparamsNucleus = paramsNucleus["material"]
        viscosityRatio_nucl = materialparamsNucleus["viscosityRatio"]
        tensionRatio_nucl = materialparamsNucleus["tensionRatio"]
        bendingRatio_nucl = materialparamsNucleus["bendingRatio"]
        peri_max_factor_nucl = materialparamsNucleus.get("peri_max_factor", 1.0)
        area_stiffness_nucl = materialparamsNucleus.get("area_stiffness", area_stiffness_cell*radiusRatio_nucl**3.0/tensionRatio_nucl)
        peri_stiffness_nucl = materialparamsNucleus.get("peri_stiffness", peri_stiffness_cell/tensionRatio_nucl)
        # Fill to nucl_params
        nucl_params["bendingTensionRatio"] = bendingTensionRatio_cell
        nucl_params["area_stiffness"] = area_stiffness_nucl
        nucl_params["peri_stiffness"] = peri_stiffness_nucl
        nucl_params["viscosityRatio"] = viscosityRatio_nucl
        nucl_params["tensionRatio"] = tensionRatio_nucl
        nucl_params["bendingRatio"] = bendingRatio_nucl
        nucl_params["peri_max_factor"] = peri_max_factor_nucl
        # }}}
    # }}}
    # Files {{{
    fileParams = params["files"]
    folder = fileParams["folder"]
    # Fill to cell_params
    cell_params["filename"] = folder + "ce_"
    if "nucleus" in params:
        # Fill to nucl_params
        nucl_params["filename"] = folder + "nu_"
    # }}}
    # Solver {{{
    solverParams = params["solver"]
    quadrature_degree = solverParams.get("quadrature_degree", 8)
    def SetSolverOpt(solver):
        # Newton solver
        solver.convergence_criterion = solverParams.get("convergence_criterion", "incremental")
        solver.rtol = solverParams.get("rtol", 1.0e-8)
        solver.atol = solverParams.get("atol", 1.0e-8)
        solver.max_it = solverParams.get("max_it", 25)
        solver.report = solverParams.get("report", True)
        solver.relaxation_parameter = solverParams.get("relaxation_parameter", 1.0)
        # Krylov solver
        ksp = solver.krylov_solver
        opts = PETSc.Options()
        option_prefix = ksp.getOptionsPrefix()
        opts[f"{option_prefix}ksp_type"]   = "preonly"
        opts[f"{option_prefix}pc_type"]    = "lu"
        opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
        opts[f"{option_prefix}ksp_max_it"] = 1000
        ksp.setFromOptions()
        return
    # Fill to cell_params
    cell_params["quadrature_degree"] = quadrature_degree
    cell_params["SetSolverOpt"] = SetSolverOpt
    if "nucleus" in params:
        # Fill to nucl_params
        nucl_params["quadrature_degree"] = quadrature_degree
        nucl_params["SetSolverOpt"] = SetSolverOpt
    # }}}
    # }}}

    # Set up output {{{
    if rank == 0:
        # Create folder to save results
        folder_path = os.path.dirname("results/" + folder)
        os.makedirs(folder_path, exist_ok = True)
        # Make a copy of the script
        script_name = os.path.basename(__file__)
        shutil.copy2(__file__, os.path.join(folder_path, script_name))
        # Save parameters
        params_path = os.path.join(folder_path, "params.json")
        with open(params_path, 'w') as fle:
            json.dump(params_copy, fle, indent = 4)
        if "ecm" in paramsCell:
            if ecmParams["dynamics"] in "ECMBarrierPointDynamics":
                PlotCircles(ecm_points, ecm_params["ecm_tol"], "results/" + folder + "ecm.vtk")
            if ecmParams["dynamics"] == "ECMBarrierPolygonDynamics":
                PlotPolygons(ecm_points, "results/" + folder + "ecm.vtp")
        # # Plot adhesion points
        # if not adhesionParams is None:
        #     PlotCircles(adhe_locations, adhe_kwargs["size"], "results/" + folder + "adhe.vtk")
    # }}}

    # Define dynamics {{{
    MembraneDynamics = CombineClasses(cell_dynamics, Dynamics)
    cell_params["Dynamics"] = MembraneDynamics
    if "nucleus" in params:
        NucleusDynamics = CombineClasses(nucl_dynamics, Dynamics)
        nucl_params["Dynamics"] = NucleusDynamics
    # }}}

    # Create cell {{{
    if "cort" in params:
        pass
    elif "nucleus" in params:
        cell = Cell(cell_params, nucl_params,
                    contact_stiffness = contact_stiffness,
                    contact_tol = contact_tol)
    else:
        cell = GSPDE(cell_params)
    # }}}

    # Calculation loop {{{
    # Initialisation
    t = 0.0
    cell.WriteResults(t = t)
    cell.WriteReport(t = t)
    mprint("------------------------------------", rank = rank)
    mprint("Simulation Start", rank = rank)
    mprint("------------------------------------", rank = rank)
    startTime = datetime.now()
    printTime0 = datetime.now()
    # Time stepping solution procedure loop
    k1 = 0
    while (round(t + dt, 9) <= Ttot):
        # Update iteration
        k1 += 1
        # Solution
        t += dt
        cell.SolveIteration(t)
        # Write output results
        if k1%print_each_report == 0:
            cell.WriteReport(t)
        if k1%print_each == 0:
            cell.WriteResults(t)
        # Print progress
        printTime1 = datetime.now()
        cpu_time = printTime1 - printTime0
        printTime0 = printTime1
        mprint("------------------------------------", rank = rank)
        mprint("Increment: {} | CPU time: {}".format(k1, cpu_time), rank = rank)
        mprint("dt: {} s | Simulation time {} of {}".format(round(dt, 4), round(t, 4), Ttot), rank = rank)
        mprint("------------------------------------", rank = rank)
    # Close files
    cell.CloseResults()
    # End analysis
    mprint("-----------------------------------------", rank = rank)
    mprint("End computation", rank = rank)
    # Report elapsed real time for the analysis
    endTime = datetime.now()
    elapseTime = endTime - startTime
    mprint("------------------------------------------", rank = rank)
    mprint("Elapsed real time:  {}".format(elapseTime))
    mprint("------------------------------------------", rank = rank)
    # }}}
    return
# }}}

# Execute if main file {{{
if __name__ == "__main__":
    # Get external parameters {{{
    # Message error
    message = 'Wrong call! The execution must have the following arguments:\n'
    message += 'Indicate the parameter codes as:\n'
    message += '-i file name with parameters (.json)\n'
    # Get external variables
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'i:')
        if len(args)>0:
            raise getopt.GetoptError('Too many arguments!')
    except getopt.GetoptError:
        print(message)
        raise getopt.GetoptError("Invalid arguments")
    # Extract parameters
    inParams = False
    for opt, arg in opts:
        # File name
        if opt in ["-i"]:
            parFile = arg
            inParams = True
    if not inParams:
        message = "Wrong call! The input file with the parameters is not present.\n Use -i 'fileName' (.json)"
        raise getopt.GetoptError(message)
    # }}}
    # Read parameters
    with open(parFile, 'r') as inFile:
        params = json.load(inFile)
    # Run single_cell
    single_cell(params)
# }}}
