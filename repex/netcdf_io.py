import os

import numpy as np
import pandas as pd

import simtk.unit as units

import netCDF4 as netcdf
import mdtraj as md

from thermodynamics import ThermodynamicState
from utils import str_to_system
from version import version as __version__

from pymbar import MBAR, timeseries

import logging
logger = logging.getLogger(__name__)


class NetCDFDatabase(object):
    """A netCDF based database for Repex simulations.
    
    
    Parameters
    ----------
    filename : str
        The filename of the database.  Use ".nc" as the suffix.
    thermodynamic_states : list(ThermodynamicState), default=None
        A list of thermodynamic states for the replica exchange run.  
        These will be saved to disk.  If None, will be loaded from disk.
    positions : list(simtk.unit)
        A list of coordinates for each repex slot.  If None, will be 
        loaded from disk.
    
    """
    def __init__(self, filename, thermodynamic_states=None, positions=None):

        # Check if netcdf file exists.
        resume = os.path.exists(filename) and (os.path.getsize(filename) > 0)
        
        if resume:
            assert thermodynamic_states is None and positions is None, "Cannot input thermodynamic_states and positions if you are resuming from disk."
            self.ncfile = netcdf.Dataset(filename, 'a', version='NETCDF4')

        else:
            assert thermodynamic_states is not None and positions is not None, "Must input thermodynamic_states and coordinates if no existing database."
            assert len(thermodynamic_states) == len(positions), "Must have same number of thermodynamic_states and coordinate sets."
            
            self.ncfile = netcdf.Dataset(filename, 'w', version='NETCDF4')
        
        self.title = "No Title."
        
        if resume:
            logger.info("Attempting to resume by reading thermodynamic states and options...")
            self.parameters = self._load_parameters()
        else:
            self._initialize_netcdf(thermodynamic_states, positions)

        # Check to make sure all states have the same number of atoms and are in the same thermodynamic ensemble.
        for state in self.thermodynamic_states:
            if not state.is_compatible_with(self.thermodynamic_states[0]):
                raise ValueError("Provided ThermodynamicState states must all be from the same thermodynamic ensemble.")

    def set_traj(self, traj):
        """Set a trajectory object for extracting trajectory slices.
        
        Parameters
        ----------
        
        traj : mdtraj.Trajectory
            A trajectory object whose topology is identical to your system.
        """
        if self.n_atoms != traj.n_atoms:
            raise(ValueError("Trajectory must have %d atoms, found %d!" % (self.n_atoms, traj.n_atoms)))

        self._traj = traj
        
    def get_traj(self, state_index=None, replica_index=None):
        """Extract a trajectory slice along a replica or state index.
        
        Parameters
        ----------
        state_index : int, optional
            Extract trajectory from this thermodynamic state
        replica_index : int, optional
            Extract trajectory from this replica slot.
        
        Returns
        -------
        traj : mdtraj.Trajectory
            A trajectory object containing the desired slice of data.            
        
        Notes
        -----
        You must specify exactly ONE of `state_index` or `replica_index`.
        
        This function is a memory hog.
        
        To do: allow Trajectory as input to __init__ and automatically call set_traj().
        To do: state_index code might be slow, possibly amenable to cython.
        """
        if not hasattr(self, "_traj"):
            raise(IOError("You must first specify a compatible trajectory with set_traj()."))

        if not (type(state_index) is int or type(replica_index) is int):
            raise(ValueError("Must input either state_index or replica_index (as integer)."))
        
        if state_index is not None and replica_index is not None:
            raise(ValueError("Cannot input both state_index and replica_index"))
        
        if state_index is not None and (state_index < 0 or state_index >= self.n_states):
            raise(ValueError("Input must be between 0 and %d" % self.n_states))

        if replica_index is not None and (replica_index < 0 or replica_index >= self.n_states):
            raise(ValueError("Input must be between 0 and %d" % self.n_states))
        
        if replica_index is not None:
            xyz = self.positions[:, replica_index]
            box_vectors = self.box_vectors[:, replica_index]

        if state_index is not None:
            replica_indices = self.states[:].argsort()[:, state_index]
            n = len(replica_indices)
            xyz = np.array([self.positions[i, replica_indices[i]] for i in range(n)])  # Have to do this because netcdf4py breaks numpy fancy indexing and calls it a "feature" because it looks "more like fortran".
            box_vectors = np.array([self.box_vectors[i, replica_indices[i]] for i in range(n)])

        traj = md.Trajectory(xyz, self._traj.top)
        traj.unitcell_vectors = box_vectors

        return traj


    def _initialize_netcdf(self, thermodynamic_states, positions):
        """Initialize NetCDF file for storage by allocating arrays on disk.
        
        Parameters
        ----------

        thermodynamic_states : list(ThermodynamicState), default=None
            A list of thermodynamic states for the replica exchange run.  
            These will be saved to disk.  If None, will be loaded from disk.
        positions : list(simtk.unit)
            A list of coordinates for each repex slot.  If None, will be 
            loaded from disk.
        
        """
        
        n_replicas = len(thermodynamic_states)
        n_atoms = len(positions[0])
        n_states = len(thermodynamic_states)
        
        
        # Create dimensions.
        self.ncfile.createDimension('iteration', 0) # unlimited number of iterations
        self.ncfile.createDimension('replica', n_replicas) # number of replicas
        self.ncfile.createDimension('atom', n_atoms) # number of atoms in system
        self.ncfile.createDimension('spatial', 3) # number of spatial dimensions

        # Set global attributes.
        self.ncfile.title = self.title
        self.ncfile.application = 'Repex'
        #self.ncfile.program = 'yank.py'
        self.ncfile.programVersion = __version__
        self.ncfile.Conventions = 'Repex'
        self.ncfile.ConventionVersion = '0.1'
        
        # Allocate variable for the ReplicaExchange (sub)class name.
        self.ncfile.repex_classname = "Unknown"
        
        # Create variables.
        ncvar_positions = self.ncfile.createVariable('positions', 'f', ('iteration','replica','atom','spatial'))
        ncvar_states    = self.ncfile.createVariable('states', 'i', ('iteration','replica'))
        ncvar_energies  = self.ncfile.createVariable('energies', 'f', ('iteration','replica','replica'))        
        ncvar_proposed  = self.ncfile.createVariable('proposed', 'l', ('iteration','replica','replica'))
        ncvar_accepted  = self.ncfile.createVariable('accepted', 'l', ('iteration','replica','replica'))                
        ncvar_box_vectors = self.ncfile.createVariable('box_vectors', 'f', ('iteration','replica','spatial','spatial'))        
        ncvar_volumes  = self.ncfile.createVariable('volumes', 'f', ('iteration','replica'))
        
        # Define units for variables.
        ncvar_positions.units = 'nm'
        ncvar_states.units = 'none'
        ncvar_energies.units = 'kT'
        ncvar_proposed.units = 'none'
        ncvar_accepted.units = 'none'
        ncvar_box_vectors.units = 'nm'
        ncvar_volumes.units = 'nm**3'

        # Define long (human-readable) names for variables.
        ncvar_positions.long_name = "positions[iteration][replica][atom][spatial] is position of coordinate 'spatial' of atom 'atom' from replica 'replica' for iteration 'iteration'."
        ncvar_states.long_name = "states[iteration][replica] is the state index (0..n_states-1) of replica 'replica' of iteration 'iteration'."
        ncvar_energies.long_name = "energies[iteration][replica][state] is the reduced (unitless) energy of replica 'replica' from iteration 'iteration' evaluated at state 'state'."
        ncvar_proposed.long_name = "proposed[iteration][i][j] is the number of proposed transitions between states i and j from iteration 'iteration-1'."
        ncvar_accepted.long_name = "accepted[iteration][i][j] is the number of proposed transitions between states i and j from iteration 'iteration-1'."
        ncvar_box_vectors.long_name = "box_vectors[iteration][replica][i][j] is dimension j of box vector i for replica 'replica' from iteration 'iteration-1'."
        ncvar_volumes.long_name = "volume[iteration][replica] is the box volume for replica 'replica' from iteration 'iteration-1'."

        # Create timestamp variable.
        ncvar_timestamp = self.ncfile.createVariable('timestamp', "f", ('iteration',))

        # Create group for performance statistics.
        ncgrp_timings = self.ncfile.createGroup('timings')
        ncvar_iteration_time = ncgrp_timings.createVariable('iteration', 'f', ('iteration',)) # total iteration time (seconds)
        ncvar_iteration_time = ncgrp_timings.createVariable('mixing', 'f', ('iteration',)) # time for mixing
        ncvar_iteration_time = ncgrp_timings.createVariable('propagate', 'f', ('iteration','replica')) # total time to propagate each replica
        
        # Store thermodynamic states using @property setter
        self.thermodynamic_states = thermodynamic_states

        # Force sync to disk to avoid data loss.
        self.ncfile.sync()


    @property
    def thermodynamic_states(self):
        """Return the thermodynamic states from a NetCDF file.
        
        Returns
        -------
        thermodynamic_states : list
            List of thermodynamic states for repex.
        """
        
        logging.debug("Restoring thermodynamic states from NetCDF file...")
           
        # Make sure this NetCDF file contains thermodynamic state information.
        if not 'thermodynamic_states' in self.ncfile.groups:
            raise(ValueError("thermodynamics_states not found in database!"))

        # Create a group to store state information.
        ncgrp_stateinfo = self.ncfile.groups['thermodynamic_states']

        # Get number of states.
        n_states = int(ncgrp_stateinfo.variables['n_states'][0])

        # Read state information.
        thermodynamic_states = list()
        for state_index in range(n_states):
                        
            temperature = ncgrp_stateinfo.variables['temperatures'][state_index] * units.kelvin
            
            pressure = None
            if 'pressures' in ncgrp_stateinfo.variables:
                pressure = ncgrp_stateinfo.variables['pressures'][state_index] * units.atmospheres
            
            # Reconstitute System object.
            system = str(ncgrp_stateinfo.variables['systems'][state_index])
            system = str_to_system(system)
            
            state = ThermodynamicState(system=system, temperature=temperature, pressure=pressure)

            thermodynamic_states.append(state)
        
        return thermodynamic_states


    @thermodynamic_states.setter
    def thermodynamic_states(self, thermodynamic_states):
        """Store the thermodynamic states in a NetCDF file.
        """
        logger.debug("Storing thermodynamic states in NetCDF file...")

        if self.ncfile.groups.has_key("thermodynamic_states"):
            raise(IOError("Thermodynamic states have already been set!"))

        # Create a group to store state information.
        ncgrp_stateinfo = self.ncfile.createGroup('thermodynamic_states')

        # Get number of states.
        ncvar_n_states = ncgrp_stateinfo.createVariable('n_states', int)
        ncvar_n_states.assignValue(self.n_states)


        # Temperatures.
        ncvar_temperatures = ncgrp_stateinfo.createVariable('temperatures', 'f', ('replica',))
        ncvar_temperatures.units = 'K'
        ncvar_temperatures.long_name = "temperatures[state] is the temperature of thermodynamic state 'state'"
        for state_index in range(self.n_states):
            ncvar_temperatures[state_index] = thermodynamic_states[state_index].temperature / units.kelvin

        # Pressures.
        if thermodynamic_states[0].pressure is not None:
            ncvar_temperatures = ncgrp_stateinfo.createVariable('pressures', 'f', ('replica',))
            ncvar_temperatures.units = 'atm'
            ncvar_temperatures.long_name = "pressures[state] is the external pressure of thermodynamic state 'state'"
            for state_index in range(self.n_states):
                ncvar_temperatures[state_index] = thermodynamic_states[state_index].pressure / units.atmospheres

        # TODO: Store other thermodynamic variables store in ThermodynamicState?  Generalize?
                
        # Systems.
        ncvar_serialized_states = ncgrp_stateinfo.createVariable('systems', str, ('replica',), zlib=True)
        ncvar_serialized_states.long_name = "systems[state] is the serialized OpenMM System corresponding to the thermodynamic state 'state'"
        for state_index in range(self.n_states):
            logger.debug("Serializing state %d..." % state_index)
            serialized = thermodynamic_states[state_index].system.__getstate__()
            logger.debug("Serialized state is %d B | %.3f KB | %.3f MB" % (len(serialized), len(serialized) / 1024.0, len(serialized) / 1024.0 / 1024.0))
            ncvar_serialized_states[state_index] = serialized


    def store_parameters(self, parameters):
        """Store run parameters in NetCDF file.
        
        Parameters
        ----------
        parameters : dict
            A dict containing ALL run parameters
            
        Notes
        -----
        
        The user supplies a dictionary of run parameters to Repex.  
        Repex (or a subclass) then fills in any missing parameters.  
        This dictionary is the correct input here.  Eventually, this dictionary
        will be converted to a namedtuple, at which point it is immutable.  
        
        """

        logger.debug("Storing run parameters in NetCDF file...")

        # Create scalar dimension if not already present.
        if 'scalar' not in self.ncfile.dimensions:
            self.ncfile.createDimension('scalar', 1) # scalar dimension
        
        # Create a group to store state information.
        ncgrp_options = self.ncfile.createGroup('options')

        for option_name, option_value in parameters.iteritems():
            self._store_parameter(option_name, option_value)

    def _store_parameter(self, option_name, option_value):
        # If Quantity, strip off units first.
        option_unit = None
        if type(option_value) == units.Quantity:
            option_unit = option_value.unit
            option_value = option_value / option_unit
        # Store the Python type.
        option_type = type(option_value)
        
        # Handle NoneType
        if type(option_value) == type(None):
            option_value = ""
        # Handle booleans
        if type(option_value) == bool:
            option_value = int(option_value)
        # Store the variable.
        logger.debug("Storing option: %s -> %s (type: %s)" % (option_name, option_value, str(option_type)))
        if type(option_value) == str:
            if option_name in self.ncfile.groups['options'].variables:
                ncvar = self.ncfile.groups['options'].variables[option_name]
            else:
                ncvar = self.ncfile.groups['options'].createVariable(option_name, type(option_value), 'scalar')
            packed_data = np.empty(1, 'O')
            packed_data[0] = option_value
            ncvar[:] = packed_data
        else:
            if option_name in self.ncfile.groups['options'].variables:
                ncvar = self.ncfile.groups['options'].variables[option_name]
            else:
                ncvar = self.ncfile.groups['options'].createVariable(option_name, type(option_value))
            ncvar.assignValue(option_value)
        if option_unit: setattr(ncvar, 'units', str(option_unit))
        setattr(ncvar, 'type', option_type.__name__)

    def _load_parameter(self, option_name):
        
        option_ncvar = self.ncfile.groups['options'].variables[option_name]
        # Get option value.
        option_value = option_ncvar[0]
        # Cast to Python type.
        type_name = getattr(option_ncvar, 'type')
        
        if type_name == "NoneType":
            return None
        
        else:
            option_value = eval(type_name + '(' + repr(option_value) + ')')
            # If Quantity, assign units.
            if hasattr(option_ncvar, 'units'):
                option_unit_name = getattr(option_ncvar, 'units')
                if option_unit_name[0] == '/': option_unit_name = '1' + option_unit_name
                option_unit = eval(option_unit_name, vars(units))
                option_value = units.Quantity(option_value, option_unit)
            # Store option.
            logger.debug("Restoring option: %s -> %s (type: %s)" % (option_name, str(option_value), type(option_value)))
            #setattr(self, option_name, option_value)
            return option_value 

    def _load_parameters(self):
        """Return a dictionary of options from a loaded NetCDF file.
        """
        
        logger.debug("Attempting to restore options from NetCDF file...")
        
        # Make sure this NetCDF file contains option information
        if not 'options' in self.ncfile.groups:
            # Not found, signal failure.
            raise(ValueError("Options not found in NetCDF file!"))

        ncgrp_options = self.ncfile.groups['options']
        
        options = {}
        for option_name in ncgrp_options.variables.keys():
            options[option_name] = self._load_parameter(option_name)

        return options


    def write(self, key, value, iteration, sync=True):
        """Write a variable to the database and sync."""
        
        self.ncfile.variables[key][iteration] = value
        
        if sync == True:
            self.sync()


    def sync(self):
        """Sync the database."""
        self.ncfile.sync()
    
    def _finalize(self):
        self.sync()

    def __del__(self):
        self._finalize()
        self.ncfile.close()

    @property
    def positions(self):
        """Return the positions."""
        return self.ncfile.variables['positions']

    @property
    def box_vectors(self):
        """Return the box vectors."""
        return self.ncfile.variables['box_vectors']
                    
    @property
    def volumes(self):
        """Return the volumes."""
        return self.ncfile.variables['volumes']

    @property
    def states(self):
        """Return the state indices."""
        return self.ncfile.variables['states']
        
    @property
    def energies(self):
        """Return the energies."""
        return self.ncfile.variables['energies']
                                    
    @property
    def proposed(self):
        """Return the proposed moves."""
        return self.ncfile.variables['proposed']
        
    @property
    def accepted(self):
        """Return the accepted moves."""
        return self.ncfile.variables['accepted']

    @property
    def timestamp(self):
        """Return the timestamp."""
        return self.ncfile.variables['timestamp']

    @property
    def repex_classname(self):
        return self.ncfile.repex_classname

    @property
    def last_proposed(self):
        iteration = self.last_iteration
        return self.proposed[iteration]

    @property
    def last_accepted(self):
        iteration = self.last_iteration
        return self.accepted[iteration]

    @property
    def last_box_vectors(self):
        """Return data from the last iteration saved in a database.
        
        Returns
        -------
        
        replica_coordinates : list
            The coordinates of each replica
        """
        
        iteration = self.last_iteration
        
        replica_box_vectors = list()
        for replica_index in range(self.n_states):
            x = self.ncfile.variables['box_vectors'][iteration, replica_index,:,:].astype(np.float64).copy()
            box_vectors = units.Quantity(x, units.nanometers)
            replica_box_vectors.append(box_vectors)        
        
        return replica_box_vectors
        
    @property
    def last_u_kl(self):
        """Return data from the last iteration saved in a database.
        
        Returns
        -------
        
        replica_coordinates : list
            The coordinates of each replica
        """
        iteration = self.last_iteration
        return self.energies[iteration,:].copy()

    
    @property
    def last_iteration(self):
        """Return data from the last iteration saved in a database.
        
        Returns
        -------
        
        replica_coordinates : list
            The coordinates of each replica
        """
        return self.positions.shape[0] - 1

    @property
    def last_positions(self):
        """Return positions from the last iteration saved in a database.
        
        Returns
        -------
        
        replica_coordinates : list
            The coordinates of each replica
            
        Notes
        -----
        Formatted for input to Repex
        """    
        
        iteration = self.last_iteration
        
        replica_coordinates = list()
        for replica_index in range(self.n_states):
            x = self.positions[iteration, replica_index, :, :].astype(np.float64).copy()
            coordinates = units.Quantity(x, units.nanometers)
            replica_coordinates.append(coordinates)
        
        return replica_coordinates

    @property
    def last_replica_states(self):
        
        iteration = self.last_iteration
        return self.states[iteration,:].copy()


    @property
    def n_states(self):
        return self.positions.shape[1]

    
    @property
    def n_atoms(self):
        return self.positions.shape[2]

    def extract_reordered(self, trace):
        """Return a given quantity re-sorted by thermodynamic state, rather than replica.
        
        Parameters
        ----------
        trace : database property
            The netcdf variable to sort by state.

        Returns
        -------
        reordered : np.ndarray
            The reordered variable, in memory.
        
        Notes
        -----
        
        1.  This is a memory hog.
        2.  Be aware that there are funny slicing differences between netCDF and numpy.  
        This function *should* work for both ndarray and netcdf inputs, but 
        be aware.
        """
        replica_indices = self.states[:].argsort()
        n = len(replica_indices)
        reordered = np.array([trace[i, replica_indices[i]] for i in range(n)])  # Have to do this because netcdf4py breaks numpy fancy indexing and calls it a "feature" because it looks "more like fortran".
        
        return reordered


    def check_energies(self):
        """Examine energy history for signs of instability (nans)."""

        # Get current dimensions.
        n_iterations = self.energies.shape[0]

        logger.info("Reading energies...")
        u_kln_replica = np.zeros([self.n_states, self.n_states, n_iterations], np.float64)
        for n in range(n_iterations):
            u_kln_replica[:, :, n] = self.energies[n]
        logger.info("Done.")

        logger.info("Deconvoluting replicas...")
        u_kln = np.zeros([self.n_states, self.n_states, n_iterations], np.float64)
        for iteration in range(n_iterations):
            state_indices = self.states[iteration]
            u_kln[state_indices, :, iteration] = self.energies[iteration]
        logger.info("Done.")

        # If no energies are 'nan', we're clean.
        if not np.any(np.isnan(self.energies[:])):
            return

        # There are some energies that are 'nan', so check if the first iteration has nans in their *own* energies:
        u_k = np.diag(self.energies[0])
        if np.any(np.isnan(u_k)):
            logger.info("First iteration has exploded replicas.  Check to make sure structures are minimized before dynamics")
            logger.info("Energies for all replicas after equilibration:")
            logger.info(u_k)
            return

        # There are some energies that are 'nan' past the first iteration.  Find the first instances for each replica and write PDB files.
        first_nan_k = np.zeros([self.n_states], np.int32)
        for iteration in range(n_iterations):
            for k in range(self.n_states):
                if np.isnan(self.energies[iteration, k, k]) and first_nan_k[k] == 0:
                    first_nan_k[k] = iteration
        
        if not all(first_nan_k == 0):
            logger.info("Some replicas exploded during the simulation.")
            logger.info("Iterations where explosions were detected for each replica:")
            logger.info(first_nan_k)
            logger.info("Writing PDB files immediately before explosions were detected...")
            for replica in range(self.n_states):            
                if (first_nan_k[replica] > 0):
                    state = self.states[iteration,replica]
                    iteration = first_nan_k[replica] - 1
                    filename = 'replica-%d-before-explosion.pdb' % replica

                    # TO DO: output frame as pdb
                    #write_pdb(atoms, filename, iteration, replica, title, ncfile)

        # There are some energies that are 'nan', but these are energies at foreign lambdas.  We'll just have to be careful with MBAR.
        # Raise a warning.
        logger.info("WARNING: Some energies at foreign lambdas are 'nan'.  This is recoverable.")            


    def check_positions(self):
        """Make sure no positions are nan."""

        # Get current dimension.
        n_iterations = self.positions.shape[0]

        for iteration in range(n_iterations):
            for replica in range(self.n_states):
                positions = self.positions[iteration, replica]
                # Check for nan
                if np.any(np.isnan(positions)):
                    # Nan found -- raise error
                    logger.info("Iteration %d, state %d - nan found in positions." % (iteration, replica))
                    # Report coordinates
                    for atom_index in range(self.n_atoms):
                        logger.info("%16.3f %16.3f %16.3f" % (positions[atom_index, 0], positions[atom_index, 1], positions[atom_index, 2]))
                        if np.any(np.isnan(positions[atom_index])):
                            logger.info("nan detected in positions")


    def get_u_kln(self):
        """Extract energies by thermodynamic state.


        Returns
        -------
        
        u_kln : np.ndarray, shape=(n_states, n_states, n_iterations)
            The statewise energies
        """

        # Get current dimensions.
        n_iterations = self.positions.shape[0]

        u_kln_replica = np.transpose(self.energies, (1, 2, 0)).astype('float64')
        u_kln = np.transpose(self.extract_reordered(self.energies), (1, 2, 0)).astype('float64')

        # Compute total negative log probability over all iterations.
        u_n = u_kln.diagonal(axis1=0, axis2=1).sum(1)
        
        return u_kln_replica, u_kln, u_n
        
    def run_mbar(self, ndiscard=0, nuse=None):
        """Estimate free energies of all alchemical states.

        Parameters
        ----------
        ndiscard : int, optinoal, default=0
            number of iterations to discard to equilibration
        nuse : int, optional, default=None
            maximum number of iterations to use (after discarding)

        Returns
        -------
        
        Deltaf_ij : np.ndarray, shape=(n_states, n_states)
            The statewise free energy differences

        dDeltaf_ij : np.ndarray, shape=(n_states, n_states)
            The statewise free energy difference uncertainties

        """    
        
        u_kln_replica, u_kln, u_n = self.get_u_kln()

        u_kln_replica, u_kln, u_n, N_k, N = self.equilibrate_and_subsample(u_kln_replica, u_kln, u_n, ndiscard=ndiscard, nuse=nuse)

        logger.info("Initialing MBAR and computing free energy differences...")
        mbar = MBAR(u_kln, N_k, verbose = False, method = 'self-consistent-iteration', maximum_iterations = 50000) # use slow self-consistent-iteration (the default)

        # Get matrix of dimensionless free energy differences and uncertainty estimate.
        logger.info("Computing covariance matrix...")
        (Deltaf_ij, dDeltaf_ij) = mbar.getFreeEnergyDifferences(uncertainty_method='svd-ew')
       
        logger.info("\n%-24s %16s\n%s" % ("Deltaf_ij", "current state", pd.DataFrame(Deltaf_ij).to_string()))
        logger.info("\n%-24s %16s\n%s" % ("Deltaf_ij", "current state", pd.DataFrame(dDeltaf_ij).to_string()))        
                
        return (Deltaf_ij, dDeltaf_ij)

    def equilibrate_and_subsample(self, u_kln_replica, u_kln, u_n, ndiscard=0, nuse=None):
        """Equilibrate, truncate, and subsample uncorrelated samples.

        Parameters
        ----------
        ndiscard : int, optinoal, default=0
            number of iterations to discard to equilibration
        nuse : int, optional, default=None
            maximum number of iterations to use (after discarding)
        
        Returns
        -------
        """
        
        logger.info("Discarding initial data as equilibration (ndiscard = %d)" % ndiscard)
        u_kln_replica = u_kln_replica[:,:,ndiscard:]
        u_kln = u_kln[:,:, ndiscard:]
        u_n = u_n[ndiscard:]

        
        if nuse is not None:
            logger.info("Truncating to number of specified conforamtions to use(nuse = %d)" % nuse)
            u_kln_replica = u_kln_replica[:,:,0:nuse]
            u_kln = u_kln[:,:,0:nuse]
            u_n = u_n[0:nuse]
        
        logger.info("Subsample data to obtain uncorrelated samples")
        N_k = np.zeros(self.n_states, np.int32)    
        indices = timeseries.subsampleCorrelatedData(u_n) # indices of uncorrelated samples

        N = len(indices) # number of uncorrelated samples
        N_k[:] = N      
        u_kln[:, :, 0:N] = u_kln[:, :, indices]
        logger.info("number of uncorrelated samples:")
        logger.info(N_k)
        logger.info("")
        
        return u_kln_replica, u_kln, u_n, N_k, N


    def estimate_enthalpies(self, ndiscard=0, nuse=None):
        """Estimate average enthalpies of all alchemical states.

        Parameters
        ----------
        ndiscard : int, optional, default=0
            number of iterations to discard to equilibration
        nuse : int, optional, default=None
            maximum number of iterations to use (after discarding)
        
        """
        
        n_iterations = self.positions.shape[0]
        
        u_kln_replica, u_kln, u_n = self.get_u_kln()
        u_kln_replica, u_kln, u_n, N_k, N = self.equilibrate_and_subsample(u_kln_replica, u_kln, u_n, ndiscard=ndiscard, nuse=nuse)

        # Compute average enthalpies.
        H_k = np.zeros([self.n_states], np.float64) # H_i[i] is estimated enthalpy of state i
        dH_k = np.zeros([self.n_states], np.float64)
        for k in range(self.n_states):
            H_k[k] = u_kln[k,k,:].mean()
            dH_k[k] = u_kln[k,k,:].std() / np.sqrt(N)

        return (H_k, dH_k)


    def _accumulate_mixing_statistics(self):
        """Return the mixing transition matrix Tij."""
        try:
            return self._accumulate_mixing_statistics_update()
        except AttributeError:
            pass
        except ValueError:
            logger.info("Inconsistent transition count matrix detected, recalculating from scratch.")

        return self._accumulate_mixing_statistics_full()

    def _accumulate_mixing_statistics_full(self):
        """Compute statistics of transitions iterating over all iterations of repex."""
        self._Nij = np.zeros([self.n_states, self.n_states], np.float64)
        for iteration in range(self.states.shape[0] - 1):
            for ireplica in range(self.n_states):
                istate = self.states[iteration, ireplica]
                jstate = self.states[iteration + 1, ireplica]
                self._Nij[istate, jstate] += 0.5
                self._Nij[jstate, istate] += 0.5
        
        Tij = np.zeros([self.n_states, self.n_states], np.float64)
        for istate in range(self.n_states):
            Tij[istate] = self._Nij[istate] / self._Nij[istate].sum()
        
        return Tij
    
    def _accumulate_mixing_statistics_update(self):
        """Compute statistics of transitions updating Nij of last iteration of repex."""

        if self._Nij.sum() != (self.states.shape[0] - 2) * self.n_states:  # n_iter - 2 = (n_iter - 1) - 1.  Meaning that you have exactly one new iteration to process.
            raise(ValueError("Inconsistent transition count matrix detected.  Perhaps you tried updating twice in a row?"))

        for ireplica in range(self.n_states):
            istate = self.states[-2, ireplica]
            jstate = self.states[-1, ireplica]
            self._Nij[istate, jstate] += 0.5
            self._Nij[jstate, istate] += 0.5

        Tij = np.zeros([self.n_states, self.n_states], np.float64)
        for istate in range(self.n_states):
            Tij[istate] = self._Nij[istate] / self._Nij[istate].sum()
        
        return Tij


    def _show_mixing_statistics(self):
        Tij = self._accumulate_mixing_statistics()

        P = pd.DataFrame(Tij)
        logger.info("\nCumulative symmetrized state mixing transition matrix:\n%s" % P.to_string())

        # Estimate second eigenvalue and equilibration time.
        mu = np.linalg.eigvals(Tij)
        mu = -np.sort(-mu) # sort in descending order
        if (mu[1] >= 1):
            logger.info("\nPerron eigenvalue is unity; Markov chain is decomposable.")
        else:
            logger.info("\nPerron eigenvalue is %9.5f; state equilibration timescale is ~ %.1f iterations" % (mu[1], 1.0 / (1.0 - mu[1])))

