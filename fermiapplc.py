#!/usr/bin/python

import os
from os.path import join
import sys
from pprint import pprint
import argparse
import numpy as np

import enrico
from enrico.extern.configobj import ConfigObj, flatten_errors
from enrico.extern.validate import Validator
from enrico.environ import CONFIG_DIR, DOWNLOAD_DIR
from enrico import Loggin, utils

from multiprocessing import Pool, Process, Lock
mpool = Pool(5) 

# SOME CONSTANTS
TIMEDELAY=6*3600
PERIOD="p302"
CATALOG="/usr/local/enrico/Data/catalog/gll_psc_v16.fit"

epoch_ref = 1331679600+3600
mjd_ref = 56000.
met_ref = 353376002.000

tofile=None
clearfile=True

# flexible pretty print
def printc(text):
  global tofile
  global clearfile
  if(tofile!=None):
    if (clearfile):
      open(tofile,"w").close()
    outf = open(tofile,"a+")
    pprint(text,outf)
    outf.close()
  pprint(text)
  clearfile=False

def fermilat_data_path():
  basepath = "/".join(enrico.__path__[0].split("/")[0:-1])
  datapath = basepath+"/Data/download/"
  printc("Exploring: "+datapath+"/weekly/photon/")
  weekly_filelist = [datapath+"/weekly/photon/"+F+"\n" \
    for F in os.listdir(datapath+"/weekly/photon/") if "_"+PERIOD+"_" in F]
  
  printc("Adding %d files" %(len(weekly_filelist)))
  with open(datapath+"/event.list", "w+") as f:
    f.writelines(weekly_filelist)

def update_fermilat_data():
  from enrico.data import Data
  printc("-> Getting latest Fermi-LAT weekly files")
  currentdir = os.getcwd()
  data = Data()
  data.download(spacecraft=True, photon=True)
  fermilat_data_path()
  os.chdir(currentdir)

def get_current_datetime():
  from datetime.datetime import utcnow
  return(utcnow())

def epoch2met(epoch):
  return(epoch-epoch_ref+met_ref+3600)

def dt2met(date):
  import datetime
  import time
  pattern = '%Y-%m-%d %H:%M:%S'
  strdt = datetime.datetime.strftime(date,pattern)
  epoch = int(time.mktime(time.strptime(strdt, pattern)))
  return epoch2met(epoch)

def met2dt(date):
  import time
  epoch = met-met_ref+epoch_ref
  pattern = '%Y-%m-%d %H:%M:%S'
  return time.strftime(pattern,time.gmtime(epoch))

# Generate Apperture LC (XX h bins) for the latest YY days of Fermi-LAT data.

##### 1. parse input parameters and write config file
class FermiCatalog(object):
  def __init__(self,path="/usr/local/enrico/Data/catalog/gll_psc_v16.fit"):
    import astropy.io.fits as pyfits
    self.Cat   = pyfits.open(path)
    self.Table = self.Cat[1].data
    self.NameList = []
    column_names  = [0,64,65,66,67,68,69,70,74,75]

    for k in self.Table:
      namelist = np.append(\
        [k[j] for j in column_names],
        [l.replace(" ","") for l in [k[j] for j in column_names]])
      self.NameList.append(namelist)

  def find_source(self,name):
    column_types    = 73
    column_raj2000  = 1
    column_decj2000 = 2
    column_glon     = 3
    column_glat     = 4
    column_pwlindex = 30
    
    src = dict()
    for k in xrange(len(self.NameList)):
      if name in self.NameList[k]:
        src["arg"]      = k
        src["raj2000"]  = self.Table[k][column_raj2000]
        src["decj2000"] = self.Table[k][column_decj2000]
        src["glon"]     = self.Table[k][column_glon]
        src["glat"]     = self.Table[k][column_glat]
        src["pwlindex"] = self.Table[k][column_pwlindex]
        return(src)

    printc("Source name %s not found!." %name) 


class Analysis(object):
  def __init__(self):
    self.current_source=-1
    self.parse_arguments()

  def parse_arguments(self):
    parser = argparse.ArgumentParser(
      description='Create Fermi Apperture LCs for the given objects.')
    parser.add_argument('sourcelistfn', metavar='sourcelistfn',
      type=str, 
      help='Filename with a list of source to analyze')
    parser.add_argument('-o', '--output', dest='output', 
      type=str, default=os.getcwd(), 
      help='Output dir to save products (default: current dir)')
    parser.add_argument('-l', '--log', dest='log', 
      type=str, default=None, 
      help='Output dir to save products (default: current dir)')
    parser.add_argument('-p', '--period', dest='period', 
      type=str, default=PERIOD, 
      help=str('Select period (default: %s)' %PERIOD))
    parser.add_argument('-c', '--catalog', dest='catalog', 
      type=str, default=CATALOG, 
      help=str('Catalog file path (default: %s)' %CATALOG))
    parser.add_argument('-r', '--roi', dest='roi', 
      type=float, default=1, 
      help='Search radius (deg, default: 1deg)')
    parser.add_argument('-b', '--binsize', dest='binsize', 
      type=int, default=6, 
      help='Bin size in hours (default: 6h)')
    parser.add_argument('-t', '--timewindow', dest='timewindow', 
      type=int, default=2, 
      help='Time window in days (default: 2d)')
    parser.add_argument('-emin', '--energymin', dest='energymin', 
      type=float, default=100, 
      help='Minimum energy in MeV (default: 100MeV)')
    parser.add_argument('-emax', '--energymax', dest='energymax', 
      type=float, default=300000, 
      help='Maximum energy in MeV (default: 300000MeV)')
    parser.add_argument('--index', dest='spindex', 
      type=float, default=2.0, 
      help='Spectral Index (assuming PWL) (default: 2.0)')

    args = parser.parse_args()
    vars(self).update(args.__dict__)
    
    # Update log file
    global tofile
    tofile = os.path.abspath(args.log)
    printc("-> Parsing arguments: ")
    #printc(args)
    printc("-> Converting relative paths to absolute")
    self.sourcelistfn = os.path.abspath(self.sourcelistfn)
    self.output = os.path.abspath(self.output)

  def get_list_of_sources(self):
    self.sourcelist = np.loadtxt(self.sourcelistfn, dtype=str, ndmin=1, unpack=True)
    return(len(self.sourcelist))

  def select_next_source(self):
    self.current_source = (self.current_source+1)%len(self.sourcelist)
    self.name = self.sourcelist[self.current_source]
    return(self.name)

  def get_source_coordinates(self,catalog):
    printc("-> Solve source name %s with the Fermi-LAT catalog" %(self.name))
    try:
      source = catalog.find_source(self.name)
      self.RA  = source["raj2000"]
      self.DEC = source["decj2000"]
      return(1)
    except:
      printc("-x Failed to locate source in the Fermi catalog")

    try:
      printc("-> Solve source name %s with SIMBAD")
      from astroquery.simbad import Simbad
      object = Simbad.query_object(self.name)
      RA  = np.array(str(object["RA"].data.data[0]).split(" "),dtype=float)
      DEC = np.array(str(object["DEC"].data.data[0]).split(" "),dtype=float)
      self.RA  = 15.*(RA[0]+RA[1]/60.+RA[2]/3600.)
      self.DEC = (DEC[0]+DEC[1]/60.+DEC[2]/3600.)
      return(1)
    except:
      printc("-x Failed to solve source name with simbad")
    
    printc("Trying name, RA, DEC")
    try:
      self.name, self.RA, self.DEC = np.array(self.name.split(","))
      self.RA  = float(self.RA)
      self.DEC = float(self.DEC)
    except:
      printc("-x Something went wrong while processing %s" %self.name)
    
    sys.exit(1)


  def calculate_times(self):
    import datetime
    current   = datetime.datetime.now()
    available = current-datetime.timedelta(seconds=TIMEDELAY)
    self.timemin = dt2met(available-datetime.timedelta(days=self.timewindow))
    self.timemax = dt2met(available)

  def write_config(self):
    printc("-> Write config file for %s" %(self.name))
    import enrico
    from enrico.config import get_config
    # Create object
    config = ConfigObj(indent_type='\t')
    mes = Loggin.Message()
    config['out']    = self.output+'/%s/'%self.name
    config['Submit'] = 'yes' 
    # target
    config['target'] = {}
    config['target']['name'] = self.name
    config['target']['ra']   = str("%.4s" %self.RA)
    config['target']['dec']  = str("%.4s" %self.DEC)
    config['target']['redshift'] = '0'
    config['target']['spectrum'] = 'PowerLaw'
    # space
    config['space'] = {}
    config['space']['xref'] = config['target']['ra']
    config['space']['yref'] = config['target']['dec']
    config['space']['rad']  = self.roi
    # files
    basepath = "/".join(enrico.__path__[0].split("/")[0:-1])
    datapath = basepath+"/Data/download/"
    config['file'] = {}
    config['file']['spacecraft'] = str("%s/lat_spacecraft_merged.fits" %datapath)
    config['file']['event']      = str("%s/event.list" %datapath)
    config['file']['tag']        = 'fast'
    # time
    config['time'] = {}
    self.calculate_times()
    config['time']['tmin'] = self.timemin
    config['time']['tmax'] = self.timemax
    # energy
    config['energy'] = {}
    config['energy']['emin'] = self.energymin
    config['energy']['emax'] = self.energymax
    # event class
    config['event'] = {}
    config['event']['irfs'] = 'CALDB'
    config['event']['evclass'] = '128'
    config['event']['evtype'] = '3'
    # analysis
    config['analysis'] = {}
    config['analysis']['zmax'] = 100
    # Validate
    printc(config)
    # Add the rest of the values
    config = get_config(config)
    # Tune the remaining variables
    config['AppLC']['index'] = self.spindex
    config['AppLC']['NLCbin'] = int(24*self.timewindow/self.binsize+0.5)
    # Write config file
    self.configfile = str("%s/%s.conf" %(self.output,self.name))
    with open(self.configfile,'w') as f:
      config.write(f)

  def get_data(self):
    from astroquery.fermi import FermiLAT
  
  def remove_prev_dir(self):
    from enrico.constants import AppLCPath
    import shutil
    fname = str("%s/%s/%s/%s_fast_applc.fits" \
        %(self.output, self.name, AppLCPath, self.name))
    shutil.rmtree(fname, ignore_errors=True)

  def run_analysis(self):
    printc("-> Running the analysis for %s" %(analysis.name))
    import shutil
    from enrico import environ
    from enrico.config import get_config
    from enrico.appertureLC import AppLC
    from enrico.submit import call
    from enrico.constants import AppLCPath
    infile = self.configfile
    config = get_config(infile)

    # We will always try to run it in parallel, 
    # either with python multiproc or with external torque pbs.
    if config['Submit'] == 'no':
      global mpool
      mpool.apply_asyncProcess(AppLC, args=(infile,))
      mpool.close()
      #mpool.join()
    else:
      enricodir = environ.DIRS.get('ENRICO_DIR')
      fermidir = environ.DIRS.get('FERMI_DIR')
      cmd = enricodir+"/enrico/appertureLC.py %s" %infile
      LCoutfolder =  config['out']+"/"+AppLCPath
      os.system("mkdir -p "+LCoutfolder)
      prefix = LCoutfolder +"/"+ config['target']['name'] + "_AppertureLightCurve"
      scriptname = prefix+"_Script.sh"
      JobLog = prefix + "_Job.log"
      JobName = "LC_%s" %self.name
      call(cmd, enricodir, fermidir, scriptname, JobLog, JobName)
    
    printc("-> Job sent successfully")

  def arm_triggers(self):
    from multiprocessing import Process, Lock
    #p = Pool(len(self.sourcelist))
    lock = Lock()
    printc("-> Preparing triggers for %s" %(analysis.name))
    for source in self.sourcelist:
      args = (lock, self.output, source)
      Process(target=analyze_results, args=args).start()
    

def analyze_results(lock,outdir,source,triggermode='local',fluxref=None,sigma=3):  
  import time
  from numpy import sum, sqrt
  import astropy.io.fits as pyfits
  from enrico.constants import AppLCPath
  # Retrieve results
  applcfile = str("%s/%s/%s/%s_fast_applc.fits" \
      %(outdir, source, AppLCPath, source))
  
  while(not os.isfile(applcfile)):
    time.sleep(30)
  
  with pyfits.open(applcfile) as F: applc = F[1].data
  total_exp_prev = sum(applc['EXPOSURE'][:-1])
  total_cts_prev = sum(applc['COUNTS'][:-1])
  total_err_prev = sqrt(sum(applc['ERROR'][:-1]**2))
  last_exp = applc['EXPOSURE'][-1]
  last_cts = applc['COUNTS'][-1]
  last_err = applc['ERROR'][-1]

  if (triggermode is 'local'):
    expratio = (1.*last_exp/total_exp_prev)
    proj_cts = ratio*total_cts_prev
    proj_err = ratio*total_err_prev
    if ((last_cts-proj_cts)>sigma*proj_err):
      l.acquire()
      printc("-> ALERT: %s might be flaring! Flux is %.1f times the average"\
        %(source,(last_cts-proj_cts)/proj_err))
      l.release()
      return(True)
    else:
      l.acquire()
      printc("Flux level: %f +/- %f ph/cm2/s" \
          %(last_cts/last_exp, last_err/last_exp))
      l.release()
      return(False)

  

##### 2. Query object in the Fermi LAT datacenter


if __name__ == '__main__':
  analysis = Analysis()
  catalog = FermiCatalog(analysis.catalog)
  #update_fermilat_data()
  #while(analysis.current_source+1<analysis.get_list_of_sources()):
  while(True):
    if (analysis.current_source+1==analysis.get_list_of_sources()):
        break
        #time.sleep(3600)
    analysis.select_next_source()
    analysis.get_source_coordinates(catalog)
    analysis.write_config()
    analysis.run_analysis()

