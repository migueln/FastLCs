#!/usr/bin/python

import os
from os.path import join
import sys
from pprint import pprint
import argparse
import numpy as np

import urllib2

import time
import enrico
from enrico.appertureLC import AppLC
from enrico.extern.configobj import ConfigObj, flatten_errors
from enrico.extern.validate import Validator
from enrico.environ import CONFIG_DIR, DOWNLOAD_DIR
from enrico import Loggin, utils

from multiprocessing import Queue, Pool #Pool, Queue, Semaphore, Process, Lock

# SOME CONSTANTS
NTRIALS=10
UPDATE_FERMI=False
TIMEDELAY=3*3600
PERIOD="p302"
CATALOG="/usr/local/enrico/Data/catalog/gll_psc_v16.fit"

epoch_ref = 1331679600+3600
mjd_ref = 56000.
met_ref = 353376002.000

tofile=None
clearfile=True
verbosemode=None

def verbose(message):
  if (verbosemode): printc("[VERBOSE] "+message)

# Use Torque PBS
jobscheduler=False

# Internal job scheduler (multiprocessing)
fqueue   = None
rqueue   = None
fpool    = None
rpool    = None
catalog  = None
analysis = None

def worker(fqueue,rqueue):
  import time
  verbose("%s busy" %(str(os.getpid())))
  while(True):
    try:
      if (not fqueue.empty()):
        verbose("fqueue is not empty")
        item = fqueue.get(True)
        AppLC(item)
      elif (not rqueue.empty()):
        args = rqueue.get(True)
        verbose("rqueue is not empty: %s" %(str(args)))
        if (args=="stop"):
          verbose("Finished signal received, breaking the loop")
          break
        else:
          analyze_results(**args)
      time.sleep(1)
    except KeyboardInterrupt:
      break

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
  verbose("Exploring: "+datapath+"/weekly/photon/")
  weekly_filelist = [datapath+"/weekly/photon/"+F+"\n" \
    for F in os.listdir(datapath+"/weekly/photon/") if "_"+PERIOD+"_" in F]

  verbose("Adding %d files" %(len(weekly_filelist)))
  with open(datapath+"/event.list", "w+") as f:
    f.writelines(weekly_filelist)

def update_fermilat_data():
  from enrico.data import Data
  verbose("-> Getting latest Fermi-LAT weekly files")
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
      if name in str(self.NameList[k]):
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
    self.parse_arguments()
    self.current_source = -1
    self.band='fu'

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
      type=str, default=os.devnull,
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
    parser.add_argument('-sm', '--simthreads', dest='simthreads',
      type=int, default=1,
      help='Simultaneous threads / jobs to process')
    parser.add_argument('-b', '--binsize', dest='binsize',
      type=int, default=12,
      help='Bin size in hours (default: 6h)')
    parser.add_argument('-t', '--timewindow', dest='timewindow',
      type=int, default=4,
      help='Time window in days (default: 2d)')
    parser.add_argument('-emin', '--energymin', dest='energymin',
      type=float, default=100,
      help='Minimum energy in MeV (default: 100MeV)')
    parser.add_argument('-ecut', '--energycut', dest='energycut',
      type=float, default=1000,
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
    verbose("-> Parsing arguments")
    #printc(args)
    verbose("-> Converting relative paths to absolute")
    self.sourcelistfn = os.path.abspath(self.sourcelistfn)
    self.output = os.path.abspath(self.output)

  def get_list_of_sources(self):
    self.sourcelist = np.loadtxt(self.sourcelistfn, dtype=str, \
        ndmin=1, unpack=True)
    return(len(self.sourcelist))

  def select_next_source(self):
    self.get_list_of_sources()
    #result = (self.current_source+1)%(len(self.sourcelist)+1)
    self.current_source = (self.current_source+1)%(len(self.sourcelist)+1)
    self.name = self.sourcelist[self.current_source%len(self.sourcelist)]
    return(self.current_source)

  def query_fermilat_data(self,retrytimes=5):
    import urllib2
    from astroquery.fermi import FermiLAT
    object=self.name
    Erange = str("%d, %d" %(self.energymin,self.energymax))
    LATdatatype = "Extended"
    self.calculate_times()
    timesys = 'MET'
    obsdates = str("%s, %s" %(str(self.timemin),"END")) #str(self.timemax)))
    verbose("Query:")
    verbose(str("%s, %s, %s, %s, %s, %s" \
        %(object, Erange, self.roi, LATdatatype, obsdates, timesys)))

    result = FermiLAT.query_object(\
        name_or_coords=object,\
        energyrange_MeV=Erange,\
        searchradius=self.roi,\
        LATdatatype=LATdatatype,\
        obsdates=obsdates,\
        timesys=timesys)

    for url in result:
      verbose("Downloading file %s" %url.split("/")[-1])
      u = urllib2.urlopen(url)
      with open("%s/%s" %(self.output,url.split("/")[-1]), "wb") as f:
        f.write(u.read())

    try: 
      self.SCfile = [str("%s/%s" %(self.output,r.split("/")[-1])) \
          for r in result if "_SC" in r][0]
      self.PHlist = [str("%s/%s" %(self.output,r.split("/")[-1])) \
          for r in result if "_PH" in r or "_EV" in r]
    except:
      if (retrytimes==0):
        printc("Error. Cannot resolv source %s" %object)
        sys.exit(1)
      else:
        verbose("Retrying %s, left %d" %(object, retrytimes-1))
        self.query_fermilat_data(retrytimes-1)
        return
      
    self.PHfile = str("%s/phfiles_%s.txt" %(self.output,self.name))
    with open(self.PHfile,'w') as f:
      f.write(str('\n'.join(self.PHlist)+'\n'))

  def get_source_coordinates(self,catalog):
    verbose("-> Identifying the source")
    verbose("-> Solve source name %s with the Fermi-LAT catalog" %(self.name))
    try:
      source = catalog.find_source(self.name)
      self.RA  = source["raj2000"]
      self.DEC = source["decj2000"]
      return(1)
    except:
      verbose("-> Failed to locate source in the Fermi catalog")

    try:
      verbose("-> Solve source name %s with SIMBAD")
      from astroquery.simbad import Simbad
      object = Simbad.query_object(self.name)
      RA  = np.array(str(object["RA"].data.data[0]).split(" "),dtype=float)
      DEC = np.array(str(object["DEC"].data.data[0]).split(" "),dtype=float)
      self.RA  = 15.*(RA[0]+RA[1]/60.+RA[2]/3600.)
      self.DEC = (DEC[0]+DEC[1]/60.+DEC[2]/3600.)
      return(1)
    except:
      verbose("--> Failed to solve source name with simbad")

    verbose("Trying name, RA, DEC")
    try:
      self.name, self.RA, self.DEC = np.array(self.name.split(","))
      self.RA  = float(self.RA)
      self.DEC = float(self.DEC)
    except:
      verbose("--> Something went wrong while processing %s" %self.name)

    printc("-> Couldn't identify the source, check %s" %self.name)
    sys.exit(1)


  def calculate_times(self):
    import datetime
    current   = datetime.datetime.now()
    available = current-datetime.timedelta(seconds=TIMEDELAY)
    self.timemin = dt2met(available-datetime.timedelta(days=self.timewindow))
    self.timemax = dt2met(available)

  def write_config(self):
    verbose("-> Write config file for %s" %(self.name))
    import enrico
    from enrico.config import get_config
    # Create object
    config = ConfigObj(indent_type='\t')
    mes = Loggin.Message()
    config['out']    = self.output+'/%s_%s/'%(self.name,self.band)
    config['Submit'] = 'no'
    # target
    config['target'] = {}
    config['target']['name'] = self.name
    config['target']['ra']   = str("%.4f" %self.RA)
    config['target']['dec']  = str("%.4f" %self.DEC)
    config['target']['redshift'] = '0'
    config['target']['spectrum'] = 'PowerLaw'
    # space
    config['space'] = {}
    config['space']['xref'] = config['target']['ra']
    config['space']['yref'] = config['target']['dec']
    config['space']['rad']  = self.roi
    # files
    #basepath = "/".join(enrico.__path__[0].split("/")[0:-1])
    #datapath = basepath+"/Data/download/"
    config['file'] = {}
    #config['file']['spacecraft'] = str("%s/lat_spacecraft_merged.fits" %datapath)
    #config['file']['event']      = str("%s/event.list" %datapath)
    config['file']['spacecraft'] = self.SCfile
    config['file']['event']      = self.PHfile
    config['file']['tag']        = 'fast'
    # time
    config['time'] = {}
    self.calculate_times()
    config['time']['tmin'] = self.timemin
    config['time']['tmax'] = self.timemax
    # energy
    config['energy'] = {}
    if (self.band=='lo'):
      config['energy']['emin'] = self.energymin
      config['energy']['emax'] = self.energycut
    elif (self.band=='hi'):
      config['energy']['emin'] = self.energycut
      config['energy']['emax'] = self.energymax
    else:
      config['energy']['emin'] = self.energymin
      config['energy']['emax'] = self.energymax

    config['energy']['enumbins_per_decade'] = 5
    # event class
    config['event'] = {}
    config['event']['irfs'] = 'CALDB'
    config['event']['evclass'] = '64' #'128' #64=transients, 128=source
    config['event']['evtype'] = '3'
    # analysis
    config['analysis'] = {}
    config['analysis']['zmax'] = 100
    # Validate
    verbose(config)
    # Add the rest of the values
    config = get_config(config)
    # Tune the remaining variables
    config['AppLC']['index'] = self.spindex
    config['AppLC']['NLCbin'] = int(24*self.timewindow/self.binsize+0.5)
    config['AppLC']['rad']  = self.roi
    # Write config file
    self.configfile = str("%s/%s_%sE.conf" %(self.output,self.name,self.band))
    with open(self.configfile,'w') as f:
      config.write(f)

  def get_data(self):
    from astroquery.fermi import FermiLAT

  def remove_prev_dir(self):
    from enrico.constants import AppLCPath
    import shutil
    fname = str("%s/%s/" \
        %(self.output, self.name))
    shutil.rmtree(fname, ignore_errors=True)

  def run_analysis(self):
    import shutil
    from enrico import environ
    from enrico.config import get_config
    from enrico.appertureLC import AppLC
    from enrico.submit import call
    from enrico.constants import AppLCPath
    infile = self.configfile
    config = get_config(infile)

    self.remove_prev_dir()

    verbose("-> Running the analysis for %s" %(analysis.name))

    # We will always try to run it in parallel,
    # either with python multiproc or with external torque pbs.
    if config['Submit'] == 'no':
      global fqueue
      fqueue.put(infile)
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

    verbose("--> Job sent successfully")

  def arm_triggers(self):
    verbose("--> Preparing triggers for %s" %(analysis.name))
    args={'outdir':self.output, 'source':self.name}
    global rqueue
    rqueue.put(args)

def analyze_results(outdir,source,triggermode='local',fluxref=None,sigma=3):
  import time
  from numpy import sum, sqrt
  import astropy.io.fits as pyfits
  from enrico.constants import AppLCPath
  # Retrieve results
  applcfile = str("%s/%s/%s/%s_fast_applc.fits" \
      %(outdir, source, AppLCPath, source))

  trials=NTRIALS;
  while(True):
    try:
      F=pyfits.open(applcfile)
    except:
      time.sleep(30)
      continue
    else:
      trials=NTRIALS

    try:
      applc = F[1].data
    except:
      verbose("--> Cannot open fits table, closing it")
      F.close()
      time.sleep(30)
      trials-=1
      if (trials==0):
        verbose("--> FatalError opening fits table in %s." %source); break
      continue
    else:
      trials=NTRIALS

    try:
      total_exp_prev = sum(applc['EXPOSURE'][:-1])
      total_cts_prev = sum(applc['COUNTS'][:-1])
      total_err_prev = sqrt(sum(applc['ERROR'][:-1]**2))
      last_exp = applc['EXPOSURE'][-1]
      last_cts = applc['COUNTS'][-1]
      last_err = applc['ERROR'][-1]
    except:
      verbose("--> Error reading contents.")
      time.sleep(30)
      trials-=1
      if (trials==0):
        verbose("--> FatalError reading contents." %source); break
      continue
    else:
      break

  printc("-> ##### %s #####" %source)
  verbose("--> Contents of file %s" %(applcfile))
  verbose(applc)

  if (triggermode=='local'):
    expratio = (1.*last_exp/total_exp_prev)
    proj_cts = expratio*total_cts_prev
    proj_err = expratio*total_err_prev
    if ((last_cts-proj_cts)>sigma*proj_err):
      printc("--> Flux: (%e +/- %.2e) ph/cm2/s" \
          %(last_cts/last_exp, last_err/last_exp))
      printc("--> ALERT: %s might be flaring! Excess is %1.2f sigma above average"\
        %(source,(last_cts-proj_cts)/proj_err))
      return(True)
    else:
      printc("-> Flux: (%e +/- %.2e) ph/cm2/s [%1.2f sigma]"\
          %(last_cts/last_exp, last_err/last_exp,\
            (last_cts-proj_cts)/proj_err))
      return(False)


##### Main iterator

def full_analysis():
    verbose("-> Updating Fermi-LAT data")
    if (UPDATE_FERMI): update_fermilat_data()

    # Point to the first source in the list
    verbose("-> Entering ScienceTools analysis loop")
    while(True):
      if (analysis.select_next_source()==analysis.get_list_of_sources()):
        verbose('--> Waiting for the ScienceTools threads to finish')
        while (not fqueue.empty()): time.sleep(10)
        # Removing data files:
        break

      for band in ["lo","hi"]:
        analysis.band = band
        analysis.query_fermilat_data()
        analysis.get_source_coordinates(catalog)
        analysis.write_config()
        analysis.run_analysis()

    verbose("-> Entering StatisticAnalysis loop")
    while(True):
      if (analysis.select_next_source()==analysis.get_list_of_sources()):
        verbose('--> Waiting for the StatisticAnalysis threads to finish')
        while (not rqueue.empty()): time.sleep(10)
        for _file_ in analysis.PHlist: os.remove(_file_)
        os.remove(analysis.SCfile)
        break

      for band in ["lo","hi"]:
        analysis.band = band
        analysis.arm_triggers()


##### Main loop

iterations=3
if __name__ == '__main__':
  analysis = Analysis()
  # Jobs
  fqueue = Queue()
  rqueue = Queue()
  fpool = Pool(analysis.simthreads, worker, (fqueue,rqueue))
  verbose("-> Loading Fermi-LAT catalog")
  catalog = FermiCatalog(analysis.catalog)

  while(True):
    full_analysis()
    break
    time.sleep(3600)

  for p in xrange(analysis.simthreads):
    rqueue.put("stop")

  fpool.close();
  fpool.join()

  verbose("-> Closing queues")
  printc('-> Analysis completed')
  verbose('-> Exiting...')


