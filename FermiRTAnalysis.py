
import os
import os.path
import sys
from pprint import pprint
import argparse
import numpy as np
import scipy

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits

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

class fg:
		black='\033[30m'
		red='\033[31m'
		green='\033[32m'
		orange='\033[33m'
		blue='\033[34m'
		purple='\033[35m'
		cyan='\033[36m'
		lightgrey='\033[37m'
		darkgrey='\033[90m'
		lightred='\033[91m'
		lightgreen='\033[92m'
		yellow='\033[93m'
		lightblue='\033[94m'
		pink='\033[95m'
		lightcyan='\033[96m'
		reset='\033[0m'
		bold='\033[01m'
		disable='\033[02m'
		underline='\033[04m'
		reverse='\033[07m'
		strikethrough='\033[09m'
		invisible='\033[08m'


def pnormal(msg):
		print(fg.white + msg + fg.reset)

def psuccess(msg):
		print(fg.lightblue + msg + fg.reset)

def pwarn(msg):
		print(fg.yellow + msg + fg.reset)

def perr(msg):
		print(fg.red + msg + fg.reset)

def pverbose(msg):
		from datetime import datetime
		dt = datetime.strftime(datetime.now(), "[%Y%m%d_%H%M]")
		printc(fg.lightblue + dt + fg.reset + msg)



class FermiAppLC(object):
    def __init__(self,lcfile):
        # Load file
        self.load_file(lcfile)
        self.get_prop_from_filename(lcfile)
        self.header_analysis()
        self.load_data(obfilter)
        self.calculate_average()
        self.close_file()

    def load_file(self,lcfile):
        import astropy.io.fits as pyfits
        self.openedfile = pyfits.open(lcfile)

    def close_file(self):
        self.openedfile.close()

    def header_analysis(self):
        header = self.openedfile[1].header
        self.srcname = self.openedfile[0].header['FILENAME'].split("_")[1]
        self.timedel = self.openedfile[1].data['TIMEDEL'][0]
        self.energy_range = [header[h.replace("DSTYP","DSVAL")] \
                for h in header.keys() if "DSTYP" in h \
                and "ENERGY" in header[h]][0].split(":")
        var = self.openedfile[1].header['DSTYP1']
        for V in ['BIT_MASK','(',')']: var = var.replace(V,'')
        self.fclass, self.ftype,self.fversion = var.split(',')

    def load_data(self,obfilter=None):
        self.data = self.openedfile[1].data
        self.rawcounts   = self.data['COUNTS']
        self.rawerrors   = self.data['ERROR']
        self.rawexposure = self.data['EXPOSURE']

    def calculate_average(self,obfilter=None):
        self.times    = self.data['TIME']
        self.timeerr  = self.data['TIMEDEL']/2.
        self.meanflux = np.sum(self.counts)/np.sum(self.exposure)
        self.errflux  = np.sqrt(np.sum(self.counts))/np.sum(self.exposure)

    def calc_pointsignificance(self,max_bins=1):
        self.significance_nn = dict()
        for k in xrange(max_bins):
            print(" [] Computing SumSignificance for [%d]" %k)
            self.significance_nn[k] = self.cumulative_significance(k)
    
    def update_configfile(configfile):
        import yaml
        with open(configfile,"r") as cfg:
            data = yaml.load(cfg)
            data["Analysis"][self.triggernumber]['flux'] =\
                str("%s,%s" %(self.meanflux,self.errflux))
        
        with open(configfile,"w") as cfg:
            cfg.write(yaml.dump(data, default_flow_style=False))

class MultipleFileAnalysis(object):
    def __init__(self,srcfiles):
        self.srcfiles = srcfiles
        self.time_array = np.arange(200000000,600000000,2000)
        self.trig_array = np.zeros(np.size(self.time_array),dtype=int)
        self.analysis = dict()

    def single_filter_file_selector(\
        self,src,
        energy=None,
        roi=None,
        srcclass=None,
        binsize=None,
        index=None):

        mask = np.ones(np.size(self.srcfiles[src]),dtype=bool)
        for k,s in enumerate(self.srcfiles[src]):
            if (energy!=None):
                mask[k] *= ('_%s_MeV'%str(energy) in s)
            if (roi!=None):
                mask[k] *= ('ROI_%s_deg'%str(roi) in s)
            if (srcclass!=None):
                mask[k] *= ('evclass_%s_'%str(srcclass) in s)
            if (binsize!=None):
                mask[k] *= ('_%s_s'%str(binsize) in s)
            if (index!=None):
                mask[k] *= ('index_%s_'%str(index) in s)

        self.selected_files = np.array(self.srcfiles[src])[mask]
        print(self.selected_files)
        return(mask)

    def custom_filter_file_selector(self,src,rois=None,\
            srcclasses=None,energyranges=None,binsizes=None,append=True):
        
        self.src = src
        self.roi = rois[0]
        self.srcclass = srcclasses[0]

        try: self.mask
        except: append=False

        if (not append):
            self.mask = np.zeros(np.size(self.srcfiles[src]),dtype=bool)
        
        if (rois==None): rois = [1.0]
        if (srcclasses==None): srcclasses = [128]
        if (binsizes==None): binsizes = [10800,21600,43200]
        if (energyranges==None): 
            energyranges = ["300_800000","100_800000","1000_800000"]

        for binsize in binsizes:
            for srcclass in srcclasses:
                for roi in rois:
                    for erange in energyranges:
                        self.mask += self.single_filter_file_selector(src=src,\
                        energy=erange,roi=roi,srcclass=srcclass,
                        binsize=binsize,index="-2.0")


        self.mask[self.mask>1]=True
        self.selected_files = np.array(self.srcfiles[src])[self.mask]

        print(len(self.selected_files))

    def get_triggered_data(self,sigmas=3.,minsize=0, deltatime=5, min_triggers=1):
        # For each file, do the analysis and get triggers
        to_remove = []

        for F in self.selected_files:
            try:
                self.analysis[F] = FermiAppLC(F)
                self.analysis[F].get_trigger(sigmas)
                self.analysis[F].apply_trigger()
                nflares = self.analysis[F].flare_episodes(\
                    minsize=minsize,deltatime=deltatime)
                print("Flares in %s: %d" %(F,nflares))
                self.trig_array = self.analysis[F].flag_flares_in_common_array(\
                    self.time_array,\
                    self.trig_array)
            except:
                to_remove.append(F)

        for ToRemove in to_remove:
            try:
                del self.analysis[ToRemove]
            except:
                print('Cannot remove %s' %ToRemove)

        # Flare episodes
        # Select region around each triggered time, then isolate high state blobs
        from scipy import ndimage
        from scipy.ndimage import morphology as morph # binary_dilation
        from scipy.ndimage import label
        self.flaringstate = np.array(self.trig_array>=min_triggers)
        self.highstate=np.array(self.flaringstate)

        print("There are %d bins in high state (raw)" %np.sum(self.highstate))
        if (minsize>0):
            self.highstate = morph.binary_erosion(\
                self.highstate,iterations=int(minsize*86400/20000.))
        print("There are %d bins in high state after erosion"\
            %np.sum(self.highstate))
        if (deltatime>0):
            self.highstate = morph.binary_dilation(self.highstate,\
                iterations=int((minsize+deltatime)*86400/20000.))
        print("There are %d bins in high state after dilation"\
            %np.sum(self.highstate))
        self.flares, self.number_of_flares = label(self.highstate)
        return(self.number_of_flares)

    def met_to_mjd(self,met):
        from astropy.time import Time
        ref_tt = Time('2001-01-01 00:00:00', scale='utc')
        mjd = Time(met+ref_tt.unix, scale='utc', format='unix').mjd
        return(mjd)

    def plot_flares(self,extend_days=5):
        from scipy.ndimage import morphology as morph

        print('Selecting flare')
        try: self.flarenum = (self.flarenum)%(self.number_of_flares)+1
        except: self.flarenum = 1

        # For each analysis
        nf = 0

        try:
            plt.close()
            del(self.flareplot)
            del(self.flarefig)
        except:
            pass

        print('Creating figure')
        self.flarefig = plt.figure(figsize=(8,len(self.analysis.keys())*2+4),dpi=150)
        self.flareplot = []

        print('Looping over files')
        for filename in self.analysis:
            print(filename)
            A = self.analysis[filename]
            # Get the times and flag the corresponding
            contour = np.zeros(np.size(A.times),dtype=bool)
            idmin = (np.abs(A.times-min(\
                self.time_array[self.flares==self.flarenum]))).argmin()
            idmax = (np.abs(A.times-max(\
                self.time_array[self.flares==self.flarenum]))).argmin()
            contour[np.arange(idmin,idmax+1)]=True

            center  = contour*A.flaringstate

            region=np.bitwise_xor(morph.binary_dilation(contour,\
                iterations=int(extend_days*86400./(2*A.timeerr[1]))),contour)
            contour = np.bitwise_xor(contour,center)

            flare_times   = self.met_to_mjd(A.times[center])
            flare_timeerr = A.timeerr[center]/86400.
            flare_flux    = A.flux[center]
            flare_fluxerr = A.fluxerr[center]

            high_times   = self.met_to_mjd(A.times[contour])
            high_timeerr = A.timeerr[contour]/86400.
            high_flux    = A.flux[contour]
            high_fluxerr = A.fluxerr[contour]

            reg_times   = self.met_to_mjd(A.times[region])
            reg_timeerr = A.timeerr[region]/86400.
            reg_flux    = A.flux[region]
            reg_fluxerr = A.fluxerr[region]

            print('Creating subplot %d' %(len(self.analysis.keys())))

            try: self.flareplot[0]
            except:
                flareplot = self.flarefig.add_subplot(\
                    len(self.analysis.keys()),1,nf+1)
            else:
                flareplot = self.flarefig.add_subplot(\
                    len(self.analysis.keys()),1,nf+1,sharex=self.flareplot[0])

            flareplot.text(.5,.9,filename.split("/")[-1],\
                horizontalalignment='center', fontsize='xx-small',
                transform=flareplot.transAxes)

            flareplot.axvspan(\
                self.met_to_mjd(min(self.time_array[self.flares==self.flarenum])),
                self.met_to_mjd(max(self.time_array[self.flares==self.flarenum])),
                ymin=0, ymax=1, color='blue', alpha=0.25)

            flareplot.errorbar(\
                x=high_times,
                y=high_flux,
                xerr=high_timeerr,
                yerr=high_fluxerr,
                marker='o',ms=4,ls='',
                color='red', ecolor='black')

            flareplot.errorbar(\
                x=flare_times,
                y=flare_flux,
                xerr=flare_timeerr,
                yerr=flare_fluxerr,
                marker='o',ms=5,ls='',
                color='purple', ecolor='black')

            flareplot.errorbar(\
                x=reg_times,
                y=reg_flux,
                xerr=reg_timeerr,
                yerr=reg_fluxerr,
                marker='D',ms=3,ls='',
                color='orange', ecolor='gray')

            flareplot.ticklabel_format(axis='x', useOffset=False)
            flareplot.ticklabel_format(axis='y', style = 'sci', \
                useOffset=False, scilimits=(-2,2))
            flareplot.set_ylabel('Flux $(ph\ cm^{-2} s^{-1})$')

            if (len(self.analysis.keys())==nf+1):
                flareplot.set_xlabel('MJD')

            self.flareplot.append(flareplot)
            nf += 1

        #flareplot.legend(numpoints=1,framealpha=0.5,fontsize='xx-small')

    def saveflare(self,path=None):
        if (path==None):
            if not os.path.exists("%s/FlarePlots/%s/" %(basedir,self.src)):
                os.makedirs("%s/FlarePlots/%s/" %(basedir,self.src))
            path = str("%s/FlarePlots/%s/%s_%s_%s_flare_%d-%d.png" \
                %(basedir,self.src,self.src,\
                self.roi,self.srcclass, \
                self.flarenum,self.number_of_flares))

        while (" " in path):
            path = path.replace(" ","")

        print(path)
        plt.tight_layout()
        self.flarefig.savefig(path,dpi=150)



def worker(fqueue,rqueue):
	import time
	global analock
	verbose("%s busy" %(str(os.getpid())))
	while(True):
		try:
			if (not fqueue.empty()):
				verbose("fqueue is not empty")
				item = fqueue.get(True)
				name=item.split("/")[-1].split(".conf")[0]
				analock[name]='Busy[LC]'
				AppLC(item)
				analock[name]='Done[LC]'
			elif (not rqueue.empty()):
				args = rqueue.get(True)
				verbose("rqueue is not empty: %s" %(str(args)))
				if (args=="stop"):
					verbose("Finished signal received, breaking the loop")
					break
				analyze_results(**args)

			time.sleep(1)
		except KeyboardInterrupt:
			break

# flexible pretty print
def printc(text,kind="normal"):
	global tofile
	global clearfile
	if(tofile!=None):
		if (clearfile):
			open(tofile,"w").close()
		outf = open(tofile,"a+")
		pprint(text,outf)
		outf.close()

		if (kind=="info"):      pinfo(text)
		elif (kind=="success"): psuccess(text)
		elif (kind=="warning"): pwarning(text)
		elif (kind=="error"):   perror(text)
		else:		            pnormal(text)
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
		self.Cat	 = pyfits.open(path)
		self.Table = self.Cat[1].data
		self.NameList = []
		column_names	= [0,64,65,66,67,68,69,70,74,75]

		for k in self.Table:
			namelist = np.append(\
				[k[j] for j in column_names],
				[l.replace(" ","") for l in [k[j] for j in column_names]])
			self.NameList.append(namelist)


	def find_source(self,name):
		column_types		= 73
		column_raj2000	= 1
		column_decj2000 = 2
		column_glon		 = 3
		column_glat		 = 4
		column_pwlindex = 30

		src = dict()
		for k in xrange(len(self.NameList)):
			for j in xrange(len(self.NameList[k])):
				if name in str(self.NameList[k][j]):
					self.name			 = self.NameList[k][j]
					src["arg"]			= k
					src["raj2000"]	= self.Table[k][column_raj2000]
					src["decj2000"] = self.Table[k][column_decj2000]
					src["glon"]		 = self.Table[k][column_glon]
					src["glat"]		 = self.Table[k][column_glat]
					src["pwlindex"] = self.Table[k][column_pwlindex]
					return(src)

		printc("Source name %s not found!." %name)


class Analysis(object):
	def __init__(self):
		self.parse_arguments()
		self.current_source = -1

	def parse_arguments(self):
		parser = argparse.ArgumentParser(
			description='Create Fermi Apperture LCs for the given objects.')
		parser.add_argument('config', metavar='configfile',
			type=str,
			help='Config file')
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
		parser.add_argument('-sm', '--simthreads', dest='simthreads',
			type=int, default=1,
			help='Simultaneous threads / jobs to process')
		parser.add_argument('-v', '--verbose', dest='verbosemode',
			action='store_true', 
			help='Activate verbose mode')

		args = parser.parse_args()
		vars(self).update(args.__dict__)

		# verbose?
		global verbosemode
		verbosemode=args.verbosemode

		# Update log file
		global tofile
		tofile = os.path.abspath(args.log)
		verbose("-> Parsing arguments")
		#printc(args)
		verbose("-> Converting relative paths to absolute")
		self.sourcelistfn = os.path.abspath(self.sourcelistfn)
		self.output = os.path.abspath(self.output)

    def parse_config(self,configfile):
        import yaml
        self.configfile = configfile
        with open(configfile,"r") as cfgfile:
            self.config = yaml.load(cfgfile)
	    

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
		ra = self.RA
		dec = self.DEC
		Erange = str("%d, %d" %(self.energymin,self.energymax))
		LATdatatype = "Extended"
		self.calculate_times()
		timesys = 'MET'
		obsdates = str("%s, %s" %(str(self.timemin),"END")) #str(self.timemax)))
		verbose("Query:")
		verbose(str("%s, %s, %s, %s, %s, %s, %s, %s" \
			%(object, ra, dec, Erange, self.roi, LATdatatype, obsdates, timesys)))

		result = FermiLAT.query_object(\
			name_or_coords=str("%.4f, %.4f" %(ra,dec)),\
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
			self.RA	= source["raj2000"]
			self.DEC = source["decj2000"]
			return(1)
		except:
			verbose("-> Failed to locate source in the Fermi catalog")

		try:
			verbose("-> Solve source name %s with SIMBAD")
			from astroquery.simbad import Simbad
			object = Simbad.query_object(self.name)
			RA	= np.array(str(object["RA"].data.data[0]).split(" "),dtype=float)
			DEC = np.array(str(object["DEC"].data.data[0]).split(" "),dtype=float)
			self.RA	= 15.*(RA[0]+RA[1]/60.+RA[2]/3600.)
			self.DEC = (DEC[0]+DEC[1]/60.+DEC[2]/3600.)
			return(1)
		except:
			verbose("--> Failed to solve source name with simbad")

		verbose("Trying name, RA, DEC")
		try:
			self.name, self.RA, self.DEC = np.array(self.name.split(","))
			self.RA	= float(self.RA)
			self.DEC = float(self.DEC)
		except:
			verbose("--> Something went wrong while processing %s" %self.name)

		printc("-> Couldn't identify the source, check %s" %self.name)
		sys.exit(1)


	def calculate_times(self):
		import datetime
		current	 = datetime.datetime.now()
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
		config['target']['name']     = self.name
		config['target']['ra']       = str("%.4f" %self.RA)
		config['target']['dec']      = str("%.4f" %self.DEC)
		config['target']['redshift'] = '0'
		config['target']['spectrum'] = 'PowerLaw'
		# space
		config['space'] = {}
		config['space']['xref'] = config['target']['ra']
		config['space']['yref'] = config['target']['dec']
		config['space']['rad']	= self.roi
		# files
		#basepath = "/".join(enrico.__path__[0].split("/")[0:-1])
		#datapath = basepath+"/Data/download/"
		config['file'] = {}
		#config['file']['spacecraft'] =\
        #    str("%s/lat_spacecraft_merged.fits" %datapath)
		#config['file']['event']			= str("%s/event.list" %datapath)
		config['file']['spacecraft'] = self.SCfile
		config['file']['event']	     = self.PHfile
		config['file']['tag']		 = 'fast'
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
		config['AppLC']['rad']	= self.roi
		# Write config file
		self.configfile = str("%s/%s_%s.conf" %(self.output,self.name,self.band))
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
			LCoutfolder =	config['out']+"/"+AppLCPath
			os.system("mkdir -p "+LCoutfolder)
			prefix = LCoutfolder +"/"+ config['target']['name'] +\
                "_AppertureLightCurve"
			scriptname = prefix+"_Script.sh"
			JobLog = prefix + "_Job.log"
			JobName = "LC_%s" %self.name
			call(cmd, enricodir, fermidir, scriptname, JobLog, JobName)

		verbose("--> Job sent successfully")

	def arm_triggers(self):
		verbose("--> Preparing triggers for %s" %(analysis.name))
		args={'outdir':self.output, 'source':self.name, 'band':self.band}
		global rqueue
		rqueue.put(args)

class StatisticalAnalysis(object):
    def __init__(self,config,source):
        self.source = source
        self.config = config
        self.outdir = config["outputdir"]
        global analock
        applcfile = str("%s/%s_%s/%s/%s_fast_applc.fits"\
            %(outdir, source, band, AppLCPath, source))




def analyze_results(outdir,source,band,triggermode='local',fluxref=None,sigma=3):
	import time
	from numpy import sum, sqrt
	import astropy.io.fits as pyfits
	from enrico.constants import AppLCPath
	global analock
	# Retrieve results 
	applcfile = str("%s/%s_%s/%s/%s_fast_applc.fits" \
        %(outdir, source, band, AppLCPath, source))
	if (analock[source+"_"+band] not in ["Busy[LC]" or "Done[LC]"]):
		verbose("--> Warning, inconsistent state for source %s: %s"\
            %(source,analock[source+"_"+band]))
	while (analock[source+"_"+band]=="Busy[LC]"): time.sleep(10)
	analock[source+"_"+band]="Busy[ST]"
	with pyfits.open(applcfile) as F:
		applc = F[1].data
	total_exp_prev = sum(applc['EXPOSURE'][:-1])
	total_cts_prev = sum(applc['COUNTS'][:-1])
	total_err_prev = sqrt(sum(applc['ERROR'][:-1]**2))
	last_exp = applc['EXPOSURE'][-1]
	last_cts = applc['COUNTS'][-1]
	last_err = applc['ERROR'][-1]
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
			printc("--> ALERT: %s might be flaring! %1.2f sigma above average"\
				%(source,(last_cts-proj_cts)/proj_err))
			analock[source+"_"+band]="Done[ST]"
			return(True)
		else:
			printc("-> Flux: (%e +/- %.2e) ph/cm2/s [%1.2f sigma]"\
					%(last_cts/last_exp, last_err/last_exp,\
						(last_cts-proj_cts)/proj_err))
			analock[source+"_"+band]="Done[ST]"
			return(False)


##### Main iterator

def full_analysis():
    verbose("-> Updating Fermi-LAT data")
    if (UPDATE_FERMI): update_fermilat_data()
    analysis.blacklist = []
    # Point to the first source in the list
    verbose("-> Entering ScienceTools analysis loop")
    while(True):
        if (analysis.select_next_source()==analysis.get_list_of_sources()):
            verbose('--> Waiting for the ScienceTools threads to finish')
            while (not fqueue.empty()): time.sleep(10)
            # Removing data files:
            break

        for band in ["lo","hi"]:
            if (str(analysis.name+"_"+band) in analysis.blacklist): continue 
            analock[analysis.name+"_"+band] = ""
            analysis.band = band
            try:
                analysis.get_source_coordinates(catalog)
                analysis.query_fermilat_data()
                analysis.write_config()
                analysis.run_analysis()
            except:
                verbose(str("-> Error analyzing %s_%s, blacklisting it"\
                    %(analysis.name,band)))
                analysis.blacklist.append(analysis.name+"_"+band)

    verbose("-> Entering StatisticAnalysis loop")
    while(True):
        if (analysis.select_next_source()==analysis.get_list_of_sources()):
            verbose('--> Waiting for the StatisticAnalysis threads to finish')
            while (not rqueue.empty()): time.sleep(10)
            for _file_ in analysis.PHlist: 
                verbose("-> Removing %s" %_file_)
                os.remove(_file_)
            verbose("-> Removing %s" %analysis.SCfile)
            os.remove(analysis.SCfile)
            break

        for band in ["lo","hi"]:
            if (str(analysis.name+"_"+band) in analysis.blacklist): continue 
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
		print(analock)
		time.sleep(3600)

	for p in xrange(analysis.simthreads):
		rqueue.put("stop")

	fpool.close();
	fpool.join()

	verbose("-> Closing queues")
	printc('-> Analysis completed')
	verbose('-> Exiting...')


