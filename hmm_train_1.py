##  PROGRAM TRANSFORMS THE GIVEN SPEECH SIGNALS TO MFCC FEATURES AND CREATES MODELS FOR THE SENTENCES     ##

from python_speech_features import mfcc
from spafe.features.lpc import lpc, lpcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import os,sys
import traceback
import numpy as np
import hmmlearn
from hmmlearn import hmm
import pickle
import os
import statistics 

path="Inputs"
#path="17761A0592"

class SIG_PROCESS:    
	def __init__(self,dirs,wsize,modelfolder):
		try:
			self.windowsize=wsize
			self.dirs=dirs
			self.modelfolder=modelfolder
			print(self.dirs)
			self.files=os.listdir(os.path.join(path,dirs))
			self.filelist=[]
	            # PREPARE THE LIST OF FILES
			for i in range(0,len(self.files)):
				if self.files[i].endswith('wav'):
					str=os.path.join(path,dirs,self.files[i])
					self.filelist.append(str)
		    #print(os.path.join(path,dirs,self.files[i]))    	
		except Exception as e:
			print(e)
			traceback.print_exc()
	
	def gen_features(self):
		try:
			all_mfcc=[]	
			
			self.final_mfcc=[]   # Consists of the mfcc features for all the utterances of the same word
			mfcc_len=[]
			self.lengths=[]
			print(self.filelist)
			# Read .Wav files and create features from that
			for i in range(0,len(self.filelist)):
				inputsignal=[]
				(srate,sig)=wav.read(self.filelist[i])
				print(self.filelist[i])
				#PREPARE THE INPUT SPEECH SIGNAL WITH THE SPECIFIED WINDOW SIZE
				# IF THE NUMBER OF SAMPLES IS LESS THAN THE GIVEN WINDOW SIZE, THEN FILL THE ARRAY WITH 0s				
				if len(sig)<self.windowsize:
					for k in range(0,len(sig)):
						inputsignal.append(sig[k])
					for k in range(len(sig),self.windowsize):
						inputsignal.append(0.0)
				# OTHERWISE USE ONLY WINDOWSIZE NUMBER OF SAMPLES
				if len(sig)>self.windowsize:
					inputsignal=sig[0:self.windowsize]
				#print(type(sig))
				signal=[float(item) for item in inputsignal]
				signal=np.asarray(signal,dtype=float)				
				#print(type(signal))
				#print(len(signal))
				mfcc_feat=mfcc(signal,srate,winlen=0.030, winstep=0.025,winfunc=np.hamming)				
				#mfcc_feat=lpc(sig=signal, fs=srate, num_ceps=13)
				#mfcc_feat=lpcc(sig=signal, fs=srate, num_ceps=13, lifter=1,normalize=True)
				#print(len(mfcc_feat))
				#mfcc_len.append(len(mfcc_feat))				
				all_mfcc.append(mfcc_feat)
			#print('length:',len(all_mfcc))		
			
			# CONVERT THE all_mfcc columnwise
			for i in range(0,len(all_mfcc)):
				min_mfcc=all_mfcc[i][0:]
				for k in range(0,13):	
					self.final_mfcc.append(np.array([row[k] for row in min_mfcc]))
					#print(len(self.final_mfcc[i]))
			
			#print('length:',len(self.final_mfcc))
			#print(type(self.final_mfcc))	
		
		
			
		except Exception as e:
			print(e)
			traceback.print_exc()
			
	def train_model(self):
		try:
			# TRAIN THE HMM USING THE DATA PROVIDED IN self.final_mfcc
			print('..................WRITING MODELS.................. ')
			model = hmm.GaussianHMM(n_components=5, covariance_type="full",tol=.7,n_iter=150)
			model.startprob_ = np.array([1, 0, 0, 0, 0])
			model.transmat_ = np.array([[0.8, 0.2, 0, 0, 0],
										[0, 0.8, 0.2, 0, 0],
										[0, 0, 0.8, 0.2, 0],
										[0, 0, 0, 0.8, 0.2],
										[0, 0, 0, 0, 1]])
			model.means_ = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
			model.covars_ = np.tile(np.identity(2), (3, 1, 1))
			length=len(self.final_mfcc)
			
			m1 = hmm.GaussianHMM(n_components=5).fit(self.final_mfcc)
			
			modelpath=self.modelfolder+"/{}_hmm_model.pkl"			
			with open(modelpath.format(self.dirs), "wb") as file: pickle.dump(m1, file)
			
		except Exception as e:
			print(e)
			traceback.print_exc()
	
 



#path="17761A0592"
path="Person1_sasi"
#path="Person3_phaneendra"
#path="Person3_phaneendra"
modelfolder="Person3_Models"

SpeechPath=["SPEAKERDEPENDENTANALYSIS/DATA/person1","SPEAKERDEPENDENTANALYSIS/DATA/person2","SPEAKERDEPENDENTANALYSIS/DATA/person3","SPEAKERDEPENDENTANALYSIS/DATA/person4","SPEAKERDEPENDENTANALYSIS/DATA/Person1_sasi","SPEAKERDEPENDENTANALYSIS/DATA/Person2_jagadeeswari","SPEAKERDEPENDENTANALYSIS/DATA/Person3_phaneendra","SPEAKERDEPENDENTANALYSIS/DATA/Person4_Vamsi","SPEAKERDEPENDENTANALYSIS/DATA/17761A0592"]


ModelPath=["SPEAKERDEPENDENTANALYSIS/MODELS/person1_Models", "SPEAKERDEPENDENTANALYSIS/MODELS/person2_Models","SPEAKERDEPENDENTANALYSIS/MODELS/person3_Models","SPEAKERDEPENDENTANALYSIS/MODELS/person4_Models","SPEAKERDEPENDENTANALYSIS/MODELS/Person1_sasi_Models","SPEAKERDEPENDENTANALYSIS/MODELS/Person2_jagadeeswari_Models","SPEAKERDEPENDENTANALYSIS/MODELS/Person3_phaneendra_Models","SPEAKERDEPENDENTANALYSIS/MODELS/Person4_Vamsi_Models","SPEAKERDEPENDENTANALYSIS/MODELS/Samanvi_Models"]


def find_windowsize(p):
	try:
		path=p
		print("OPTIMAL WINDOW SIZE")
		dirlist=os.listdir(path)
		ssize=[]
		#print(dirlist)
		for i in dirlist:
		#	print(i)
			dir_in=os.listdir(os.path.join(path,i))
		#	print(dir_in)
			for j in dir_in:
				if j.endswith('wav'):           
					fullpath=os.path.join(path,i,j)
		#			print(fullpath)
					(srate,sig)=wav.read(fullpath)
		#			print(len(sig),'<==>',srate)
					ssize.append(len(sig))
		#print(min(ssize))
		#print(max(ssize))
		wsize=(sum(ssize)/len(ssize))
		#wsize=max(ssize)
		wsize=int(wsize/5000)
		wsize=wsize*5000
		return wsize
		
	except Exception as e:
		print(e)
# Read the files from the input folders
#spath="SPEAKERINDEPENDENT/DATA/17761A0592"
#mpath="SPEAKERINDEPENDENT/MODELS"
spath="SAMANVI2/85TELUGU"
mpath="SAMANVI2/85TELUGUMODELS"
#spath="SAMANVI2/92"
#mpath="SAMANVI2/92Models"
#spath="Meghana/18761A0546"
#mpath="Meghana/18761A0546Models"
#Train Sasi data
#spath=SpeechPath[4]
#mpath=ModelPath[4]
#spath="URDUEXPERIMENT/person1"
#mpath="URDUEXPERIMENT/PERSON1MODELS"
WINDOWSIZE=[]
# Generate Models for each input folder
for j in range(8,9):
	modelpath=mpath
	# FUNCTION TO FIND THE WINDOWS SIZE
	path=spath
	wsize=find_windowsize(path)
	WINDOWSIZE.append(wsize)
	print(wsize)
	dirlist=os.listdir(path)
	# CREATE AN OBJECT FOR THE SIG_PROCESS
	for i in range(0, len(dirlist)):
		o1=SIG_PROCESS(dirlist[i],wsize,modelpath)
		o1.gen_features()
		o1.train_model()

print(WINDOWSIZE)
