##  PROGRAM TRANSFORMS THE GIVEN SPEECH SIGNALS TO MFCC FEATURES AND CREATES MODELS FOR THE SENTENCES     ##
from python_speech_features import mfcc
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

ModelPath=["SPEAKERDEPENDENTANALYSIS/MODELS/person1_Models", "SPEAKERDEPENDENTANALYSIS/MODELS/person2_Models","SPEAKERDEPENDENTANALYSIS/MODELS/person3_Models","SPEAKERDEPENDENTANALYSIS/MODELS/person4_Models","SPEAKERDEPENDENTANALYSIS/MODELS/Person1_sasi_Models","SPEAKERDEPENDENTANALYSIS/MODELS/Person2_jagadeeswari_Models","SPEAKERDEPENDENTANALYSIS/MODELS/Person3_phaneendra_Models","SPEAKERDEPENDENTANALYSIS/MODELS/Person4_Vamsi_Models","SPEAKERDEPENDENTANALYSIS/MODELS/Samanvi_Models"]

TestPath=["SPEAKERDEPENDENTANALYSIS/TESTDATA/testdataperson1","SPEAKERDEPENDENTANALYSIS/TESTDATA/testdataperson2","SPEAKERDEPENDENTANALYSIS/TESTDATA/testdataperson3","SPEAKERDEPENDENTANALYSIS/TESTDATA/testdataperson4","SPEAKERDEPENDENTANALYSIS/TESTDATA/testdatasasi","SPEAKERDEPENDENTANALYSIS/TESTDATA/testdatajagadeeswari","SPEAKERDEPENDENTANALYSIS/TESTDATA/TestdataPhaneendra","SPEAKERDEPENDENTANALYSIS/TESTDATA/testdatavamsi","SPEAKERDEPENDENTANALYSIS/TESTDATA/testdata92"]

WINDOWSIZE=[15000, 15000, 20000, 20000, 20000, 30000, 20000, 30000,50000]

#testpath='Person3Testdata'
#modelfolder='Person3Models'
mpath="SPEAKERINDEPENDENT/MODELS"

#testpath=TestPath[8]


testpath="SAMANVI2/85TESTDATA"
modelfolder="SAMANVI2/85TELUGUMODELS"

#testpath="SAMANVI2/92TESTDATA"
#mpath="SAMANVI2/92Models"
#modelfolder=mpath
windowsize=50000

class HMM_TEST:
	def __init__(self,path):
		try:
			self.files=os.listdir(os.path.join(path))
			self.filelist=[]
	            # PREPARE THE LIST OF FILES
			for i in range(0,len(self.files)):
				str=os.path.join(path,self.files[i])
				self.filelist.append(str)
			    #print(os.path.join(path,dirs,self.files[i]))
			print('File List: ',self.filelist)
		    	
		except Exception as e:
			print(e)
			traceback.print_exc()
    
	def loadModels(self,modelfolder):
		try:
			print('Sentence Models Read')
			self.models=os.listdir(os.path.join(modelfolder))
			self.modellist=[]
			self.senmodel=[]   # Stores the pickle files
			self.sentences=[]
			print(self.models)
			for i in range(0,len(self.models)):
				str=os.path.join(modelfolder,self.models[i])
				self.modellist.append(str)
				
			print('Models List:',self.modellist)	
			# Read the sentence strings
			for i in range(0,len(self.models)):
				temp=self.models[i]
				pos=temp.find('_')
				temp=temp[0:pos]
				self.sentences.append(temp)
			# Read the models into senmodel variable
			for j in range(0,len(self.modellist)):
				self.senmodel.append(pickle.load(open(self.modellist[j], "rb")))
			#print(self.senmodel)
			print('SENTENCES: ',self.sentences)
		except Exception as e:
			print(e)
				
			traceback.print_exc()
			
	def TestModels(self):
		try:
			print(self.filelist)
			# Read each file from the test files folder and find the accuracy
			total_correct=0
			total_wrong=0
			Accuracy=0.0			
			for m in range(0,len(self.filelist)):
				(srate,sig)=wav.read(self.filelist[m])
				inputsignal=[]
				print('====Processing====',self.filelist[m])
				temp=self.filelist[m]
				pos=temp.find('_')
				pos1=temp.find('_',pos+2,len(temp))
				
				print(pos,' ',pos1)
				inputword=temp[pos+1:pos1]
				print('Input Word:',inputword)
				# Pad the input file with 0s
				if(len(sig)>windowsize):
					inputsignal=sig[0:windowsize]
				else:
					for k in range(0,len(sig)):
						inputsignal.append(sig[k])
					for k in range(len(sig),windowsize):
						inputsignal.append(0.0)
				signal=[float(item) for item in inputsignal]
				signal=np.asarray(signal,dtype=float)	
		#		mfcc_feat=mfcc(signal,srate)
				mfcc_feat=mfcc(signal,srate,winlen=0.030, winstep=0.025,winfunc=np.hamming)				
				final_mfcc=[]
				scores=[]
			
				for k in range(0,13):	
					final_mfcc.append(np.array([row[k] for row in mfcc_feat]))
				#print(len(final_mfcc))	
				for i in range(0,len(self.senmodel)):	
					scores.append(self.senmodel[i].score_samples(final_mfcc))

				map_index=scores.index(max(scores))	
				# FIND OUT THE MAPPED SENTENCE
				temp_sentence=self.modellist[map_index]
				pos1=temp_sentence.rfind('/')
				pos2=temp_sentence.find('_',pos1+1,len(temp_sentence))
				mapped_sentence=temp_sentence[pos1+1:pos2]
				print(inputword,'<===>',m,'<==>',map_index,'<===>',mapped_sentence)	
				if(inputword==mapped_sentence):
					total_correct=total_correct+1
				else:
					total_wrong=total_wrong+1
			overall_accuracy = total_correct*100/(total_correct+total_wrong)
			print('Overall Accuracy:',overall_accuracy)							
		except Exception as e:
			print(e)
			traceback.print_exc()
print(testpath)
print(modelfolder)  
o1=HMM_TEST(testpath)
#Step1: Load all the models
o1.loadModels(modelfolder)
#Step2: Load all the test files
o1.TestModels()	
