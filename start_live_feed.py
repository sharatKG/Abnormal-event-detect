
import cv2
from tkinter import *
from model import load_model
import numpy as np 
#from scipy.misc import imresize
from skimage.transform import resize
from test import mean_squared_loss
from keras.models import load_model
import argparse
import winsound
duration = 1000  # milliseconds
freq = 440  # Hz

def live():

	window.destroy()

	parser=argparse.ArgumentParser()

	parser.add_argument('modelpath',type=str)

	args=parser.parse_args()

	modelpath=args.modelpath


	vc=cv2.VideoCapture(0,cv2.CAP_DSHOW)
	rval=True
	print('Loading model')
	model=load_model(modelpath)
	print('Model loaded')

	threshold=0.82
	while True:
		imagedump=[]
		for i in range(10):
			rval,frame=vc.read()
			frame=resize(frame,(477,477,3))
			cv2.imshow('frame', frame)

			#Convert the Image to Grayscale


			gray=0.2989*frame[:,:,0]+0.5870*frame[:,:,1]+0.1140*frame[:,:,2]
			gray=(gray-gray.mean())/gray.std()
			gray=np.clip(gray,0,1)
			imagedump.append(gray)

		imagedump=np.array(imagedump)

		imagedump.resize(227,227,10)
		imagedump=np.expand_dims(imagedump,axis=0)
		imagedump=np.expand_dims(imagedump,axis=4)


		print('Processing data')

		output=model.predict(imagedump)



		loss=mean_squared_loss(imagedump,output)
		print("loss is:",loss)


		if loss>threshold:
			print('Abnormal Event Detected')
			winsound.Beep(freq, duration)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	vc.release()
	cv2.destroyAllWindows()

window = Tk()
window.geometry("500x500")

li=Label(window,text="Abnormal Event Detection", font=("Helvetica", 20))
li.pack(pady=20)

button = Button(window, text="Start", command=live,height=2, width=10, font=("Helvetica", 14),bg="yellow")
button.pack(fill="none", expand=False)

window.mainloop()






