import nltk.classify
from tkinter import *
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import  pandas as pd
import numpy as np
print("Designing UI")
root = Tk()
root.wm_title('Sentiment Analysis Application')

top_frame = Frame(root)
top_frame.pack()

bottom_frame = Frame(root)
bottom_frame.pack(side=BOTTOM)

l1 = Label(top_frame, text='Enter a review:')
l1.pack(side=LEFT)

w = Text(top_frame, height=3 )
w.pack(side=LEFT)

print("UI COMPLETE")
print()
def main_op():
    review_spirit = w.get('1.0',END)
    clf = joblib.load("svm.pkl")
    x = np.array(review_spirit.split(" "))
    print(x.dtype)
    demo2 = ('review is ' + clf.predict(x))
    l2 = Label(bottom_frame, text=demo2)
    l2.pack()

button = Button(bottom_frame, text='Analyse', command=main_op )
button.pack(side=BOTTOM)

root.mainloop()