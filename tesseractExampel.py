import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import os

inputPath="E:/PROJECT ALL/kaggle/project/dataExtract/VID7/"

#im = Image.open(inputPath+"-8.jpg") # the second one
#im = Image.open( "bangla2.jpg") # the second one
im = Image.open( "NP3.jpg") # the second one
im = im.filter(ImageFilter.MedianFilter())
enhancer = ImageEnhance.Contrast(im)
im = enhancer.enhance(10)
#im = im.convert('1')
im.save("english_pp.jpg")

pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
text = pytesseract.image_to_string(Image.open("english_pp.jpg"),lang="ben")
#text = pytesseract.image_to_string(Image.open("english_pp.jpg"),lang="eng")
print(text)