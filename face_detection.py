import numpy as np
import cv2 as cv
import streamlit as st
import urllib

st.header("Face Detection App")
st.write("This app detects faces in photographs and puts bounding boxes around them")
st.write("Increase the value of **minNeighbors** in the sidebar to increase the threshold for a contendor to be classified as a face.")
st.sidebar.header("minNeighbors")
st.sidebar.write('''This parameter specifies how many neighbours each candidate rectangle should have to retain it. This parameter affects the quality of the detected faces: higher value results in less detections but with higher quality.''')
minNeighbors = st.sidebar.slider("minNeighbors", 1,20, 10)

# for uploaded image
# the output type is <class 'streamlit.uploaded_file_manager.UploadedFile'>
uploaded_img = st.file_uploader("Upload an image.", type=['png','jpeg','jpg'])

# for image through url
def url_to_image(url):
	resp = urllib.request.urlopen(url)      # get the image from the url
	image = np.asarray(bytearray(resp.read()), dtype="uint8")   # conver it to numpy array
	image = cv.imdecode(image, cv.IMREAD_COLOR)     # convert to opencv format
	return image

url_img = st.text_input("Or enter image adress from the web. Try: https://i.redd.it/0oh62jyzl5g61.jpg", "enter image address")

if uploaded_img is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    img = cv.imdecode(file_bytes, 1)

elif url_img[:4] == 'http':
    img = url_to_image(url_img)

else:
    # default
    img = url_to_image('https://cdn.vox-cdn.com/thumbor/ohzylsEPVKYzivA3t1VwIug7R70=/0x0:4285x2856/1200x800/filters:focal(1595x903:2279x1587)/cdn.vox-cdn.com/uploads/chorus_image/image/65751058/1180478852.jpg.0.jpg')
    # 'https://i.dawn.com/large/2019/06/5d0261b7b14c7.jpg'

img_copy = img.copy()   # used for dsplaying cropped faces later


# detecting feces
@st.cache
def detetct_faces(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier("haarcascade_frontalface.xml")
    face_boxes = face_detector.detectMultiScale(image=gray, scaleFactor=1.1, minNeighbors=minNeighbors)
    return face_boxes

face_boxes = detetct_faces(img)

# drawing bounding boxes
for face_box in face_boxes:
    x,y, width,height = face_box
    cv.rectangle(img=img, pt1=(x, y), pt2=(x + width, y + height), color = (0,255,0), thickness=2)

st.image(img, channels='BGR')

# crop faces
faces = []

for face_box in face_boxes:
    x, y, w, h = face_box
    face = img_copy[y:y+h, x:x+w]
    faces.append(face)

st.subheader('Faces Detected')
columns = st.beta_columns(5)

for i,face in enumerate(faces):
    with columns[i % 5]:
        st.image(face, channels='BGR')