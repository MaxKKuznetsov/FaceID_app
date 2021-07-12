# https://github.com/sozykin/dlpython_course/blob/master/computer_vision/foto_comparison/foto_verification.ipynb
import cv2
from datetime import datetime
import dlib
from skimage import io
from scipy.spatial import distance

# Файлы с моделями
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()

# Загружаем первую фотографию
img = io.imread('test1.jpg')

# Показываем фотографию средствами dlib
win1 = dlib.image_window()
win1.clear_overlay()
win1.set_image(img)

# Находим лицо на фотографии
dets = detector(img, 1)

for k, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        k, d.left(), d.top(), d.right(), d.bottom()))
    shape = sp(img, d)
    win1.clear_overlay()
    win1.add_overlay(d)
    win1.add_overlay(shape)

    # Извлекаем дескриптор из лица
    face_descriptor1 = facerec.compute_face_descriptor(img, shape)

video_capture = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    ##########
    dets = detector(frame, 1)
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))

        start = datetime.now()
        shape = sp(img, d)
        end = datetime.now()
        elapsed = (end - start).total_seconds()
        print(f'>> функция sp время выполнения: {elapsed}')
    ##############

        start = datetime.now()
        face_descriptor2 = facerec.compute_face_descriptor(img, shape)
        end = datetime.now()
        elapsed = (end - start).total_seconds()
        print(f'>> функция faceCascade.detectMultiScale время выполнения: {elapsed}')
    #############

    # Draw a rectangle around the faces
    #for (x, y, w, h) in faces:
    #    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

# Рассчитываем Евклидово расстояние между двумя дексрипторами лиц
# a = distance.euclidean(face_descriptor1, face_descriptor2)
