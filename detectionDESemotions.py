import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Charger le modèle pré-entraîné
model = load_model('emotion_model3.h5')

# Dictionnaire des émotions
emotion_dict = {0: "Colere", 1: "Degout", 2: "Peur", 3: "Joie", 4: "Tristesse", 5: "Surprise", 6: "Neutre"}

# Initialiser la webcam
cap = cv2.VideoCapture(0)

while True:
    # Lire une image de la webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir l'image en niveaux de gris
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecter les visages
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extraire la région d'intérêt (ROI)
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype('float32') / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        # Prédire l'émotion
        predictions = model.predict(roi_gray)
        max_index = np.argmax(predictions[0])
        emotion = emotion_dict[max_index]

        # Dessiner un rectangle autour du visage et afficher l'émotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Afficher le résultat
    cv2.imshow('Detection des emotions', frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la webcam et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()