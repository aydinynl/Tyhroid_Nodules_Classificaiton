import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2

img = cv2.imread('benign1.jpg')  #İlgili image dosyasının pathi girilir.
nodul_model_path = 'thyroidnodules.h5'  #Sınıflandırma modelimiz değişkene atanır.
nodul_classifier = load_model(nodul_model_path, compile=False)  #Sınıflandırma modelimiz eklenir.
NODULS = ["benign","malign"] #Sınıf isimlerini barından bir liste oluşturulur.

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Görüntü grayscale ya çevirilir.
roi = cv2.resize(gray, (224, 224)) #Modelimizi eğitirken görüntüler 224*224 formatına getirildiği için deneme image'da bu formata getirilir.
roi = roi.astype("float") / 255.0  #Modelimizi eğitirken normalizasyon(0-1) yaptığımız için deneme image içinde 0-1 dönüşüm yapılır.
roi = img_to_array(roi)   #dizine çevrilir.
roi = np.expand_dims(roi, axis=0) 
preds = nodul_classifier.predict(roi)[0] #sınıflandırıcı modelde tahmin işlemi yürütülür.
emotion_probability = np.max(preds)  #max olan sınıf ve sırası ile nodul listesinde hangisine denk geliyor ise 
label = NODULS[preds.argmax()]  #label değişkine atama yapılır.
    
deneme = cv2.putText(img,label,(50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,0),2)
cv2.imshow("Sonuç", deneme)
cv2.waitKey(0)
cv2.destroyAllWindows()

