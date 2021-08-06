import cv2
import glob
import os

#Segmente edilmiş görüntüyü boyunlandırma - filtreleme :

for file in glob.glob(r"C:\Users\Hp\Desktop\DATASURGERY\DATASET\BENİGN\*.jpg"): 
    #Sırası ile ilk önce benign nodullerinin olduğu dosyaya ardından malign nodulleri dosyasına gidilir.
    #Ve içerisinde imagelar sırası ile file değişkenine atanır.
    print (file)
    image=cv2.imread(file) #İmage okunur
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image= cv2.medianBlur(gray,5)
    #Görüntüye medyan filtre uygulanmıştır. Bu şekilde gürültülerin büyük bir kısmı görüntüden arıldırılmıştır.
    #5*5 lik filtre uygulunarak ; resim yumuşatılmış ve yüksek frekanslı bölgeler kaldırılmıştır.
    resized=cv2.resize(image,(224,224),interpolation=cv2.INTER_CUBIC)
    #Filtrelenmiş nodül görüntüsü , 224*224 pixele formatlandırılmıştır.
    cv2.imwrite(f'{file}resized.jpg', resized) #modifiye edilen image kayıt edilir.
  
#Aynı zamanda imagelar üzerinde normalizasyon işlemide yapılcaktır.(Binary normalization) Bu kısmıda imageları
#modele vermeden  gerçekleştireceğiz.

