#python tem que estar como administrador, se não tiver pode ocorrer erros
# IMPORTANTE !!! Sempre dar um 'python -m pip install --upgrade pip' antes de instalar a biblioteca 

import cv2  # ---> 'pip install opencv-python'

# URL do servidor IP da câmera
url = "http://xxx.xxx.xxx.x:xxxx/video" # Se não tiver webcam, usar essa forma de acesso para usar a camera do celular

#Se tiver camera no pc ou notebook, comentar a linha a cima para não ocorrer erro 

# Conecte-se ao servidor IP
cap = cv2.VideoCapture(url) # Se tiver camera no notebook ou Pc, substituir a url por 0 ou 1 


#XML significa Extensible Markup Language (Linguagem de Marcação Extensível) e é uma linguagem de marcação usada para armazenar e transportar dados. Ele é muito similar ao HTML, porém, ao invés de definir como exibir os dados, o XML descreve o que os dados são.
# Carregue o classificador de cascata Haar para detecção de rosto, olhos e boca
rosto_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
olho_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
boca_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

while True:
    
    
    #Frame: Quadro de vídeo, também conhecido como frames de vídeo ou frames por segundo
    # Capture um frame
    ret, frame = cap.read()

    # Converter o frame em escala de cinza para melhorar o desempenho
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecte rostos na imagem
    rostos = rosto_cascade.detectMultiScale(gray, 1.3, 5)

    # Desenhe retângulos ao redor dos rostos detectados, olhos e boca
    for (x,y,w,h) in rostos:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        olhos = olho_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in olhos:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        boca = boca_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
        for (mx,my,mw,mh) in boca:
            cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(0,0,255),2)

    # Mostre a imagem com os retângulos desenhados
    cv2.imshow('IP Webcam - Face Detection', frame)

    # Verifique se a tecla 'q' foi pressionada para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libere a captura e feche a janela
cap.release()
cv2.destroyAllWindows()

