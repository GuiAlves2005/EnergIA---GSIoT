import cv2
import mediapipe as mp
import numpy as np
import time

# Inicialização do MediaPipe Hands
mp_desenho       = mp.solutions.drawing_utils
mp_estilos       = mp.solutions.drawing_styles
maos             = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def ajustar_gamma(frame, gamma=1.0):
    """
    Ajusta brilho via correção de gamma.
    Retorna nova imagem em BGR.
    """
    inv_gamma = 1.0 / gamma
    tabela = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)], dtype="uint8")
    return cv2.LUT(frame, tabela)

def classificar_gesto(landmarks_mao, rotulo_mao):
    """
    Identifica gesto a partir dos 21 landmarks:
    - "PUNHO": punho fechado (todas as pontas TIP abaixo das juntas PIP, com margem)
    - "JOINHA": polegar para cima; demais dedos dobrados
    - "DESCONHECIDO": qualquer outro arranjo
    rotulo_mao indica "Right" ou "Left" para orientar o polegar.
    """
    lm = landmarks_mao.landmark
    ID_PONTA = [4, 8, 12, 16, 20]  # IDs das pontas (TIP) dos dedos
    ID_PIP   = [3, 6, 10, 14, 18]  # IDs das juntas PIP dos dedos
    margem   = 0.03  # margem mínima para descartar ruídos

    # 1) Detecta punho fechado: todas as pontas TIP estão abaixo (y maior) das PIP
    punho_fechado = True
    for ponta_id, pip_id in zip(ID_PONTA, ID_PIP):
        if lm[ponta_id].y + margem < lm[pip_id].y:
            # se qualquer ponta estiver acima da junta+marca → não é punho fechado
            punho_fechado = False
            break
    if punho_fechado:
        return "PUNHO"

    # 3) Detecta joinha: polegar para cima (TIP y + margem < PIP y) e demais dedos fechados
    polegar_cima = False
    if lm[4].y + margem < lm[3].y:
        outros_fechados = True
        for ponta_id, pip_id in zip(ID_PONTA[1:], ID_PIP[1:]):
            if lm[ponta_id].y + margem < lm[pip_id].y:
                # se algum dos outros dedos estiver aberto => não é joinha
                outros_fechados = False
                break
        if outros_fechados:
            polegar_cima = True
    if polegar_cima:
        return "JOINHA"

    return "DESCONHECIDO"

def main():
    captura = cv2.VideoCapture(0)
    captura.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    prev_time = 0

    # Contadores de debounce para cada gesto
    cont_punho   = 0
    cont_joinha  = 0

    while True:
        ret, frame = captura.read()
        if not ret:
            break

        # Espelha a imagem da webcam para alinhar "Right"/"Left"
        frame = cv2.flip(frame, 1)

        # Ajusta brilho via gamma
        img = ajustar_gamma(frame, gamma=0.6)

        # Converte para RGB para processar no MediaPipe
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resultados = maos.process(rgb)

        h, w, _ = img.shape
        gesto_detectado = "NONE"

        # Se alguma mão foi detectada, desenha os landmarks e classifica gesto
        if resultados.multi_hand_landmarks:
            for lm_mao, rotulo in zip(resultados.multi_hand_landmarks,
                                      resultados.multi_handedness):

                # Desenha pontos e conexões da mão
                mp_desenho.draw_landmarks(
                    img,
                    lm_mao,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp_estilos.get_default_hand_landmarks_style(),
                    mp_estilos.get_default_hand_connections_style()
                )

                rotulo_mao = rotulo.classification[0].label  # "Right" ou "Left"
                gesto = classificar_gesto(lm_mao, rotulo_mao)

                # DEBUG: imprime no terminal qual gesto foi identificado
                print(f"Mão: {rotulo_mao} — Gesto: {gesto}")

                # Atualiza contadores de debounce
                if gesto == "PUNHO":
                    cont_punho  += 1
                    cont_joinha  = 0
                elif gesto == "JOINHA":
                    cont_joinha += 1
                    cont_punho   = 0
                else:
                    cont_punho = cont_joinha = 0

                # Se o mesmo gesto ocorrer por 3 quadros consecutivos, valida-o
                if cont_punho >= 3:
                    gesto_detectado = "SOS"
                    cont_punho = 0
                elif cont_joinha >= 3:
                    gesto_detectado = "OK"
                    cont_joinha = 0

        # Exibe texto do gesto detectado ("SOS", "PAZ" ou "OK")
        if gesto_detectado != "NONE":
            cv2.putText(
                img,
                gesto_detectado,
                (10, 50),
                cv2.FONT_HERSHEY_DUPLEX,
                1.2,
                (0, 0, 255),
                3
            )

        # Calcula e exibe FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time
        cv2.putText(
            img,
            f"FPS: {int(fps)}",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        # Mostra a janela final
        cv2.imshow("EnergIA IoT - Gestos de Mão", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    captura.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
