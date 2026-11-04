import time
import cv2
import mediapipe as mp
from flask import Flask, Response, jsonify, render_template

app = Flask(__name__)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7
)

tip_ids = [4, 8, 12, 16, 20]

# Variáveis globais de estado
estado_atual = "NUM1"  # NUM1, OP, NUM2, RESULTADO
val1 = 0
operacao = ""
val2 = 0
resultado = "-"
total_dedos_global = 0

ultimo_numero_detectado = -1
tempo_inicio_gesto = time.time()
tempo_necessario = 3.0


def contar_dedos(hand_landmarks, hand_label):
  dedos = []
  if hand_label == "Right":
    if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
      dedos.append(1)
    else:
      dedos.append(0)
  else:
    if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0] - 1].x:
      dedos.append(1)
    else:
      dedos.append(0)

  for id in range(1, 5):
    if (
        hand_landmarks.landmark[tip_ids[id]].y
        < hand_landmarks.landmark[tip_ids[id] - 2].y
    ):
      dedos.append(1)
    else:
      dedos.append(0)

  return dedos.count(1)


def gerar_frames():
  global estado_atual, val1, operacao, val2, resultado, total_dedos_global
  global ultimo_numero_detectado, tempo_inicio_gesto

  cap = cv2.VideoCapture(0)

  while cap.isOpened():
    success, frame = cap.read()
    if not success:
      break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    total_dedos = 0
    mao_detectada = False

    if results.multi_hand_landmarks:
      mao_detectada = True
      for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        hand_label = results.multi_handedness[idx].classification[0].label
        num_dedos = contar_dedos(hand_landmarks, hand_label)
        total_dedos += num_dedos
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    total_dedos_global = total_dedos
    tempo_restante = tempo_necessario

    if estado_atual != "RESULTADO":
      if not mao_detectada:
        tempo_inicio_gesto = time.time()
        ultimo_numero_detectado = -1
      else:
        if total_dedos == ultimo_numero_detectado:
          tempo_decorrido = time.time() - tempo_inicio_gesto
          tempo_restante = max(0, round(tempo_necessario - tempo_decorrido, 1))

          if tempo_decorrido >= tempo_necessario:
            if estado_atual == "NUM1":
              val1 = total_dedos
              estado_atual = "OP"
            elif estado_atual == "OP":
              mapa_ops = {1: "+", 2: "-", 3: "*", 4: "/"}
              operacao = mapa_ops.get(total_dedos, "+")
              estado_atual = "NUM2"
            elif estado_atual == "NUM2":
              val2 = total_dedos
              if operacao == "+":
                resultado = val1 + val2
              elif operacao == "-":
                resultado = val1 - val2
              elif operacao == "*":
                resultado = val1 * val2
              elif operacao == "/":
                resultado = (
                    round(val1 / val2, 2) if val2 != 0 else "Erro (Div 0)"
                )
              estado_atual = "RESULTADO"

            tempo_inicio_gesto = time.time()
            ultimo_numero_detectado = -1
        else:
          ultimo_numero_detectado = total_dedos
          tempo_inicio_gesto = time.time()
    else:
      if mao_detectada and total_dedos == 0:
        estado_atual = "NUM1"
        val1 = 0
        val2 = 0
        operacao = ""
        resultado = "-"
        ultimo_numero_detectado = -1
        time.sleep(0.5)

    ret, buffer = cv2.imencode(".jpg", frame)
    frame = buffer.tobytes()
    yield (
        b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
    )

  cap.release()


@app.route("/")
def index():
  return render_template("index.html")


@app.route("/video_feed")
def video_feed():
  return Response(
      gerar_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
  )


# Rota para o JavaScript atualizar a interface em tempo real
@app.route("/status")
def status():
  return jsonify({
      "estado": estado_atual,
      "val1": val1,
      "operacao": operacao,
      "val2": val2,
      "resultado": resultado,
      "dedos": total_dedos_global,
  })

# Função para limpar e reiniciar os estados
def resetar_calculadora():
  global estado_atual, val1, operacao, val2, resultado, ultimo_numero_detectado, tempo_inicio_gesto
  estado_atual = "NUM1"
  val1 = 0
  operacao = ""
  val2 = 0
  resultado = "-"
  ultimo_numero_detectado = -1
  tempo_inicio_gesto = time.time()


# Rota que o botão "Cancelar" do HTML chama via POST
@app.route("/cancelar", methods=["POST"])
def cancelar():
  resetar_calculadora()
  return jsonify({"status": "sucesso"})

if __name__ == "__main__":
  app.run(debug=True)