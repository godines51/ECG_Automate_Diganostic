################################################################################
# app.py – Servidor Flask para inferência de ECG                         
# Autor: Gabriel H. V. Braum (2025)                                      
# Descrição: API e UI web com pré-visualização de imagem e retorno de    
# probabilidades para todas as classes mapeadas                          
################################################################################

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict

from pathlib import Path
import requests

from flask import Flask, jsonify, render_template_string, request
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

###############################################################################
# Configurações gerais                                                        #
###############################################################################
MODEL_PATH: Path = Path("ecg_classifier.pth")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_FILE_MB = 5

MODEL_PATH = Path("ecg_classifier.pth")
FILE_ID = "1DtRXlB0NQHom05rWrCxTXVbkKatdC6qe"  # só o ID, exemplo: 1DxR...C5qe

if not MODEL_PATH.exists():
    print("Baixando modelo do Google Drive...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, str(MODEL_PATH), quiet=False)
    print("Modelo baixado com sucesso!")
    

# Rótulos traduzidos ----------------------------------------------------------
LABELS_PT = {
    "MI_Images_2880":          "Infarto agudo do miocárdio",
    "Abnormal_Heartbeat_2796": "Arritmia",
    "History_MI_2064":         "Histórico de infarto (cicatriz)",
    "Normal_Person_3408":      "ECG normal",
}

###############################################################################
# Utilitários                                                                  #
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def load_model(path: Path) -> tuple[torch.nn.Module, List[str]]:
    """Carrega o modelo salvo em disco."""
    ckpt    = torch.load(path, map_location=DEVICE)
    classes = ckpt["classes"]

    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc    = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(DEVICE).eval()
    logging.info("Modelo carregado com %d classes: %s", len(classes), classes)
    return model, classes


MODEL, CLASSES = load_model(MODEL_PATH)

# Pipeline de pré‑processamento ----------------------------------------------
TFM = T.Compose([
    T.Grayscale(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.5], [0.5]),
])

@torch.inference_mode()
def predict_image(img: Image.Image) -> Dict[str, any]:
    """Recebe PIL.Image, devolve dict com predição e probabilidades."""
    x = TFM(img).unsqueeze(0).to(DEVICE)  # (1,1,224,224)
    logits = MODEL(x)
    probs  = torch.softmax(logits, 1).squeeze(0).cpu().tolist()  # lista de floats

    # mapeia classes às probabilidades em % e traduz
    prob_map = {
        LABELS_PT.get(cls, cls): round(p * 100, 2)
        for cls, p in zip(CLASSES, probs)
    }
    # obtém classe de maior probabilidade
    best_idx = int(torch.argmax(logits, 1).item())
    best_class = CLASSES[best_idx]
    return {
        "predicao": LABELS_PT.get(best_class, best_class),
        "probabilidade": prob_map[LABELS_PT.get(best_class, best_class)],
        "todas_probabilidades": prob_map,
    }

###############################################################################
# Flask App                                                                    #
###############################################################################
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_MB * 1024 * 1024

# --- UI ----------------------------------------------------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html lang=\"pt-br\"><meta charset=\"utf-8\">
<head>
  <title>Classificador de ECG</title>
  <link href=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css\" rel=\"stylesheet\">
  <style>
    body{background:#f8fafc;margin:0;padding:2rem;font-family:system-ui,Arial,sans-serif}
    .card-drop{border:2px dashed #ced4da;padding:2rem;text-align:center;color:#6c757d;cursor:pointer}
    #results{white-space:pre-wrap}
    #preview{max-width:100%%;max-height:400px;margin-bottom:1rem;display:none}
  </style>
</head>
<body>
<div class=\"container\">
  <h1 class=\"mb-4 text-center\">Classificador de ECG</h1>

  <div class=\"card p-4 mb-4 shadow-sm\">
    <!-- Preview da imagem carregada -->
    <img id=\"preview\" src=\"\" alt=\"Pré-visualização da imagem\" />
    <form id=\"form\" enctype=\"multipart/form-data\" class=\"vstack gap-3\">
      <input id=\"file\" type=\"file\" name=\"file\" accept=\"image/*\" required class=\"form-control\">
      <button class=\"btn btn-primary w-100\" type=\"submit\">Enviar</button>
    </form>
  </div>

  <div id=\"results\" class=\"card p-3 shadow-sm\" style=\"display:none\"></div>
</div>
<script>
const form   = document.getElementById('form');
const fileIn = document.getElementById('file');
const preview = document.getElementById('preview');
const resDiv = document.getElementById('results');

// Mostrar pré-visualização ao selecionar imagem
fileIn.onchange = () => {
  const file = fileIn.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = e => {
      preview.src = e.target.result;
      preview.style.display = 'block';
    };
    reader.readAsDataURL(file);
  }
};

// Enviar e processar
form.onsubmit = async e => {
  e.preventDefault();
  const data = new FormData(form);
  resDiv.style.display = 'block';
  resDiv.innerHTML = 'Processando…';
  const r = await fetch('/predict', { method: 'POST', body: data });
  if (!r.ok) {
    resDiv.textContent = 'Erro: ' + r.statusText;
    return;
  }
  const json = await r.json();

  // Exibir resultados formatados
  const out = json.map(item => {
    const probs = item.todas_probabilidades;
    let html = `<h5>${item.predicao} (${item.probabilidade}%)</h5><ul>`;
    for (const [label, pct] of Object.entries(probs)) {
      html += `<li>${label}: ${pct}%</li>`;
    }
    html += '</ul>';
    return html;
  }).join('');

  resDiv.innerHTML = out;
};
</script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    """Página principal com interface web."""
    return render_template_string(HTML_PAGE)


@app.route("/health", methods=["GET"])
def healthcheck():
    """Endpoint de status."""
    return jsonify({"status": "ok", "device": str(DEVICE)}), 200


@app.route("/predict", methods=["POST"])
def predict_route():
    """Recebe imagem e retorna JSON com probabilidades."""
    files = request.files.getlist("file")
    if not files:
        return jsonify({"erro": "nenhum arquivo recebido"}), 400

    respostas = []
    for f in files:
        try:
            img = Image.open(f.stream).convert("RGB")
            resp = predict_image(img)
            logging.info("Processado %s → %s (%.2f%%)", f.filename, resp["predicao"], resp["probabilidade"])  
            respostas.append(resp)
        except Exception as exc:
            logging.exception("Erro ao processar %s", f.filename)
            respostas.append({"arquivo": f.filename, "erro": str(exc)})

    return jsonify(respostas)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

