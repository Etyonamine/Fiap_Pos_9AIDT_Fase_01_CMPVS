# ────────────────────────────────────────────────────────────────────────────
# Trecho a ser ADICIONADO ao app.py do repositório Fiap_Pos_9AIDT_Fase_01_API
# ────────────────────────────────────────────────────────────────────────────
#
# 1. Adicionar no bloco de imports (após os imports existentes):
#
#    from schemas.output_schema_melanoma import output_schema_melanoma
#    from predict_melanoma import load_melanoma_model, predict_melanoma as run_predict_melanoma
#
# ─────────────────────────────────────────────────────────────────────────────
# 2. Adicionar em swagger_template["definitions"] (junto com InputModel / OutputModel):
#
#    "OutputModelMelanoma": output_schema_melanoma,
#
# ─────────────────────────────────────────────────────────────────────────────
# 3. Adicionar após o bloco try/except que carrega os modelos joblib:

_MELANOMA_CKPT = "model/efficientnet_b4_melanoma.pth"
melanoma_model = None
try:
    melanoma_model = load_melanoma_model(_MELANOMA_CKPT)
except FileNotFoundError:
    logger.warning(
        "Checkpoint do modelo de melanoma não encontrado em %s. "
        "O endpoint /predict/melanoma estará indisponível.",
        _MELANOMA_CKPT,
    )

# ─────────────────────────────────────────────────────────────────────────────
# 4. Adicionar como novo endpoint (após o endpoint /predict existente):


@app.route("/predict/melanoma", methods=["POST"])
def predict_melanoma_endpoint():
    """
    Predição de melanoma a partir de imagem de lesão cutânea e metadados clínicos.
    ---
    tags:
      - Melanoma
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: image
        type: file
        required: true
        description: "Imagem JPEG/PNG da lesão cutânea"
      - in: formData
        name: age_approx
        type: number
        required: false
        description: "Idade aproximada do paciente em anos (ex.: 45)"
      - in: formData
        name: anatom_site
        type: string
        required: false
        description: >
          Localização anatômica da lesão. Valores aceitos:
          torso, lower extremity, upper extremity, head/neck, palms/soles, oral/genital
      - in: formData
        name: tta_steps
        type: integer
        required: false
        default: 5
        description: "Número de passos de Test-Time Augmentation (1–10)"
    responses:
      200:
        description: Resultado da predição de melanoma
        schema:
          $ref: '#/definitions/OutputModelMelanoma'
      400:
        description: Requisição inválida (imagem ausente ou formato inválido)
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Envie um arquivo de imagem no campo 'image'."
      503:
        description: Modelo não carregado
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Modelo de melanoma não disponível."
      500:
        description: Erro interno do servidor
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Erro interno ao processar a imagem."
    """
    if melanoma_model is None:
        return jsonify({"error": "Modelo de melanoma não disponível."}), 503

    if "image" not in request.files:
        return jsonify({"error": "Envie um arquivo de imagem no campo 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Nenhum arquivo selecionado."}), 400

    try:
        tta_steps = int(request.form.get("tta_steps", 5))
        tta_steps = max(1, min(10, tta_steps))
    except (ValueError, TypeError):
        return jsonify({"error": "Valor inválido para 'tta_steps': deve ser um inteiro entre 1 e 10."}), 400

    age_approx_raw = request.form.get("age_approx")
    age_approx: float | None = None
    if age_approx_raw is not None:
        try:
            age_approx = float(age_approx_raw)
            if age_approx < 0 or age_approx > 120:
                return jsonify({"error": "Valor inválido para 'age_approx': deve ser entre 0 e 120."}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "Valor inválido para 'age_approx': deve ser um número."}), 400

    anatom_site: str | None = request.form.get("anatom_site") or None

    try:
        image_bytes = file.read()
        result = run_predict_melanoma(
            image_bytes, melanoma_model,
            age_approx=age_approx,
            anatom_site=anatom_site,
            tta_steps=tta_steps,
        )
    except Exception as exc:
        logger.exception("Erro ao processar predição de melanoma: %s", exc)
        return jsonify({"error": "Erro interno ao processar a imagem."}), 500

    return jsonify(result), 200
