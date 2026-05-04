output_schema_melanoma = {
    "type": "object",
    "properties": {
        "probabilidade_melanoma": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Probabilidade estimada de melanoma (0–1)",
        },
        "classificacao": {
            "type": "string",
            "enum": ["maligno", "benigno"],
            "description": "Classificação final da lesão cutânea",
        },
        "alerta": {
            "type": "boolean",
            "description": "true se probabilidade ≥ 0.5",
        },
        "age_approx": {
            "type": ["number", "null"],
            "description": "Idade aproximada utilizada na predição (null se não informada)",
        },
        "anatom_site": {
            "type": ["string", "null"],
            "description": (
                "Localização anatômica utilizada na predição (null se não informada). "
                "Valores possíveis: torso, lower extremity, upper extremity, "
                "head/neck, palms/soles, oral/genital"
            ),
        },
    },
    "required": ["probabilidade_melanoma", "classificacao", "alerta", "age_approx", "anatom_site"],
}
