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
    },
    "required": ["probabilidade_melanoma", "classificacao", "alerta"],
}
