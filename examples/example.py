from summarizer_graph import process_text

result = process_text(
    question="Купит ли Илон Маск ТикТок до 15-го апреля 2025 года?",
    text="""WASHINGTON (Reuters) - US President Donald Trump said on Tuesday..."""
)

print(f"Суммаризация: {result.summary}")
print(f"Вероятность события: {result.probability}")
