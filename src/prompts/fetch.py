FETCH_PROMPT_1 = (
    """I will provide you with a forecasting question and the background information for the question. I will then ask you to generate short search queries (up to {max_words} words each) that I'll use to find articles in GDELT to help answer the question.

Question:
{question}

Question Background:
{background}

Today's date: {date_begin}
Question close date: {date_end}

You must generate this exact amount of queries: {num_queries}

Start off by writing down sub-questions. Then use your sub-questions to help steer the search queries you produce.

Your response should take the following structure:
Thoughts:
{{ Insert your thinking here. }}
Search Queries:
{{ Insert the queries here. Use semicolons to separate the queries. }}""",
    (
        "QUESTION",
        "BACKGROUND",
        "DATES",
        "NUM_KEYWORDS",
        "MAX_WORDS",
    ),
)