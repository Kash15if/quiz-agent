from typing import TypedDict


class QuizState(TypedDict):
    document: str
    questions: list

    current_question: int

    score: int
    correct: int
    wrong: int
    skipped: int

    user_input: str
    user_action: str

    finished: bool