from typing import Literal

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama


# ============================================================
# LLM
# ============================================================

llm = ChatOllama(
    model="phi3",
    temperature=0
)


# ============================================================
# QUIZ GENERATION
# ============================================================

class Question(BaseModel):
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str

    correct_answer: Literal["A", "B", "C", "D"]

    explanation: str


class Quiz(BaseModel):
    questions: list[Question]


quiz_llm = llm.with_structured_output(Quiz)


QUIZ_SYSTEM_PROMPT = """
You are an AI quiz generator.

Generate a quiz ONLY from the provided document.

Rules:

1. Generate exactly 10 questions.
2. Each question must have exactly 4 options.
3. The options must be A, B, C and D.
4. Exactly one option must be correct.
5. The correct answer must be either A, B, C or D.
6. Every question must be answerable using ONLY information from the document.
7. Do not use outside knowledge.
8. Do not invent facts.
9. Questions should test understanding of the document.
10. Try to vary the difficulty.
11. Provide a short explanation for the correct answer.

The document is the only source of truth.
"""


# ============================================================
# ANSWER INTERPRETATION
# ============================================================

class UserResponse(BaseModel):
    action: Literal[
        "A",
        "B",
        "C",
        "D",
        "SKIP",
        "QUIT",
        "INVALID"
    ] = Field(
        description="The user's intended quiz action"
    )


answer_llm = llm.with_structured_output(UserResponse)


ANSWER_SYSTEM_PROMPT = """
You are an AI quiz answer interpretation agent.

Your job is to understand what the user wants to do
with the current multiple-choice quiz question.

Classify the user's response into exactly one of:

A
B
C
D
SKIP
QUIT
INVALID


Rules:

1. If the user selects option A, return A.
2. If the user selects option B, return B.
3. If the user selects option C, return C.
4. If the user selects option D, return D.
5. If the user wants to skip the question, return SKIP.
6. If the user wants to stop/end the quiz, return QUIT.
7. If the response is invalid or ambiguous, return INVALID.
8. Never determine whether the answer is correct.
9. Only interpret the user's intended action.


Examples:

"A" -> A

"a" -> A

"option A" -> A

"I choose A" -> A

"B" -> B

"option B" -> B

"I think option B is correct" -> B

"C" -> C

"D" -> D

"skip" -> SKIP

"s" -> SKIP

"I don't know" -> SKIP

"I don't know the answer" -> SKIP

"next question" -> SKIP

"move to the next question" -> SKIP

"quit" -> QUIT

"exit" -> QUIT

"stop the quiz" -> QUIT

"F" -> INVALID

"X" -> INVALID

"hello" -> INVALID

"maybe B or C" -> INVALID

"not sure if A or D" -> INVALID
"""