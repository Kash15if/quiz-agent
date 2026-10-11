from langgraph.graph import StateGraph, START, END

from .state import QuizState
from .tools import retrieve_context
from .agent import (
    quiz_llm,
    answer_llm,
    QUIZ_SYSTEM_PROMPT,
    ANSWER_SYSTEM_PROMPT,
)


# ============================================================
# GENERATE QUIZ
# ============================================================

def generate_quiz(state: QuizState):

    document = ""

    while not document:

        topic = input(
            "\nEnter a topic to generate the quiz from: "
        ).strip()

        if not topic:
            topic = "Java fundamentals"

        print(
            f"\nRetrieving content about '{topic}' "
            "from the vector store...\n"
        )

        document = retrieve_context(topic)

        if not document:
            print("No relevant content found. Try another topic.")

    print("Generating quiz from the retrieved content...\n")

    prompt = f"""
{QUIZ_SYSTEM_PROMPT}

Here is the content retrieved from the knowledge base:

---------------- CONTEXT ----------------

{document}

-------------- END CONTEXT --------------

Generate the quiz now.
"""

    quiz = quiz_llm.invoke(prompt)

    return {
        **state,
        "document": document,
        "questions": [
            question.model_dump()
            for question in quiz.questions
        ],
        "current_question": 0,
        "score": 0,
        "correct": 0,
        "wrong": 0,
        "skipped": 0,
        "finished": False,
    }


# ============================================================
# ASK QUESTION
# ============================================================

def ask_question(state: QuizState):

    index = state["current_question"]

    # Defensive check
    if index >= len(state["questions"]):
        return {
            **state,
            "finished": True
        }

    question = state["questions"][index]

    print("\n")
    print("=" * 60)
    print(f"Question {index + 1}/10")
    print("=" * 60)

    print(f"\n{question['question']}")

    print(f"\nA. {question['option_a']}")
    print(f"B. {question['option_b']}")
    print(f"C. {question['option_c']}")
    print(f"D. {question['option_d']}")

    print("\nType A/B/C/D, 'skip', or 'quit'.")

    user_input = input("\nYour answer: ").strip()

    return {
        **state,
        "user_input": user_input,
    }


# ============================================================
# INTERPRET USER ANSWER
# ============================================================

def interpret_answer(state: QuizState):

    user_input = state["user_input"]

    prompt = f"""
{ANSWER_SYSTEM_PROMPT}

The user's response is:

"{user_input}"
"""

    result = answer_llm.invoke(prompt)

    return {
        **state,
        "user_action": result.action,
    }


# ============================================================
# HANDLE ANSWER
# ============================================================

def evaluate_answer(state: QuizState):

    action = state["user_action"]

    index = state["current_question"]

    question = state["questions"][index]

    # --------------------------------------------------------
    # INVALID
    # --------------------------------------------------------

    if action == "INVALID":

        print("\n❌ Invalid selection.")
        print("Please choose A, B, C, D, skip, or quit.")

        # Don't move to next question
        return {
            **state,
        }

    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    if action == "QUIT":

        print("\nQuiz ended by user.")

        return {
            **state,
            "finished": True,
        }

    # --------------------------------------------------------
    # SKIP
    # --------------------------------------------------------

    if action == "SKIP":

        print("\n⏭️ Question skipped.")

        return {
            **state,
            "skipped": state["skipped"] + 1,
            "current_question": index + 1,
        }

    # --------------------------------------------------------
    # ANSWER A/B/C/D
    # --------------------------------------------------------

    correct_answer = question["correct_answer"]

    if action == correct_answer:

        print("\n✅ Correct!")

        print(
            f"Explanation: {question['explanation']}"
        )

        return {
            **state,
            "score": state["score"] + 4,
            "correct": state["correct"] + 1,
            "current_question": index + 1,
        }

    else:

        print("\n❌ Wrong!")

        print(
            f"Correct answer: {correct_answer}"
        )

        print(
            f"Explanation: {question['explanation']}"
        )

        return {
            **state,
            "score": state["score"] - 1,
            "wrong": state["wrong"] + 1,
            "current_question": index + 1,
        }


# ============================================================
# ROUTING
# ============================================================

def route_after_interpretation(state: QuizState):

    action = state["user_action"]

    if action == "INVALID":

        return "ask_question"

    return "evaluate_answer"


def route_after_evaluation(state: QuizState):

    if state["finished"]:
        return "final_result"

    if state["current_question"] >= len(state["questions"]):
        return "final_result"

    return "ask_question"


# ============================================================
# FINAL RESULT
# ============================================================

def final_result(state: QuizState):

    total_answered = (
        state["correct"]
        + state["wrong"]
    )

    total_questions = len(state["questions"])

    accuracy = (
        (state["correct"] / total_answered) * 100
        if total_answered > 0
        else 0
    )

    print("\n")
    print("=" * 60)
    print("                 QUIZ COMPLETE")
    print("=" * 60)

    print(f"\nCorrect : {state['correct']}")
    print(f"Wrong   : {state['wrong']}")
    print(f"Skipped : {state['skipped']}")

    print(
        f"\nScore   : {state['score']} / 40"
    )

    print(
        f"Accuracy: {accuracy:.2f}%"
    )

    print("\n" + "=" * 60)

    return {
        **state,
        "finished": True,
    }


# ============================================================
# BUILD GRAPH
# ============================================================

builder = StateGraph(QuizState)


builder.add_node(
    "generate_quiz",
    generate_quiz
)

builder.add_node(
    "ask_question",
    ask_question
)

builder.add_node(
    "interpret_answer",
    interpret_answer
)

builder.add_node(
    "evaluate_answer",
    evaluate_answer
)

builder.add_node(
    "final_result",
    final_result
)


# ------------------------------------------------------------
# Edges
# ------------------------------------------------------------

builder.add_edge(
    START,
    "generate_quiz"
)

builder.add_edge(
    "generate_quiz",
    "ask_question"
)

builder.add_edge(
    "ask_question",
    "interpret_answer"
)


builder.add_conditional_edges(
    "interpret_answer",
    route_after_interpretation,
    {
        "ask_question": "ask_question",
        "evaluate_answer": "evaluate_answer",
    }
)


builder.add_conditional_edges(
    "evaluate_answer",
    route_after_evaluation,
    {
        "ask_question": "ask_question",
        "final_result": "final_result",
    }
)


builder.add_edge(
    "final_result",
    END
)


quiz_graph = builder.compile()