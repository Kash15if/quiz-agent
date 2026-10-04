from agent.graph import quiz_graph


initial_state = {
    "document": "",
    "questions": [],
    "current_question": 0,
    "score": 0,
    "correct": 0,
    "wrong": 0,
    "skipped": 0,
    "user_input": "",
    "user_action": "",
    "finished": False,
}


quiz_graph.invoke(initial_state)