from app.guards import (
    validate_report_input,
    validate_question_input,
    validate_retrieval_results,
    enforce_disclaimer,
)

if __name__ == "__main__":
    print(validate_report_input(""))
    print(validate_report_input("Short text"))

    print(validate_question_input(""))
    print(validate_question_input("What does this mean?"))

    print(validate_retrieval_results([]))
    print(validate_retrieval_results([{"text": "x"}]))

    print(enforce_disclaimer("This is a test answer."))
