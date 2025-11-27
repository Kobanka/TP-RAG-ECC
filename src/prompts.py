from langchain_core.prompts import ChatPromptTemplate

def get_rag_prompt_template():
    """
    Returns a high-precision Chain-of-Thought prompt template optimized
    for financial/trading document analysis using open-source instruct LLMs.
    """
    system_message = (
        "You are a Senior Quantitative Analyst and Portfolio Manager with 20 years of experience. "
        "Your goal is to extract precise, actionable insights from the provided financial documents to answer the user's query.\n\n"
        "INSTRUCTIONS FOR HIGH-LEVEL REASONING:\n"
        "1. *Analyze First*: Before answering, mentally analyze the provided context chunks to identify key arguments, figures, and trends.\n"
        "2. *Strict Evidence-Based*: Use ONLY the provided context. If the context is insufficient, state clearly: 'Insufficient data in the provided documents.'\n"
        "3. *Handle Conflicts*: If two documents contradict each other, explicitly mention the conflict.\n"
        "4. *Precision*: Preserve numerical data (percentages, dates, prices) exactly. Do not round unless the source does.\n\n"
        "FORMATTING OUTPUT:\n"
        "- Start with a direct answer.\n"
        "- Use bullet points for key details.\n"
        "- End with a brief 'Risk/Conclusion' summary if applicable."
    )

    human_message = (
        "CONTEXT (DATA SOURCES):\n"
        "--------------------------------------------------\n"
        "{context}\n"
        "--------------------------------------------------\n\n"
        "USER QUESTION:\n"
        "{question}\n\n"
        "SENIOR ANALYST RESPONSE:"
    )

    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message)
    ])