

from app import * 


def get_response_if_context(vectorstore ,query , title_id , name):
    # Retrieve relevant context
    retriever =  vectorstore.as_retriever(search_type="similarity_score_threshold",
                                        search_kwargs={"score_threshold": 0.25,
                                                        "k": 5})
    docs = retriever.invoke(query)
    ranked_docs = rerank.invoke({"docs": docs, "question": query})
    context = format_docs(ranked_docs)

    if len(context) > 3000:
        context = context[:3000]
    


    # Load local and external chat history
    if title_id not in [None, 0]:
        full_history = fetch_chat_history(title_id , max_pairs=20)
    else:
        full_history = ""
    local_context = load_context_file()
    combined_history = (full_history or []) + local_context
    print("Local Context: " , local_context) 
    print("🎈🎈🎈"*10)
    print("Combined_history: " , combined_history) 
    print("🎀"*30)
    selected_hist = select_relevant_history(
        query,
        combined_history,
        embeddings,
        k_similar=4,
        last_n=4,
        max_chars=2000,
    ) or [] 
    print("selected_hist: " , selected_hist) 
    selected_hist.append({"user": query, "response": ""})
    chat_history_text = format_history_for_prompt(selected_hist)


    system_prompt = f"""
            "You are Arya — a warm, polite, and expert real-estate assistant. "
            "Respond by politely them back **using their name**. The user's name is: {name}.\n"
            "Your single source of truth is the section called 'Knowledge'. " 
            "Treat the Knowledge content as verified, up-to-date, and directly relevant to the user's query. "
            "If the Knowledge includes any details about the user’s question (e.g., price, area, project name, BHK type, developer), " 
            "you must answer using that information directly and confidently. "
            "⚠️ Preserve all numbers *exactly as written in the Knowledge section*, including zeros and commas (e.g., 1000, 25000, 3.50). Never round, truncate, or reformat them. "
            "Do not ask for the project or developer again — use what is provided in Knowledge. "
            "Only if Knowledge is completely empty should you ask a follow-up. " "Don't Use Certainly, first Conversation in response. "
            "Your tone should be empathetic, natural, and professional — like a helpful real estate consultant. " 
            "Avoid generic responses or repeating the user's query. " "Be concise, accurate, and factual." """

    return f"""
        <|system|>
        {system_prompt} 
        Your task:
            1. Identify the **latest user query** from the chat history.
            2. Ignore all assistant messages.
            3. Use ONLY the latest user query to generate the final answer.
            4. Use the Knowledge section as the factual source.
            5. Do NOT ask the user to repeat their question. 
        <|end|>
        <|user|>

        Chat History: Extract the correct question from chat history.
        {chat_history_text}

        The following Knowledge is guaranteed to be relevant to this user's query — it has been carefully retrieved from verified real estate data. Use it to answer directly.

        Knowledge:
        {context}

    
        <|end|>
        <|assistant|>
        """  


# print(get_response_if_context(vectorstore=vectorstore , query="1bhk" , title_id=1 , name="Rohit")) 