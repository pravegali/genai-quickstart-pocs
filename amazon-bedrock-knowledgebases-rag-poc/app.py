import streamlit as st
from query_against_knowledgeBases import answer_query
from query_against_knowledgeBases import get_contexts

# Header/Title of streamlit app
st.title(f""":rainbow[RAG with Amazon Knowledge Bases]""")

# configuring values for session state
if "messages" not in st.session_state:
    st.session_state.messages = []
# writing the message that is stored in session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
# adding some special effects from the UI perspective
st.balloons()
# evaluating st.chat_input and determining if a question has been input
if question := st.chat_input("Ask about your data stored in Amazon Knowledge Bases"):
    # with the user icon, write the question to the front end
    with st.chat_message("user"):
        st.markdown(question)
    # append the question and the role (user) as a message to the session state
    st.session_state.messages.append({"role": "user",
                                      "content": question})
    # respond as the assistant with the answer
    with st.chat_message("assistant"):
        # making sure there are no messages present when generating the answer
        message_placeholder = st.empty()
        # putting a spinning icon to show that the query is in progress
        with st.status("Determining the best possible answer!", expanded=False) as status:
            # passing the question into the OpenSearch search function, which later invokes the llm
            result = answer_query(question)
            # #printing the result
            # print(result)
            # writing the answer to the front end
            message_placeholder.markdown(f"{result['answer']}")
            
            # showing a completion message to the front end
            status.update(label="Question Answered...", state="complete", expanded=False)
            
            # writing the context to the front end
            st.write(f"Context: {result['context']}")
            
            
    # appending the results to the session state
    st.session_state.messages.append({"role": "assistant",
                                      "content": result['answer']})