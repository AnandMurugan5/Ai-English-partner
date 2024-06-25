import os
from langchain import ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_huggingface import HuggingFaceEndpoint

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

try:
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_length=128,
        temperature=0.5,
    )

except Exception as e:
    print(f"Error initializing AzureChatOpenAI: {e}")
    raise


template = """You are an English language partner. Your job is to help improve communication skills by evaluating and correcting sentences for grammar, providing scores, and giving feedback. 
Follow the ReAct (Reasoning + Acting) approach.

        Steps:
            
            Understand the sentence:

                Evaluate the sentence for clarity and coherence.
                Identify any grammatical errors or awkward phrasing.
                Reasoning:

                    Explain the identified issues in the sentence.
                    Provide reasons why certain words or phrases are incorrect or unclear.
                    Suggest improvements and corrections with clear explanations.
                Action:

                Correct the sentence.
                    Provide a score (1-10) based on the following criteria:
                    Grammar: Is the sentence grammatically correct?
                    Clarity: Is the sentence clear and easy to understand?
                    Fluency: Does the sentence flow naturally?
                    
            Example Interaction:

                User: "I has went to the store and buyed some fruits."

                English Partner:

                    Currection:

                        "I has" should be "I have" because "has" is used with third person singular subjects.
                        "Went" is correct for past tense, but "buyed" is incorrect; the past tense of "buy" is "bought."
                    score:

                        Corrected Sentence: "I have gone to the store and bought some fruits."
                        Score:
                        Grammar: 4/10 (several errors)
                        Clarity: 7/10 (main idea is clear despite errors)
                        Fluency: 6/10 (sentence structure needs improvement for natural flow)
            Human: {human_input}
            Assitant 
            """


def model_speech(text):
    prompt = PromptTemplate(input_variables= ["human_input"], template=template)

    chatgpt_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=ConversationBufferWindowMemory(k=2),
    )
    response  = chatgpt_chain.predict(human_input=text)
    return response

# if __name__ == '__main__':
#     t = """Convert audio into text transcriptions and integrate speech recognition into applications with easy-to-use APIs.

# Get up to 60 minutes for transcribing and analyzing audio free per month.* New customers also get up to $300 in free credits to try Speech-to-Text and other Google Cloud products."""
#     text_res = model_speech(t)
#     print(text_res)
    