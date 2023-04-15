import os
import subprocess
from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
from gpt_index import GPTSimpleVectorIndex

import gradio as gr


os.environ["OPENAI_API_KEY"] = 'sk-aYtJiuEi' # api key


def generate_pdf(doc_path, path):
    # Return the path of the generated PDF file
    subprocess.call(['soffice', '--headless', '--convert-to', 'pdf', '--outdir', path, doc_path])
    return os.path.join(path, os.path.splitext(os.path.basename(doc_path))[0] + ".pdf")


def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 12000
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="text-davinci-003", max_tokens=num_outputs))
    documents = SimpleDirectoryReader(directory_path).load_data()
    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index.save_to_disk('index.json')
    return 'Index file genrated'

fold_name = 'D:/' # folder path
index = construct_index(os.path.join(fold_name, "docs"))
print('Knowledge source ', index)

input_text = 'full name of rahul'
index = GPTSimpleVectorIndex.load_from_disk('index.json')
response = index.query(input_text, response_mode="compact")

print('response',response)



def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    return response.response

iface = gr.Interface(fn=chatbot,
                     inputs=gr.components.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="Custom-trained AI Chatbot")

iface.launch(share=True)
