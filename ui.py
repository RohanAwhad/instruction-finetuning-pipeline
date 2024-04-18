"""UI to interact with the model"""
import streamlit as st
import requests

st.title("Text Generation")
st.write("### Enter a prompt")
prompt = st.text_area("")

with st.sidebar:
    temperature = st.slider('Temperature', min_value=0.0, max_value=2.0, value=0.7, step=0.01)
    max_new_tokens = st.slider('Max Tokens', min_value=1, max_value=1024, value=128, step=1)
    stop_tokens = st.text_input(
        'Stop Tokens',
        placeholder='Add comma (,) separated values',
    )
    do_stream = st.checkbox('Stream?', value=True)

if st.button("Generate"):
    st.write("### Generated Text")

    url = "http://localhost:50505/generate"
    req_body = dict(
        prompt = prompt,
        temperature = temperature,
        stop_tokens = [x.strip() for x in stop_tokens.split(',') if x != ''] if stop_tokens else [],
        max_new_tokens = max_new_tokens,
        do_stream = do_stream
    )
    with st.container():
        text_element = st.empty()
        if do_stream:
            gen_text = ''
            with requests.Session() as sess:
                res = sess.post(url, json=req_body, stream=True)
                if res.status_code == 200:
                    #print(help(res.iter_content))
                    #for line in res.iter_lines():
                    for chunk in res.iter_content(chunk_size=128, decode_unicode=True):
                        if chunk:
                            #text_area.value += line.decode('utf-8')
                            #gen_text += '\n' + line.decode('utf-8')
                            gen_text += chunk
                            text_element.write(gen_text)
        else:
            response = requests.post(url, json=req_body)
            text_element.text_area('', response.json())

