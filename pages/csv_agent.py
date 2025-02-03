
import streamlit as st
from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.tools import PythonAstREPLTool
from langchain_teddynote.messages import AgentStreamParser, AgentCallbacks
from typing import List, Union
import pandas as pd
import matplotlib.pyplot as plt

load_dotenv()
logging.langsmith("csv_agent")


# streamlit 앱 설정
st.title("CSV Data Analysis Agent")

# 세션 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [] # 대화내용 저장리스트 초기화


# 상수 정의 
class MessageRole :
    """
    메시지 역할 정의
    """
    USER = "user" # 사용자
    ASSISTANT = "assistant" # 봇

class MessageType :
    """
    메시지 유형 정의 클래스
    """
    TEXT = "text" # 텍스트 메세지
    FIGURE = "figure" # 그림 메세지
    CODE = "code" # 코드 메세지
    DATAFRAME = "dataframe" # 데이터프레임 메세지


# 메시지 관련 함수
def print_message():
    """
    저장된 메시지를 화면에 출력하는 함수
    """
    for role, content_list in st.session_state["messages"]:
        with st.chat_message(role):
            for content in content_list:
                if isinstance(content, list):
                    message_type, message_content = content
                    if message_type == MessageType.TEXT:
                        st.markdown(message_content) # text print
                    elif message_type == MessageType.FIGURE:
                        st.pyplot(message_content) # 그림 print
                    elif message_type == MessageType.CODE:
                        with st.status("코드 출력", expanded=True) :
                            st.code(message_content, language="python") # 코드 print
                    elif message_type == MessageType.DATAFRAME:
                        st.dataframe(message_content) # 데이터프레임 print
                else:
                    raise ValueError(f"알수없는 콘텐츠 유형 : {content}")

def add_mesage(role : MessageRole, content: List[Union[MessageType, str]]): # Union : 여러 MessageType(code, figure, dataframe, text) 허용
    """
    새로운 메시지를 저장하는 함수
    
    Args:
        role (MessageRole) : 메시지의 역할(user, assistant)
        content (List[Union[MessageType, str]]) : 메시지의 내용(MessageType, str)
    """
    messages = st.session_state["messages"]
    if messages and messages[-1][0] == role :
        messages[-1][1].append(content) # 같은 역할의 연속된 메시지는 하나로 합침
    else : 
        messages.append([role, [content]]) # 새로운 메시지 추가

# 사이드바 설정
with st.sidebar : 
    clear_btn = st.button("Clear")      # 대화내용 초기화 버튼
    uploaded_file = st.file_uploader(    # csv 파일 업로드 기능
        "Upload CSV File",
        type=["csv"],
        accept_multiple_files=False
    )
    selected_model = st.selectbox(      # OpenAI 모델 선택 옵션
       "Select OpenAI model",
       ["gpt-4o", "gpt-3.5-turbo", "gpt-4o-mini"],
       index=0 # 초기 선택 인덱스
    )
    apply_btn= st.button("Start Data Analysis") # 데이터 분석 시작 버튼


# 콜백 함수
def tool_callback(tool) -> None:
    """
    도구 실행 결과를 처리하는 콜백함수

    Args:
        tool(dict) : 실행된 도구 정보
    """
    if tool_name := tool.get("tool"):
        if tool_name == "python_repl_ast":
            tool_input = tool.get("tool_input", {})
            query = tool_input.get("query")
            if query :
                df_in_result = None
                with st.status("데이터분석중 ...", expanded=True) as status:
                    st. markdown(f"```python\n{query}\n```")
                    add_mesage(MessageRole.ASSISTANT, [MessageType.CODE, query])
                    if "df" in st.session_state:
                        result = st.session_state["python_tool"].invoke(
                            {"query" : query}
                        )
                        if isinstance(result, pd.DataFrame):
                            df_in_result = result
                    status.update(label="Code Print", state="complete", expanded=False)

                if df_in_result is not None:
                    st.dataframe(df_in_result)
                    add_mesage(MessageRole.ASSISTANT, [MessageType.DATAFRAME, df_in_result])
                
                if "plt.show" in query:
                    fig = plt.gcf()
                    st.pyplot(fig)
                    add_mesage(MessageRole.ASSISTANT, [MessageType.FIGURE, fig])

                return result
            else :
                st.error("데이터프레임이 정의되지 않았습니다. CSV파일을 먼저 업로드하세요.")
                return
            

def observation_callback(observation) -> None:
    """
    관찰 결과를 처리하는 콜백 함수

    Args:
        observation(dict) : 관찰 결과
    """
    if "observation" in observation:
        obs = observation["observation"]
        if isinstance(obs, str) and "Error" in obs:
            st.error(obs)
            st.session_state["messages"][-1][1].clear() # 에러 발생 시 마지막 메시지 삭제

def result_callback(result:str) -> None:
    """
    최종 결과를 처리하는 콜백 함수
    
    Args:
        result(str) : 최종 결과
    """
    pass # 현재는 아무것도 하지 않음

from langchain_openai import ChatOpenAI
# agent 생성 함수
def create_agent(dataframe, selected_model="gpt-4o"):
    """
    데이터프레임 에이전트를 생성하는 함수
    
    Args:
        dataframe(pd.DataFrame): 분석할 데이터프레임
        selected_model (str, optional): 사용할 OpenAI 모델. 기본값은 "gpt-4o"

    """
    return create_pandas_dataframe_agent(
        ChatOpenAI(model=selected_model, temperature=0),
        dataframe, 
        verbose=False,
        agent_type="tool-calling",
        allow_dangerous_code=True,
        prefix="You are a professional data analyst and expert in Pandas."
        "You must use Pandas DataFrame(`df`) to answer user's question."
        "\n\n[Important] DO NOT create or overwrite the `df` variable in your code. \n\n"
        "If you are willing to generate visualization code, please use `plot.show()` at the end of your code"
        "I prefer seaborn code for visualization, but you can use matplotlib as well."
        "\n\n<Visualization Preference\n"
        "- [Important] Use `Envlish` for your visualization title and labels."
        "- `muted` cmap;, white background and no grid for your visualization."
        "\nRecommend to set cmap, palette parameter for seaborn plot if it is applicable."
        "The language of final answer should be written in Korean."
        "\n\n###\n\n<Column Guidelines>\n"
        "If user asks with columns that are not listed in `df.columns, you may refer to the most similar columns listed below.\n"
    )


# 질문 처리 함수
def ask(query):
    """
    사용자의 질문을 처리하고 응답을 생성하는 함수
    
    Args
        query(str) : 사용자의 질문
    """
    if "agent" in st.session_state:
        st.chat_message("user").write(query)
        add_mesage(MessageRole.USER, [MessageType.TEXT, query])

        agent = st.session_state["agent"]
        response = agent.stream({"input" : query})

        ai_answer = ""
        parser_callback = AgentCallbacks(
            tool_callback, observation_callback, result_callback
        )
    stream_parser = AgentStreamParser(parser_callback)

    with st.chat_message("assistant"):
        for step in response:
            stream_parser.process_agent_steps(step)
            if "output" in step:
                ai_answer += step["output"]
        st.write(ai_answer)

    add_mesage(MessageRole.ASSISTANT, [MessageType.TEXT, ai_answer])


# 메인로직
if clear_btn:
    st.session_state["messages"] = [] # 대화내용 초기화

if apply_btn and uploaded_file:
    try:
        loaded_data= pd.read_csv(uploaded_file,encoding="utf-8") # 업로드된 파일 로드
        st.session_state["df"] = loaded_data # 데이터프레임 저장
        st.session_state["python_tool"] = PythonAstREPLTool() # 파이썬 도구 생성
        st.session_state["python_tool"].locals["df"] = loaded_data # 데이터프레임을 python 실행 환경에 추가
        st.session_state["agent"] = create_agent(loaded_data, selected_model) # 에이전트 생성
        st.success("설정이 완료되었습니다. 대화를 시작해보세요.")
    except Exception as e:
        st.error(f"파일을 읽는 중 오류 발생 : {e}")
elif apply_btn:
    st.warning("파일을 업로드하세요")

print_message() # 저장된 메세지 출럭

user_input = st.chat_input("질문을 입력하세요")
if user_input:
    ask(user_input)# 질문 처리



