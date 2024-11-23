import streamlit as st
import requests
import pandas as pd
from typing import Optional, Generator
import json
from queue import Empty
from error_types import HealthQueryErrorType

API_URL = "http://localhost:8000"

def init_session_state():
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    if "submitted" not in st.session_state:
        st.session_state.submitted = False
    if "question" not in st.session_state:
        st.session_state.question = ""
    if "error_state" not in st.session_state:
        st.session_state.error_state = None

def submit_callback():
    st.session_state.submitted = True
    st.session_state.error_state = None

def handle_error(error_data: dict, placeholder):
    """Display user-friendly error messages."""
    error_type = error_data.get('errorType', 'server_error')
    error_msg = error_data.get('error', '')
    detail = error_data.get('detail', '')
    
    # Map common database errors to user-friendly messages
    db_error_messages = {
        'unrecognized token': 'There was an issue with the database query. Please try rephrasing your question.',
        'no such column': 'The requested data field is not available in the database.',
        'syntax error': 'There was an issue generating the database query. Please try rephrasing your question.',
    }
    
    # Check if this is a database error and needs translation
    for error_pattern, friendly_message in db_error_messages.items():
        if error_pattern in error_msg.lower() or (detail and error_pattern in detail.lower()):
            error_msg = friendly_message
            break
    
    # If no specific mapping found, use a generic message for database errors
    if 'sql' in error_msg.lower() or 'database' in error_msg.lower():
        error_msg = "There was an issue processing your query. Please try rephrasing your question."
    
    with placeholder.container():
        st.error(error_msg)
    
    st.session_state.error_state = error_type

def stream_response(response: requests.Response) -> Generator[str, None, None]:
    """Stream response from the API endpoint."""
    if response.encoding is None:
        response.encoding = 'utf-8'
        
    for line in response.iter_lines(decode_unicode=True):
        if line:
            if line.startswith('data: '):
                try:
                    data = json.loads(line[6:])  # Skip 'data: ' prefix
                    
                    # Clean up error messages before yielding
                    if 'error' in data:
                        # If it's a database error, clean it up
                        error_msg = data.get('error', '')
                        if 'sql' in error_msg.lower() or 'database' in error_msg.lower():
                            data['error'] = "There was an issue processing your query. Please try rephrasing your question."
                            data['detail'] = error_msg  # Keep original error as detail
                    
                    yield data
                except json.JSONDecodeError as e:
                    yield {
                        'type': 'error',
                        'errorType': HealthQueryErrorType.SERVER_ERROR.value,
                        'error': "There was an issue processing the response. Please try again."
                    }
                    break

def display_result_progressively(question: str, spinner_placeholder):
    # Initialize placeholder containers
    query_placeholder = st.empty()
    data_placeholder = st.empty()
    analysis_placeholder = st.empty()
    error_placeholder = st.empty()
    
    try:
        # Phase 1: Stream SQL query generation
        current_query = ""
        final_query = None
        final_data = None

        with spinner_placeholder, st.spinner("Generating SQL query..."):
            try:
                response = requests.post(
                    f"{API_URL}/query/sql/stream",
                    json={"question": question},
                    stream=True,
                    headers={
                        'Accept': 'text/event-stream',
                        'Cache-Control': 'no-cache',
                        'Connection': 'keep-alive'
                    },
                    timeout=30
                )
            except requests.exceptions.Timeout:
                handle_error({
                    'errorType': HealthQueryErrorType.TIMEOUT_ERROR.value,
                    'error': 'The request timed out. Please try again.'
                }, error_placeholder)
                return
            except requests.exceptions.ConnectionError:
                handle_error({
                    'errorType': HealthQueryErrorType.CONNECTION_ERROR.value,
                    'error': 'Could not connect to the server. Please check your connection.'
                }, error_placeholder)
                return
                
            if response.status_code != 200:
                try:
                    error_data = response.json()
                    handle_error(error_data.get('detail', {}), error_placeholder)
                except:
                    handle_error({
                        'errorType': HealthQueryErrorType.SERVER_ERROR.value,
                        'error': f"Server error: {response.text}"
                    }, error_placeholder)
                return
            
            for chunk in stream_response(response):
                if chunk.get('error'):
                    handle_error(chunk, error_placeholder)
                    return
                    
                if chunk['type'] == 'token':
                    current_query += chunk['text']
                    with query_placeholder.container():
                        st.subheader("Generated SQL Query")
                        st.code(current_query, language="sql")
                elif chunk['type'] == 'complete':
                    final_query = chunk['text']
                    final_data = chunk.get('data')
                    with query_placeholder.container():
                        st.subheader("Generated SQL Query")
                        st.code(final_query, language="sql")
                    
                    if final_data:
                        df = pd.DataFrame(final_data)
                        with data_placeholder.container():
                            st.subheader("Query Results")
                            st.dataframe(df)
                    else:
                        handle_error({
                            'errorType': HealthQueryErrorType.DATA_VALIDATION.value,
                            'error': 'The query did not return any data.'
                        }, error_placeholder)
                        return

        if final_query and not st.session_state.error_state:
            # Phase 2: Stream analysis generation
            current_analysis = ""
            with spinner_placeholder, st.spinner("Generating analysis..."):
                try:
                    response = requests.post(
                        f"{API_URL}/query/analysis/stream",
                        json={
                            "question": question,
                            "sql_query": final_query,
                            "data": final_data or []
                        },
                        stream=True,
                        headers={
                            'Accept': 'text/event-stream',
                            'Cache-Control': 'no-cache',
                            'Connection': 'keep-alive'
                        },
                        timeout=30
                    )
                except requests.exceptions.Timeout:
                    handle_error({
                        'errorType': HealthQueryErrorType.TIMEOUT_ERROR.value,
                        'error': 'The analysis request timed out. Please try again.'
                    }, error_placeholder)
                    return
                except requests.exceptions.ConnectionError:
                    handle_error({
                        'errorType': HealthQueryErrorType.CONNECTION_ERROR.value,
                        'error': 'Lost connection to the server. Please try again.'
                    }, error_placeholder)
                    return
                
                if response.status_code != 200:
                    try:
                        error_data = response.json()
                        handle_error(error_data.get('detail', {}), error_placeholder)
                    except:
                        handle_error({
                            'errorType': HealthQueryErrorType.SERVER_ERROR.value,
                            'error': f"Server error during analysis: {response.text}"
                        }, error_placeholder)
                    return
                
                for chunk in stream_response(response):
                    if chunk.get('error'):
                        handle_error(chunk, error_placeholder)
                        return
                        
                    if chunk['type'] == 'token':
                        current_analysis += chunk['text']
                        with analysis_placeholder.container():
                            st.subheader("Analysis")
                            st.write(current_analysis)
                    elif chunk['type'] == 'complete':
                        final_analysis = chunk['text']
                        with analysis_placeholder.container():
                            st.subheader("Analysis")
                            st.write(final_analysis)
                        
                        if not st.session_state.error_state:
                            st.session_state.query_history.append({
                                "question": question,
                                "result": {
                                    "query": final_query,
                                    "data": final_data,
                                    "analysis": final_analysis
                                }
                            })
                    
    except Exception as e:
        handle_error({
            'errorType': HealthQueryErrorType.SERVER_ERROR.value,
            'error': str(e)
        }, error_placeholder)

def display_history_item(item):
    """Display a single history item with proper formatting."""
    st.write(f"Question: {item['question']}")
    
    if item['result'].get('query'):
        st.subheader("Generated SQL Query")
        st.code(item['result']['query'], language="sql")
    
    if item['result'].get('data'):
        st.subheader("Query Results")
        df = pd.DataFrame(item['result']['data'])
        st.dataframe(df)
    
    if item['result'].get('analysis'):
        st.subheader("Analysis")
        st.write(item['result']['analysis'])
    
    st.divider()

def main():
    st.set_page_config(
        page_title="Health Data Analysis",
        page_icon="🏥",
        layout="wide"
    )
    
    init_session_state()
    
    # Create sidebar with helpful information
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This system analyzes health data by answering your questions. You can ask about:
        - Patient demographics
        - Blood pressure and related conditions
        - Lifestyle factors (smoking, alcohol, salt intake)
        - Physical activity patterns
        - Health metrics (BMI, hemoglobin levels)
        - Stress levels and their impacts
        """)
        
        st.header("Tips for Better Questions")
        st.markdown("""
        - Be specific about what you want to analyze
        - Mention any particular groups or conditions
        - Specify if you want averages, counts, or comparisons
        - Include relevant factors or metrics
        
        Example questions:
        - What is the average BMI for patients with high stress levels?
        - Compare smoking rates between normal and abnormal blood pressure groups
        - How does physical activity vary with age?
        """)

    # Main content area
    st.title("Health Data Analysis System")
    
    with st.form(key='query_form'):
        question = st.text_input(
            "Enter your question about the health data:",
            placeholder="Example: What is the average BMI of patients with normal vs abnormal blood pressure?",
            key="question_input"
        )
        
        submit_button = st.form_submit_button(
            "Submit Query", 
            on_click=submit_callback
        )

    # Add warning for empty questions
    if st.session_state.submitted and not question.strip():
        st.warning("Please enter a question before submitting.")
        st.session_state.submitted = False
        return

    spinner_placeholder = st.empty()

    if st.session_state.submitted and question:
        st.session_state.submitted = False
        display_result_progressively(question, spinner_placeholder)
    
    # Display query history
    if st.session_state.query_history:
        with st.expander("Query History", expanded=False):
            for item in reversed(st.session_state.query_history):
                display_history_item(item)

if __name__ == "__main__":
    main()