FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_trf

COPY . .

# Set Streamlit configuration
ENV STREAMLIT_SERVER_HEADLESS=true

EXPOSE 8502

# CMD ["streamlit", "run", "graphrag_app.py"] 
CMD ["streamlit", "run", "graphrag_app.py", "--server.port", "8502"]