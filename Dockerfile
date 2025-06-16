FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_trf

COPY . .

# Set the Streamlit port to 8502
ENV STREAMLIT_SERVER_PORT=8502

EXPOSE 8502

CMD ["streamlit", "run", "graphrag_app.py", "--server.address", "0.0.0.0"] 