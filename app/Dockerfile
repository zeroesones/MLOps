FROM python:3.9.6

WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt

EXPOSE 8000
#CMD [ "python", "./main.py" ]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]