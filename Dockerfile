FROM python:3.11-slim
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD [ "jupyter","lab" ]