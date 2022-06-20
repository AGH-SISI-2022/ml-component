FROM python:3

WORKDIR /usr/src/app

COPY requirements_k8.txt ./
RUN pip install --no-cache-dir -r requirements_k8.txt

COPY . .

CMD [ "python", "./main.py" ]