FROM python:3.8-slim

COPY requirements.txt /matIE/requirements.txt
WORKDIR /matIE
RUN pip install cython pybind11
RUN pip install torch==1.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r requirements.txt

COPY . /matIE
ENV PYTHONPATH=/matIE

CMD ["uvicorn", "matie_service:app", "--host", "0.0.0.0"]
