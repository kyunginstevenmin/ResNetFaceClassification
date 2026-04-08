FROM public.ecr.aws/deep-learning-containers/pytorch-training:2.7.1-gpu-py312-cu128-ubuntu22.04-sagemaker
COPY docker/requirements.txt /opt/ml/code/requirements.txt
RUN pip install --no-cache-dir -r /opt/ml/code/requirements.txt

# Copy both model.py and train.py — train.py imports from model.py
COPY src/ /opt/ml/code/

ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
ENV SAGEMAKER_PROGRAM train.py
ENV PYTHONUNBUFFERED=1
