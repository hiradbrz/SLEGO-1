
# We start from a Python 3.8 base image
FROM python:3.9

# Set a directory for our application
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Jupyter
RUN pip install jupyter

# Copy the rest of the application into the container
COPY . .

# Jupyter Notebooks Configuration
RUN jupyter notebook --generate-config --allow-root
#RUN echo "c.NotebookApp.password = u'sha1:1234567890abcdef1234567890abcdef:1234567890abcdef1234567890abcdef'" >> /root/.jupyter/jupyter_notebook_config.py

# Specify the command to run when the container starts
#CMD ["jupyter", "notebook", "--notebook-dir=/app", "--ip='0.0.0.0'", "--port=8888",  "--allow-root", "--NotebookApp.token=''"]
# Run the web service on container startup.
CMD panel serve app.py --address 0.0.0.0 --port 8080 --allow-websocket-origin="*"