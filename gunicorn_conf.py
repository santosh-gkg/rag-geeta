from multiprocessing import cpu_count
# Add this line to your gunicorn_conf.py
umask = 0o007  # This sets the socket file permissions

# Socket Path
bind = 'unix:/home/fastapi/gunicorn.sock'

# Worker Options
workers = cpu_count() + 1
worker_class = 'uvicorn.workers.UvicornWorker'

# Logging Options
loglevel = 'debug'
accesslog = '/home/fastapi/access_log'
errorlog =  '/home/fastapi/error_log'
