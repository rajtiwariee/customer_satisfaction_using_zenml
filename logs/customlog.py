import os
import sys
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = "logs"
log_filepath = os.path.join(log_dir,"running_logs.log")
os.makedirs(log_dir, exist_ok=True)


logging.basicConfig(
    level= logging.INFO,
    format= logging_str,

    handlers=[
        logging.FileHandler(log_filepath), # this will print my output in the passed filepath
        logging.StreamHandler(sys.stdout)#this will print the output in the terminal
    ]
)

logger = logging.getLogger("customersatisfactionLogger")

#cnnClassifier is considered as my main local folder so i can access any file from it 
#this will be useful when we have our application at production we can't get the terminal logs so we can download the log file and make the changes so very useful