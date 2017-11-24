'''
Created on 23 Nov 2017

@author: mtonnicchi
'''

from utils.mailreport import MailReport

import utils.config as cfg
from syslog import LOG_INFO, LOG_WARNING

''' Load config ''' 
cfg_mail_report = cfg.Config('config.ini').section('MAIL_REPORT')

mail_report = MailReport(cfg_mail_report['smpt_server_address'],cfg_mail_report['user'],cfg_mail_report['password'],cfg_mail_report['source_address'],cfg_mail_report['destination_address'])

log_label_info = "INFO"
log_label_warning = "WARNING"
log_label_error = "ERROR"

def log(log_level, message):
    print(str(log_level) + " : " + message)
    mail_report.add_message(str(log_level) + " -- " + message)

def info(message):
    log(log_label_info, message)

def warning(message):
    log(log_label_warning, message)
    
def error(message):
    log(log_label_error, message)
