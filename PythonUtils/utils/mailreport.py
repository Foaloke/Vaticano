'''
Created on 23 Nov 2017

@author: mtonnicchi
'''
import smtplib

class MailReport(object):

    def __init__(self, smpt_server_address, user, password, source_address, destination_address):
        self.smpt_server_address = smpt_server_address
        self.source_address = source_address
        self.destination_address = destination_address
        self.user = user
        self.password = password
        
        self.report = ""
        
        pass
    
    def add_message(self, line):
        self.report = self.report + "\n" + line
    
    def send(self, subject):
        
        self.server = smtplib.SMTP(self.smpt_server_address)
        self.server.ehlo()
        self.server.starttls()
        self.server.login(self.user, self.password)
        self.server.sendmail(self.source_address, self.destination_address, 'Subject: {}\n\n{}'.format(subject, self.report))
        self.server.quit()
        
    