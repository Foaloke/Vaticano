'''
Created on 23 Nov 2017

@author: mtonnicchi
'''
import unittest
from utils.mailreport import MailReport

class MailReportTest(unittest.TestCase):

    def mailGoesThroughTest(self):
        mail_report = MailReport('smtp.gmail.com:587', 'alberello.caro', 'Alberelliasi', 'alberello.caro@gmail.com', 'matteo.tonnicchi@gmail.com')
        mail_report.add_message("TEST")
        mail_report.send()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()