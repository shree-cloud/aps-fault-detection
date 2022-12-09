import sys, os


def error_message_detail(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = "Erroroccured in python script name [{0}] line number[{1}] error message: [{2}]"(
        file_name,exc_tb.tb_lineno, str(error)
    )

class SensorException(Exception):

    def __init__(self,error_message, error_detail:sys):
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self):
        returnself.error_message