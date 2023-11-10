import sys
from src.logger import logging
def error_message(err,errdetail:sys):
    _,_,errr = errdetail.exc_info()
    file_name = errr.tb_frame.f_code.co_filename
    message = "Error Occured In python Sycript line [{0}] line number [{1}] error message [{2}]".format(
        file_name,errr.tb_lineno,str(err)
    )
    return message

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message(error_message,errdetail=error_detail)

    def __str__(self):
        return self.error_message