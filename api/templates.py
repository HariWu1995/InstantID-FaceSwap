from copy import deepcopy
from enum import Enum
from typing import Union, Dict
from pydantic import BaseModel as Template
from fastapi import Form, UploadFile
    

class OutputAPI(object):

    output_success = {
        'code': 200,
        'status': 'Success',
        'message': 'Request is responded',
        'data': {},
    }

    output_failure = {
        'code': 400,
        'status': 'Failed',
        'message': '',
        'data': {},
    }

    def __init__(self) -> None:
        pass

    def result(self, is_successful: bool, data: Dict = None, err_log: str = None,):
        if is_successful:
            response = deepcopy(self.output_success)
            response.update({   
                'data': data,
            })

        else:
            response = deepcopy(self.output_failure)
            response.update({
                'message': err_log,
            })

        return response


