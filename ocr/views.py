import json
import traceback

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from ocr.log import logger
from ocr.upload import Img

def _getArguments(request):
    # 获取请求参数，懒得改了，拿来就用
    if request.method == 'GET':
        logger.info("收到GET请求")
        arguments = dict(request.GET)
        for arg in arguments:
            if type(arguments[arg]) == type([]):
                arguments[arg] = arguments[arg][0]
    else:
        logger.info("收到POST请求")
        logger.info("post-body")
        logger.info(request.body.decode())
        if "form" in request.content_type:
            arguments = request.POST
        else:
            arguments = json.loads(request.body.decode())
    return arguments


def go_to_page(request):
    # if to == "imgOcr":
    arguments = _getArguments(request)
    pwd = arguments.get("pwd", None)
    if not pwd:
        return JsonResponse({"state": 201, "msg": "请输入密码", "data": None},
                            json_dumps_params={'ensure_ascii': False})
    if "123456" != pwd:
        return JsonResponse({"state": 202, "msg": "密码错误，请重新输入", "data": None},
                            json_dumps_params={'ensure_ascii': False})
    return render(request, 'auth_code_ocr.html')

@csrf_exempt
def cor_auth_code(request):
    try:
        return Img.upload_and_ocr(request)
    except Exception:
        traceback.print_exc()
        return JsonResponse({"state": 500, "msg": "系统出错，请联系管理员", "errMsg": traceback.format_exc()},
                            json_dumps_params={'ensure_ascii': False})