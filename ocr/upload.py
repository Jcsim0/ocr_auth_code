# -*- coding: utf-8 -*-
"""
Created with：PyCharm
@Author： Jcsim
@Date： 2021-1-28 15:43
@Project： ocr_auth_code
@File： upload.py
@Blog：https://blog.csdn.net/weixin_38676276
@Description： 
@Python：
"""
import json
import time
import traceback

from django.http import JsonResponse

from ocr.ocrModel import doCheakByBytes
from ocr_auth_code import settings


class Img:
    # 上传图片
    @staticmethod
    def upload_and_ocr(request):
        if request.method == "POST":
            try:
                classify = request.POST['classify']
                # 接收文件
                file = request.FILES['file']  # 上传的文件
                # 判断是否有 file
                if not file:
                    return JsonResponse({'state': 201, 'msg': '请上传文件'}, json_dumps_params={'ensure_ascii': False})

                if not classify:
                    return JsonResponse({'state': 201, 'msg': '请填写文件分类'}, json_dumps_params={'ensure_ascii': False})
                # print(classify)
                # 限制文件大小
                if file.size > 10 * 1024 * 1024:
                    return JsonResponse({'state': 202, 'msg': '请上传10MB以内的文件'},
                                        json_dumps_params={'ensure_ascii': False})

                # 设置允许上传的文件格式
                ALLOW_EXTENSIONS = ['png', 'jpg', 'jpeg']

                # 文件名
                filename = str(time.time()).split(".")[0] + '.' + file.name.split(".")[-1]

                # 图片路径
                # path = settings.BASE_DIR + '/static/img/' + classify + '/'

                # 判断文件格式
                if file.name.rsplit('.', 1)[1].lower() in ALLOW_EXTENSIONS:
                    pass
                    # 直接流处理，无需创建任何文件和文件夹 by fangnan 2021/01/28
                    # 假如文件夹不存在,创建文件夹

                    # if not os.path.exists(path):
                    #     os.makedirs(path)
                else:
                    return JsonResponse({"state": 203, 'msg': '文件格式错误，请上传 png, jpg, jpeg'},
                                        json_dumps_params={'ensure_ascii': False})
                # print('path:', path)
                # print('folder:',folder)

                # 直接流处理，无需创建任何文件和文件夹 by fangnan 2021/01/28
                # with open(path + filename, 'wb') as f:
                #     data = file.file.read()  # 文件字节流数据 从文件中读取整个上传的数据。小心整个方法：如果这个文件很大，你把它读到内存中会弄慢你的系统。
                #     f.write(data)
                #
                #     # for buf in file.chunks():  # 如果上传的文件足够大需要分块就返回真。默认的这个值是2.5M，当然这个值是可以调节的
                #     #     f.write(buf)
                image = file.file.read()
                data = {
                    # "url": '/' + classify + '/' + filename,
                    # "absolute_url": path + filename,
                    # "auth_code": doCheak(path + filename)
                    "auth_code": doCheakByBytes(image)

                }
                return JsonResponse({'state': 200, 'msg': '上传成功', "data": json.dumps(data)},
                                    json_dumps_params={'ensure_ascii': False})
            except Exception as e:
                traceback.print_exc()
                return JsonResponse({"state": 500, "msg": "系统出错，请联系管理员", "errMsg": traceback.format_exc()},
                                    json_dumps_params={'ensure_ascii': False})
