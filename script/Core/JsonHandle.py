import json

# 判断文件编码是否为utf-8
def is_utf8bom(pathfile):
    if b'\xef\xbb\xbf' == open(pathfile, mode='rb').read(3):
        return True
    return False

# 载入json文件
def _loadjson(filepath):
    if is_utf8bom(filepath):
        ec='utf-8-sig'
    else:
        ec='utf-8'
    with open(filepath, 'r', encoding=ec) as f:
        try:
            jsondata = json.loads(f.read())
        except json.decoder.JSONDecodeError:
            print(filepath + '  无法读取，文件可能不符合json格式')
            jsondata = []
    return jsondata