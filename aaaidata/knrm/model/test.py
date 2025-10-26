import sys
#import os
# 将包含 team2box 包的目录添加到 sys.path
sys.path.append('/home/dyx2/team2box')
sys.path.append('/home/dyx2/team2box/team2box/aaaidata/knrm/model')

try:
    # 尝试导入模块
    from team2box.aaaidata.knrm.model import BaseNN
    print("模块导入成功！")
except Exception as e:
    print("模块导入失败：", e)