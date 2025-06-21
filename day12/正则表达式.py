import re

text = ("我的电话号码是 (123) 456-7890，邮箱是 Eviltrashcan@gmail.com。访问我的网站 https://www.sdafa5848/9saf.com 或者联系客服电话 800-123-4567。日期是 "
        "2023-10-05，时间 15:30。IP地址 192.168.1.1，信用卡号 4111-1111-1111-1111。")
phone_number = re.search(r"\(\d{3}\) \d{3}-\d{4}", text)
email = re.search(r"\b[\w.-]+@gmail.com\b", text)
url = re.search(r"https?://[\w./-]+", text)
tel_num = re.search(r"\d{3}-\d{3}-\d{4}", text)
data_num = re.search(r"\d{4}-\d{2}-\d{2}", text)
time_num = re.search(r"\d{2}:\d{2}", text)
ip = re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", text)
credit_num = re.search(r"\d{4}-\d{4}-\d{4}-\d{4}", text)
total_phone_num = re.findall(r"\(\d{3}\) \d{3}-\d{4}|\d{3}-\d{3}-\d{4}", text)
print(f'我的电话是{phone_number.group()}, 邮箱是{email.group()}, 网址是{url.group()}, 客服电话{tel_num.group()}, 现在时间是{data_num.group()} {time_num.group()},IP地址为{ip.group()}, 信用卡号为{credit_num.group()}')
print(f'所有的电话号码：{total_phone_num}')
new_text = re.sub(r"\(123\)", "(987)", text)
print(new_text)
