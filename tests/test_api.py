import requests
import json
import logging
import hashlib
import time
import base64
from os import path

# 配置日志记录
logging.basicConfig(
    filename="test_api.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# 接口的 URL
url = "http://127.0.0.1:5000/api/v5/weight/calculate"

# 模拟 API 密钥
api_key = "your_api_key"

wrong_headers_1 = {"Content-Type": "application/json"}
wrong_headers_2 = {"X-Api-Key": "your_api_key_here"}

standard_input = {
    "info": "1.normal scenarios",
    "input_data": {
        "base_prompt": "未来主义城市景观",
        "attributes": [
            {
                "name": "风格",
                "type": "text",
                "value": "赛博朋克",
                "initial_weight": 0.7,
                "constraints": {
                    "min_weight": 0.4,
                    "max_weight": 0.9,
                    "conflict_terms": ["蒸汽朋克", "极简主义"],
                },
            },
            {
                "name": "光照",
                "type": "text",
                "value": "霓虹灯光",
                "initial_weight": 0.5,
                "constraints": {"min_weight": 0.2, "max_weight": 0.8},
            },
        ],
        "temperature": 1.8,
        "fallback_strategy": "creative",
        "debug_mode": True,
    },
}

def image_to_base64(file_path):
    """将图片转换为 Base64 编码"""
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return encoded_string.decode('utf-8')

def run_tests():
    try:
        # 从 examples.json 文件中读取测试用例
        with open("examples.json", "r", encoding="utf-8") as f:
            test_data = json.load(f)

        for example in test_data["examples"]:
            info = example["info"]
            input_data = example["input_data"]

            # 获取时间戳
            timestamp = str(int(time.time()))
            data_str = json.dumps(input_data)

            # 计算签名
            message = f"{data_str}{timestamp}{api_key}"
            signature = hashlib.sha256(message.encode()).hexdigest()

            headers = {
                "Content-Type": "application/json",
                "X-Api-Key": api_key,
                "X-Timestamp": timestamp,
                "X-Signature": signature,
            }

            try:
                # 发送 POST 请求
                response = requests.post(url, headers=headers, json=input_data)
                response.raise_for_status()
                result = response.json()
                logging.info(f"测试用例 {info} 的响应结果: {response.text}")
            except requests.RequestException as e:
                logging.error(
                    f"测试用例 {info} 发送请求时出错: {e} Output: {response.text}"
                )
            except json.JSONDecodeError as e:
                logging.error(
                    f"测试用例 {info} 解析响应 JSON 时出错: {e} Output: {response.text}"
                )
    except FileNotFoundError:
        logging.error("未找到 examples.json 文件，请确保文件存在。")
    except KeyError:
        logging.error("examples.json 文件格式有误，请检查是否包含 'examples' 字段。")

def other_examples():
    # 测试无请求头
    info = "28.lack headers"
    try:
        # 发送 POST 请求
        response = requests.post(url, json=standard_input)
        response.raise_for_status()
        result = response.json()
        logging.info(f"测试用例 {info} 的响应结果: {response.text}")
    except requests.RequestException as e:
        logging.error(f"测试用例 {info} 发送请求时出错: {e} Output: {response.text}")
    except json.JSONDecodeError as e:
        logging.error(
            f"测试用例 {info} 解析响应 JSON 时出错: {e} Output: {response.text}"
        )

    # 测试缺失 content-type 的请求头
    info = "29.lack content-type"
    try:
        # 发送 POST 请求
        response = requests.post(url, headers=wrong_headers_2, json=standard_input)
        response.raise_for_status()
        result = response.json()
        logging.info(f"测试用例 {info} 的响应结果: {response.text}")
    except requests.RequestException as e:
        logging.error(f"测试用例 {info} 发送请求时出错: {e} Output: {response.text}")
    except json.JSONDecodeError as e:
        logging.error(
            f"测试用例 {info} 解析响应 JSON 时出错: {e} Output: {response.text}"
        )

    # 测试缺少 x-api-key 的请求头
    info = "30.lack x-api-key"
    try:
        # 发送 POST 请求
        response = requests.post(url, headers=wrong_headers_1, json=standard_input)
        response.raise_for_status()
        result = response.json()
        logging.info(f"测试用例 {info} 的响应结果: {response.text}")
    except requests.RequestException as e:
        logging.error(f"测试用例 {info} 发送请求时出错: {e} Output: {response.text}")
    except json.JSONDecodeError as e:
        logging.error(
            f"测试用例 {info} 解析响应 JSON 时出错: {e} Output: {response.text}"
        )

    # 测试两个图片属性
    info = "31.two images"
    image1_path = path.join('examples', 'image1.jpg')
    image2_path = path.join('examples', 'image2.jpg')
    input_data = {
        "base_prompt": "未来主义城市景观",
        "attributes": [
            {
                "name": "风格",
                "type": "image",
                "value": {"base64": image_to_base64(image1_path)},
                "initial_weight": 0.7,
                "constraints": {
                    "min_weight": 0.4,
                    "max_weight": 0.9,
                    "conflict_terms": ["蒸汽朋克", "极简主义"],
                },
            },
            {
                "name": "光照",
                "type": "image",
                "value": {"base64": image_to_base64(image2_path)},
                "initial_weight": 0.5,
                "constraints": {"min_weight": 0.2, "max_weight": 0.8},
            },
        ],
        "temperature": 1.8,
        "fallback_strategy": "creative",
        "debug_mode": True,
    }
    timestamp = str(int(time.time()))
    data_str = json.dumps(input_data)
    message = f"{data_str}{timestamp}{api_key}"
    signature = hashlib.sha256(message.encode()).hexdigest()
    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": api_key,
        "X-Timestamp": timestamp,
        "X-Signature": signature,
    }
    try:
        response = requests.post(url, headers=headers, json=input_data)
        response.raise_for_status()
        result = response.json()
        logging.info(f"测试用例 {info} 的响应结果: {response.text}")
    except requests.RequestException as e:
        logging.error(f"测试用例 {info} 发送请求时出错: {e} Output: {response.text}")
    except json.JSONDecodeError as e:
        logging.error(
            f"测试用例 {info} 解析响应 JSON 时出错: {e} Output: {response.text}"
        )

    # 测试一个图片一个文本属性
    info = "32.one image one text"
    input_data = {
        "base_prompt": "未来主义城市景观",
        "attributes": [
            {
                "name": "风格",
                "type": "image",
                "value": {"base64": image_to_base64(image1_path)},
                "initial_weight": 0.7,
                "constraints": {
                    "min_weight": 0.4,
                    "max_weight": 0.9,
                    "conflict_terms": ["蒸汽朋克", "极简主义"],
                },
            },
            {
                "name": "光照",
                "type": "text",
                "value": "霓虹灯光",
                "initial_weight": 0.5,
                "constraints": {"min_weight": 0.2, "max_weight": 0.8},
            },
        ],
        "temperature": 1.8,
        "fallback_strategy": "creative",
        "debug_mode": True,
    }
    timestamp = str(int(time.time()))
    data_str = json.dumps(input_data)
    message = f"{data_str}{timestamp}{api_key}"
    signature = hashlib.sha256(message.encode()).hexdigest()
    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": api_key,
        "X-Timestamp": timestamp,
        "X-Signature": signature,
    }
    try:
        response = requests.post(url, headers=headers, json=input_data)
        response.raise_for_status()
        result = response.json()
        logging.info(f"测试用例 {info} 的响应结果: {response.text}")
    except requests.RequestException as e:
        logging.error(f"测试用例 {info} 发送请求时出错: {e} Output: {response.text}")
    except json.JSONDecodeError as e:
        logging.error(
            f"测试用例 {info} 解析响应 JSON 时出错: {e} Output: {response.text}"
        )

    # 测试一个图片一个向量属性
    info = "33.one image one vector"
    input_data = {
        "base_prompt": "未来主义城市景观",
        "attributes": [
            {
                "name": "风格",
                "type": "image",
                "value": {"base64": image_to_base64(image1_path)},
                "initial_weight": 0.7,
                "constraints": {
                    "min_weight": 0.4,
                    "max_weight": 0.9,
                    "conflict_terms": ["蒸汽朋克", "极简主义"],
                },
            },
            {
                "name": "向量特征",
                "type": "vector",
                "value": [0.1, 0.2, 0.3],
                "initial_weight": 0.5,
                "constraints": {"min_weight": 0.2, "max_weight": 0.8},
            },
        ],
        "temperature": 1.8,
        "fallback_strategy": "creative",
        "debug_mode": True,
    }
    timestamp = str(int(time.time()))
    data_str = json.dumps(input_data)
    message = f"{data_str}{timestamp}{api_key}"
    signature = hashlib.sha256(message.encode()).hexdigest()
    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": api_key,
        "X-Timestamp": timestamp,
        "X-Signature": signature,
    }
    try:
        response = requests.post(url, headers=headers, json=input_data)
        response.raise_for_status()
        result = response.json()
        logging.info(f"测试用例 {info} 的响应结果: {response.text}")
    except requests.RequestException as e:
        logging.error(f"测试用例 {info} 发送请求时出错: {e} Output: {response.text}")
    except json.JSONDecodeError as e:
        logging.error(
            f"测试用例 {info} 解析响应 JSON 时出错: {e} Output: {response.text}"
        )

def access_limit_test():
    # 测试访问限制
    info = "34.access limit"
    timestamp = str(int(time.time()))
    data_str = json.dumps(standard_input)
    message = f"{data_str}{timestamp}{api_key}"
    signature = hashlib.sha256(message.encode()).hexdigest()
    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": api_key,
        "X-Timestamp": timestamp,
        "X-Signature": signature,
    }

    # 发送多次请求以触发访问限制
    for i in range(50):
        response = requests.post(url, headers=headers, json=standard_input)

    try:
        response = requests.post(url, headers=headers, json=standard_input)
        response.raise_for_status()
        result = response.json()
        logging.info(f"测试用例 {info} 的响应结果: {response.text}")
    except requests.RequestException as e:
        logging.error(f"测试用例 {info} 发送请求时出错: {e} Output: {response.text}")
    except json.JSONDecodeError as e:
        logging.error(
            f"测试用例 {info} 解析响应 JSON 时出错: {e} Output: {response.text}"
        )
if __name__ == "__main__":
    logging.info("开始运行测试用例...")
    run_tests()
    other_examples()
    access_limit_test()
    logging.info("所有测试用例已完成。")