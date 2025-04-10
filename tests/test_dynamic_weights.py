import json
import numpy as np
import cv2  # 导入 cv2 模块
import binascii
from utils.dynamic_weights import validate_request, adjust_weights, detect_conflict, load_conflicts, \
    calculate_semantic_score, verify_signature, rate_limit, get_device_type, get_geo_location, \
    encode_context, base64_to_opencv, extract_image_features, handle_conflicts, calculate_weights
from unittest.mock import patch
import requests
from flask import Flask
import torch
import base64
from utils.dynamic_weights import app as original_app

app = Flask(__name__)


def test_verify_signature():
    with app.app_context():
        with app.test_request_context():
            # 正常签名验证
            with patch('utils.dynamic_weights.request') as mock_request:
                mock_request.headers.get.side_effect = ["timestamp", "signature"]
                mock_request.get_data.return_value = b"data"
                result = verify_signature(mock_request)
                assert isinstance(result, bool)

            # 缺少时间戳
            with patch('utils.dynamic_weights.request') as mock_request:
                mock_request.headers.get.side_effect = [None, "signature"]
                mock_request.get_data.return_value = b"data"
                result = verify_signature(mock_request)
                assert not result

            # 缺少签名
            with patch('utils.dynamic_weights.request') as mock_request:
                mock_request.headers.get.side_effect = ["timestamp", None]
                mock_request.get_data.return_value = b"data"
                result = verify_signature(mock_request)
                assert not result


def test_rate_limit():
    with app.app_context():
        with app.test_request_context():
            # 正常请求
            with patch('utils.dynamic_weights.redis_client') as mock_redis:
                with patch('utils.dynamic_weights.request') as mock_request:
                    mock_request.remote_addr = "127.0.0.1"
                    mock_redis.incr.return_value = 1
                    result = rate_limit(mock_request)
                    assert isinstance(result, bool)
                    assert result

            # 超出请求限制
            with patch('utils.dynamic_weights.redis_client') as mock_redis:
                with patch('utils.dynamic_weights.request') as mock_request:
                    mock_request.remote_addr = "127.0.0.1"
                    mock_redis.incr.return_value = 51
                    result = rate_limit(mock_request)
                    assert isinstance(result, bool)
                    assert not result


def test_validate_request():
    # 有效请求
    data = {
        "base_prompt": "A dog on the grass",
        "attributes": [
            {
                "name": "object",
                "type": "text",
                "value": "dog",
                "initial_weight": 0.8
            },
            {
                "name": "background",
                "type": "text",
                "value": "grass",
                "initial_weight": 0.2
            }
        ]
    }
    valid, error = validate_request(data)
    assert valid

    # 请求体为空
    data = {}
    valid, error = validate_request(data)
    assert not valid

    # 缺少 base_prompt 和 attributes
    data = {
        "attributes": [
            {
                "name": "object",
                "type": "text",
                "value": "dog",
                "initial_weight": 0.8
            }
        ]
    }
    valid, error = validate_request(data)
    assert not valid

    # attributes 数组为空
    data = {
        "base_prompt": "A dog on the grass",
        "attributes": []
    }
    valid, error = validate_request(data)
    assert not valid

    # initial_weight 超出范围
    data = {
        "base_prompt": "A dog on the grass",
        "attributes": [
            {
                "name": "object",
                "type": "text",
                "value": "dog",
                "initial_weight": 1.2
            },
            {
                "name": "background",
                "type": "text",
                "value": "grass",
                "initial_weight": 0.2
            }
        ]
    }
    valid, error = validate_request(data)
    assert not valid

    # type 字段无效
    data = {
        "base_prompt": "A dog on the grass",
        "attributes": [
            {
                "name": "object",
                "type": "invalid_type",
                "value": "dog",
                "initial_weight": 0.8
            },
            {
                "name": "background",
                "type": "text",
                "value": "grass",
                "initial_weight": 0.2
            }
        ]
    }
    valid, error = validate_request(data)
    assert not valid

    # text 类型的 value 不是字符串
    data = {
        "base_prompt": "A dog on the grass",
        "attributes": [
            {
                "name": "object",
                "type": "text",
                "value": 123,
                "initial_weight": 0.8
            },
            {
                "name": "background",
                "type": "text",
                "value": "grass",
                "initial_weight": 0.2
            }
        ]
    }
    valid, error = validate_request(data)
    assert not valid

    # image 类型的 value 不包含 base64 字段
    data = {
        "base_prompt": "A dog on the grass",
        "attributes": [
            {
                "name": "object",
                "type": "image",
                "value": {},
                "initial_weight": 0.8
            },
            {
                "name": "background",
                "type": "text",
                "value": "grass",
                "initial_weight": 0.2
            }
        ]
    }
    valid, error = validate_request(data)
    assert not valid

    # vector 类型的 value 不是列表
    data = {
        "base_prompt": "A dog on the grass",
        "attributes": [
            {
                "name": "object",
                "type": "vector",
                "value": 123,
                "initial_weight": 0.8
            },
            {
                "name": "background",
                "type": "text",
                "value": "grass",
                "initial_weight": 0.2
            }
        ]
    }
    valid, error = validate_request(data)
    assert not valid


def test_adjust_weights():
    attributes = [
        {
            "name": "object",
            "type": "text",
            "value": "dog",
            "initial_weight": 0.8
        },
        {
            "name": "background",
            "type": "text",
            "value": "grass",
            "initial_weight": 0.2
        }
    ]
    temperature = 1.2
    final_weights, adjustment_log = adjust_weights(attributes, temperature)
    assert isinstance(final_weights, dict)
    assert isinstance(adjustment_log, list)

    # 测试空属性列表
    attributes = []
    final_weights, adjustment_log = adjust_weights(attributes, temperature)
    assert isinstance(final_weights, dict)
    assert isinstance(adjustment_log, list)


def test_detect_conflict():
    attr1 = {
        "name": "object",
        "type": "text",
        "value": "dog"
    }
    attr2 = {
        "name": "background",
        "type": "text",
        "value": "grass"
    }
    conflicts = load_conflicts()
    conflict = detect_conflict(attr1, attr2, conflicts)
    assert isinstance(conflict, bool)

    # 测试显式冲突
    conflicts = {"conflict_cases": [{"name": "dog+grass"}]}
    conflict = detect_conflict(attr1, attr2, conflicts)
    assert conflict


def test_calculate_semantic_score():
    value1 = {"type": "text", "value": "dog"}
    value2 = {"type": "text", "value": "cat"}
    score = calculate_semantic_score(value1, value2)
    assert isinstance(score, float)

    # 测试无效类型
    value1 = {"type": "invalid_type", "value": "dog"}
    value2 = {"type": "text", "value": "cat"}
    score = calculate_semantic_score(value1, value2)
    assert score == 0.1


def test_get_device_type():
    with app.app_context():
        with app.test_request_context():
            result = get_device_type()
            assert isinstance(result, (type(None), str))


def test_get_geo_location():
    ip = "127.0.0.1"
    result = get_geo_location(ip)
    assert isinstance(result, (type(None), str))

    # 测试无效 IP
    ip = "invalid_ip"
    result = get_geo_location(ip)
    assert isinstance(result, (type(None), str))


def test_encode_context():
    with app.app_context():
        with app.test_request_context():
            result = encode_context()
            # 检查 encode_context 函数的返回值类型，确保符合预期
            assert isinstance(result, (type(None), np.ndarray))


def test_base64_to_opencv():
    # 使用有效的 Base64 编码字符串
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', test_image)
    base64_string = base64.b64encode(buffer).decode('utf-8')
    result = base64_to_opencv(base64_string)
    assert isinstance(result, (type(None), np.ndarray))

    # 测试无效 Base64 字符串
    base64_string = "invalid_base64_string"
    result = base64_to_opencv(base64_string)
    assert isinstance(result, (type(None), np.ndarray))


def test_extract_image_features():
    base64_image = "test_base64_image"
    result = extract_image_features(base64_image)
    assert isinstance(result, (type(None), torch.Tensor))

    # 测试无效 Base64 图像
    base64_image = "invalid_base64_image"
    result = extract_image_features(base64_image)
    assert isinstance(result, (type(None), torch.Tensor))


def test_handle_conflicts():
    data = {
        "base_prompt": "A dog on the grass",
        "attributes": [
            {
                "name": "object",
                "type": "text",
                "value": "dog",
                "initial_weight": 0.8
            },
            {
                "name": "background",
                "type": "text",
                "value": "grass",
                "initial_weight": 0.2
            }
        ]
    }
    conflicts = load_conflicts()
    fallback_strategy = "balanced"
    temperature = 1.2
    final_weights, conflict_report, adjustment_log = handle_conflicts(data, conflicts, fallback_strategy, temperature)
    assert isinstance(final_weights, (dict, type(None)))
    assert isinstance(conflict_report, dict)
    assert isinstance(adjustment_log, list)

    # 测试 strict 策略下的冲突
    conflicts = {"conflict_cases": [{"name": "dog+grass"}]}
    fallback_strategy = "strict"
    final_weights, conflict_report, adjustment_log = handle_conflicts(data, conflicts, fallback_strategy, temperature)
    assert final_weights is None

    # 测试 balanced 策略下的冲突
    conflicts = {"conflict_cases": [{"name": "dog+grass"}]}
    fallback_strategy = "balanced"
    final_weights, conflict_report, adjustment_log = handle_conflicts(data, conflicts, fallback_strategy, temperature)
    assert isinstance(final_weights, dict)

    # 测试 creative 策略下的冲突
    conflicts = {"conflict_cases": [{"name": "dog+grass"}]}
    fallback_strategy = "creative"
    final_weights, conflict_report, adjustment_log = handle_conflicts(data, conflicts, fallback_strategy, temperature)
    assert isinstance(final_weights, dict)



def test_calculate_weights():
    with original_app.test_client() as client:
        # 正常请求
        with patch('requests.post') as mock_post:
            mock_response = mock_post.return_value
            mock_response.status_code = 200
            mock_response.json.return_value = {"code": 200, "data": {"final_weights": {}}}

            with patch('utils.dynamic_weights.verify_signature', return_value=True):
                with patch('utils.dynamic_weights.rate_limit', return_value=True):
                    data = {
                        "base_prompt": "A dog on the grass",
                        "attributes": [
                            {
                                "name": "object",
                                "type": "text",
                                "value": "dog",
                                "initial_weight": 0.8
                            },
                            {
                                "name": "background",
                                "type": "text",
                                "value": "grass",
                                "initial_weight": 0.2
                            }
                        ],
                        "temperature": 1.2,
                        "fallback_strategy": "balanced",
                        "debug_mode": False
                    }
                    headers = {
                        'Content-Type': 'application/json',
                        'X-Api-Key': 'valid_api_key',
                        'X-Timestamp': 'timestamp',
                        'X-Signature': 'signature'
                    }
                    response = client.post('/api/v5/weight/calculate', json=data, headers=headers)
                    assert response.status_code == 200
    