import json
from utils.dynamic_weights import validate_request, adjust_weights, detect_conflict, load_conflicts, \
    calculate_semantic_score, verify_signature, rate_limit
from unittest.mock import patch
import requests

def test_validate_request():
    # 正常情况
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

    # 缺少 base_prompt 或 attributes
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

    # type 字段错误
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


def test_calculate_semantic_score():
    value1 = {"type": "text", "value": "dog"}
    value2 = {"type": "text", "value": "cat"}
    score = calculate_semantic_score(value1, value2)
    assert isinstance(score, float)


@patch('utils.dynamic_weights.request')
def test_verify_signature(mock_request):
    mock_request.headers.get.side_effect = ["timestamp", "signature"]
    mock_request.get_data.return_value = b"data"
    result = verify_signature(mock_request)
    assert isinstance(result, bool)


@patch('utils.dynamic_weights.redis_client')
@patch('utils.dynamic_weights.request')
def test_rate_limit(mock_request, mock_redis):
    mock_request.remote_addr = "127.0.0.1"
    mock_redis.incr.return_value = 1
    result = rate_limit(mock_request)
    assert isinstance(result, bool)