#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用requests库直接测试Reddit API认证
"""

import requests
import base64
import os
import pytest
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def test_reddit_auth():
    """直接测试Reddit API认证"""
    print("直接测试Reddit API认证...")
    
    # 从环境变量获取Reddit API凭据
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    
    print(f"Client ID: {client_id}")
    print(f"Client Secret: {client_secret[:5]}...{client_secret[-5:]}" if client_secret else "Client Secret: None")
    
    if not client_id or not client_secret:
        print("错误：未找到Reddit API凭据")
        pytest.skip("Reddit API credentials not configured")
        return
    
    # 准备认证数据
    auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
    data = {'grant_type': 'client_credentials'}
    headers = {'User-Agent': 'news_agent by u/yourusername'}
    
    # 发送认证请求
    print("正在发送认证请求...")
    response = requests.post(
        'https://www.reddit.com/api/v1/access_token',
        auth=auth,
        data=data,
        headers=headers
    )
    
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.text}")
    
    if response.status_code == 200:
        print("认证成功!")
        token_data = response.json()
        print(f"Token类型: {token_data.get('token_type')}")
        print(f"过期时间: {token_data.get('expires_in')}秒")
    else:
        print("认证失败!")

if __name__ == "__main__":
    test_reddit_auth()