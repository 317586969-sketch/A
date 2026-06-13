from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI()

# 身份验证令牌（个人使用，简单安全）
VALID_TOKEN = "itcast"

# 城市编码数据（直接硬编码，无需外部文件）
CITY_CODES = {
    "北京": "101010300",  # 朝阳区（免费Key可用）
    "上海": "101020200",  # 闵行区（免费Key可用）
    "广州": "101280101",  # 广州（保持原样）
    "深圳": "101280601",  # 深圳（保持原样）
    "杭州": "101210101",  # 杭州（保持原样）
    "成都": "101270101"   # 成都（保持原样）
}

class Location(BaseModel):
    location: str

@app.post("/luci.itcast/weather")
async def get_weather(request: Request, body: Location):
    # 1. 验证身份令牌
    auth_header = request.headers.get("Authorization")
    if auth_header != f"Bearer {VALID_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid Authorization header")

    location = body.location

    # 2. 获取城市编码
    city_code = CITY_CODES.get(location)
    if not city_code:
        return {
            "status": "error",
            "message": f"错误：请确保输入的【{location}】属于支持的城市，目前支持的城市：{', '.join(CITY_CODES.keys())}"
        }

    # 3. 调度天气API（已注入您刚刚成功申请的 Web API 凭据）
    QWEATHER_KEY = "40dfccbd73af43ccb3fb467d9af6b7b2"
    url = f"https://devapi.qweather.com/v7/weather/3d?location={city_code}&key={QWEATHER_KEY}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        weather_data = response.json()
    except Exception as e:
        return {"status": "error", "message": f"天气服务通信失败: {str(e)}"}

    # 4. 解析天气数据
    try:
        forecast = weather_data["daily"][0]
        weather_type = forecast["textDay"]
        # 数据整理格式化
        forecast['high'] = forecast['tempMax'].replace("℃", "")
        forecast['low'] = forecast['tempMin'].replace("℃", "")
        temperature = f"{forecast['low']} - {forecast['high']}℃"
    except (KeyError, IndexError) as e:
        return {"status": "error", "message": f"天气数据解析失败: {str(e)}"}

    # 5. 组装自动响应格式
    return {
        "status": "success",
        "data": {
            "location": location,
            "weather": weather_type,
            "temperature": temperature
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)