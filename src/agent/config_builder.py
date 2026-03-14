from datetime import datetime, timedelta

def build_config(parsed_needs: dict) -> dict:
    """
    將解析後的需求轉化為標準的研究設定 (Config).
    """
    # 決定時間範圍 (預設近 3 年以獲得穩定資料)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 3)
    
    config = {
        "ticker": parsed_needs.get("stock_id", "2330"),
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "task_type": parsed_needs.get("task_type", "regression"),
        "horizon": parsed_needs.get("horizon", 5),
        "model_name": parsed_needs.get("preferred_models", ["linear_regression"])[0],
        "features": {
            "use_price": "price" in parsed_needs.get("feature_groups", []),
            "use_volume": "volume" in parsed_needs.get("feature_groups", []),
            "use_technical": "technical" in parsed_needs.get("feature_groups", []),
            "use_fundamental": "fundamental" in parsed_needs.get("feature_groups", [])
        },
        "validation_method": parsed_needs.get("validation_method", "time_split"),
        "meta": {
            "original_prompt": parsed_needs.get("user_goal", ""),
            "retry_count": 0
        }
    }
    
    # 確保至少有一種特徵
    if not any(config["features"].values()):
        config["features"]["use_price"] = True
        config["features"]["use_technical"] = True
        
    return config
