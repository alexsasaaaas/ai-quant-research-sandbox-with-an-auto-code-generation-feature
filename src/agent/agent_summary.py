def generate_agent_summary(result: dict) -> str:
    """
    將 Agent 執行結果 (Code Interpreter 模式) 轉化為摘要。
    """
    retry_summary = result.get("retry_mgr").get_summary()
    
    if result["success"]:
        summary = f"""
### 🤖 Agent 執行摘要 (Code Interpreter)
- **最終狀態**: ✅ 成功
- **執行模型**: `{result['model_name']}` ({result['task_type']})
- **預測標的**: `{result['ticker']}` (Horizon: {result['horizon']}d)

#### 🔄 修補與重試歷程
{retry_summary}

> **Agent 註解**: 系統已根據您的需求動態生成並執行了專屬研究程式碼。
        """
    else:
        summary = f"""
### 🤖 Agent 執行摘要 (Code Interpreter)
- **最終狀態**: ❌ 失敗
- **錯誤原因**: `{result['error']}`

#### 🔄 修補與重試歷程
{retry_summary}
        """
    return summary
