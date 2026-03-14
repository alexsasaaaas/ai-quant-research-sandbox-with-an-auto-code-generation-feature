import logging
import traceback
from src.agent.retry_manager import RetryManager
from src.agent.auto_debugger import repair_code_from_error

logger = logging.getLogger(__name__)

def run_agent_workflow(generated_code: str) -> dict:
    """
    執行由 Agent 生成的程式碼，並捕捉結果與錯誤。
    """
    retry_mgr = RetryManager(max_retries=15)
    current_code = generated_code
    
    from src.agent.dependency_mgr import install_package
    
    while True:
        # 建立一個乾淨的 namespace
        exec_globals = {
            "install_package": install_package,
            "logger": logger,
        }
        
        try:
            # 執行生成的程式碼
            # 程式碼最後應將結果存入一個名為 'result' 的變數中
            exec(current_code, exec_globals)
            
            if "result" not in exec_globals:
                raise ValueError("產出的程式碼執行後未定義 'result' 變數。")
                
            res = exec_globals["result"]
            res["code"] = current_code
            res["retry_mgr"] = retry_mgr
            return res
            
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.warning(f"Generated code execution failed:\n{error_trace}")
            
            # 嘗試從 globals 抓取資料狀態 (如果有的話)
            context = {}
            if "df_final" in exec_globals:
                context["df_shape"] = exec_globals["df_final"].shape
            if "feat_cols" in exec_globals:
                context["n_features"] = len(exec_globals["feat_cols"])
                context["feature_names"] = exec_globals["feat_cols"][:5]

            attempt = retry_mgr.add_attempt({"code": current_code}, error=str(e))
            
            if retry_mgr.should_retry():
                # 傳遞完整的 Traceback 與資料 Context 以供修復
                new_code, fix = repair_code_from_error(
                    current_code, 
                    error_trace, 
                    len(retry_mgr.attempts),
                    context=context
                )
                attempt["fix_applied"] = fix
                current_code = new_code
                logger.info(f"Retrying with code fix: {fix}")
            else:
                return {
                    "success": False,
                    "error": str(e),
                    "code": current_code,
                    "retry_mgr": retry_mgr
                }
