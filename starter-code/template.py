"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.
"""

import json
from tools import TOOL_DEFINITIONS, TOOL_MAP, get_flight_info, get_weather_forecast

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
"""

class ChatbotBaseline:
    """Baseline LLM Chatbot (Không sử dụng ReAct Loop hay Tools)"""
    def query(self, user_input: str) -> dict:
        return {
            "status": "success",
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": []
        }


class ReActAgent:
    """ReAct Agent có sử dụng Thought-Action-Observation Loop"""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace = []

    def _plan_actions(self, user_input: str):
        """Giả lập bước 'Thought' của LLM: quyết định danh sách Action cần
        thực hiện dựa trên nội dung câu hỏi khách hàng."""
        user_lower = user_input.lower()
        wants_flight = "chuyến bay" in user_lower or "vé" in user_lower
        wants_weather = "thời tiết" in user_lower or "mặc gì" in user_lower

        actions = []

        if wants_flight and "han" in user_lower and "sgn" in user_lower:
            actions.append({
                "name": "get_flight_info",
                "args": {"origin": "HAN", "destination": "SGN", "max_price": 2000000}
            })
            if wants_weather:
                actions.append({
                    "name": "get_weather_forecast",
                    "args": {"city_code": "SGN"}
                })
        elif wants_flight and "han" in user_lower and "dad" in user_lower:
            actions.append({
                "name": "get_flight_info",
                "args": {"origin": "HAN", "destination": "DAD", "max_price": 1500000}
            })
        elif wants_weather and ("dad" in user_lower or "đà nẵng" in user_lower):
            actions.append({
                "name": "get_weather_forecast",
                "args": {"city_code": "DAD"}
            })

        return actions

    def _thought_for(self, action: dict) -> str:
        if action["name"] == "get_flight_info":
            args = action["args"]
            return (
                f"Cần tìm chuyến bay từ {args['origin']} đến {args['destination']} "
                f"dưới {args['max_price']:,} VND."
            )
        if action["name"] == "get_weather_forecast":
            return f"Cần kiểm tra thời tiết tại {action['args']['city_code']}."
        return "Cần thực hiện bước tiếp theo để trả lời khách hàng."

    def _execute_action(self, raw_action: dict):
        """Thực thi Action qua TOOL_MAP, xử lý 2 bẫy thường gặp:
        - Trap 1: tên tool bị thừa khoảng trắng / sai hoa thường.
        - Trap 2: Action không đúng định dạng JSON."""
        try:
            action = json.loads(json.dumps(raw_action))
        except (TypeError, ValueError):
            return raw_action, {"error": "Invalid JSON format"}

        tool_name = str(action.get("name", "")).strip().lower()
        tool_fn = TOOL_MAP.get(tool_name)

        if tool_fn is None:
            return action, {"error": f"Unknown tool: {tool_name}"}

        observation = tool_fn(**action.get("args", {}))
        return action, observation

    def _final_answer(self, user_input: str, observations: dict) -> str:
        flights = observations.get("get_flight_info")
        weather = observations.get("get_weather_forecast")

        if flights is not None and weather is not None:
            flight_text = ", ".join(
                f"{f['flight_number']} ({f['price_vnd']:,} VND)" for f in flights
            )
            return (
                f"Chuyến bay HAN → SGN dưới 2 triệu gồm: {flight_text}. "
                f"Thời tiết tại {weather['city']} hiện là "
                f"{weather['temperature_c']}°C, {weather['condition']}. "
                f"Khuyến nghị: {weather['recommendation']}"
            )

        if flights is not None:
            return f"Các chuyến bay tìm được: {flights}"

        if weather is not None:
            return (
                f"Thời tiết tại {weather['city']} hiện là "
                f"{weather['temperature_c']}°C, {weather['condition']}. "
                f"{weather['recommendation']}"
            )

        return f"Thông tin về chính sách của Vinpearl: {user_input}"

    def run(self, user_input: str) -> dict:
        self.trace = []
        actions = self._plan_actions(user_input)

        # -----------------------------------------------------
        # FAQ: không cần dùng tool
        # -----------------------------------------------------
        if not actions:
            self.trace.append({
                "iteration": 1,
                "thought": "Đây là câu hỏi FAQ, không cần sử dụng tool.",
                "action": None,
                "observation": None
            })
            return {
                "status": "completed",
                "iterations": 1,
                "trace": self.trace,
                "answer": self._final_answer(user_input, {})
            }

        observations = {}
        consecutive_errors = 0
        iteration = 0

        while iteration < len(actions):
            # Milestone 4: chặn vòng lặp vô tận khi vượt quá max_iterations
            if iteration >= self.max_iterations:
                return {
                    "status": "max_iterations_reached",
                    "iterations": iteration,
                    "trace": self.trace,
                    "answer": self._final_answer(user_input, observations)
                }

            planned = actions[iteration]
            thought = self._thought_for(planned)
            action, observation = self._execute_action(planned)
            iteration += 1

            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "action": action,
                "observation": observation
            })

            # Trap 3: nếu tool báo lỗi liên tiếp, dừng sớm thay vì lặp vô hạn
            if isinstance(observation, dict) and "error" in observation:
                consecutive_errors += 1
                if consecutive_errors >= 2:
                    return {
                        "status": "completed",
                        "iterations": iteration,
                        "trace": self.trace,
                        "answer": f"Xin lỗi, hệ thống gặp lỗi khi tra cứu: {observation['error']}"
                    }
            else:
                consecutive_errors = 0
                observations[action["name"]] = observation

        if iteration >= self.max_iterations:
            return {
                "status": "max_iterations_reached",
                "iterations": iteration,
                "trace": self.trace,
                "answer": self._final_answer(user_input, observations)
            }

        # Nếu có nhiều hơn 1 bước tool, cần thêm bước suy luận tổng hợp câu trả lời
        if len(actions) > 1:
            iteration += 1
            self.trace.append({
                "iteration": iteration,
                "thought": "Đã có đủ thông tin, tổng hợp câu trả lời cho khách hàng.",
                "action": None,
                "observation": None
            })

        return {
            "status": "completed",
            "iterations": iteration,
            "trace": self.trace,
            "answer": self._final_answer(user_input, observations)
        }


def main():
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result)
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
