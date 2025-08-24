"""
Simple test to see if water agent detects form data
"""
import json
from app.llm.agents.water_agent import water_executor

def test_form_detection():
    """Test if agent detects form data and calls verification tool"""
    
    # Simple test with obvious form data
    test_input = """Please verify this form: {"formFields": [{"data_id": "test", "fieldValue": "test"}]}"""
    
    try:
        result = water_executor.invoke({
            "input": test_input
        })
        
        print("Water Agent Response:")
        print("=" * 50)
        print(result.get("output", "No output"))
        print("=" * 50)
        
        # Also print intermediate steps if available
        if "intermediate_steps" in result:
            print("\nIntermediate Steps:")
            for step in result["intermediate_steps"]:
                print(f"Action: {step[0].tool}")
                print(f"Input: {step[0].tool_input}")
                print(f"Output: {step[1]}")
                print("-" * 30)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_form_detection()
